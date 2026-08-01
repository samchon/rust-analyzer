//! Resident HIR-backed graph snapshot export.

use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex as StdMutex},
    time::Instant,
};

use ide::{
    AssistResolveStrategy, FileId, FileRange, Severity, StaticIndex, StaticReferenceRole,
    StaticRelationKind, SymbolInformationKind, VendoredLibrariesConfig,
};
use notify::{Config as NotifyConfig, EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use project_model::ProjectWorkspaceKind;
use rustc_hash::FxHashSet;
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use syntax::{AstNode, TextRange, ast};
use toolchain::Tool;
use walkdir::WalkDir;

use crate::{
    global_state::GlobalStateSnapshot,
    lsp::to_proto,
    lsp_ext::{
        GraphSnapshotCheckpoint, GraphSnapshotCoverage, GraphSnapshotDiagnostic, GraphSnapshotEdge,
        GraphSnapshotEvidence, GraphSnapshotManifestEntry, GraphSnapshotNode, GraphSnapshotParams,
        GraphSnapshotPhases, GraphSnapshotProducer, GraphSnapshotResult, GraphSnapshotShard,
        GraphSnapshotUniverse, GraphSnapshotUnresolved,
    },
};

const PROTOCOL_VERSION: u32 = 1;
const SCHEMA_VERSION: u32 = 1;
const MAX_SYMLINK_HOPS: usize = 64;
const COVERAGE: [(&str, &str); 15] = [
    ("contains", "partial"),
    ("exports", "partial"),
    ("imports", "partial"),
    ("calls", "partial"),
    ("accesses", "partial"),
    ("instantiates", "partial"),
    ("type_ref", "partial"),
    ("extends", "partial"),
    ("implements", "partial"),
    ("overrides", "partial"),
    ("dispatches", "partial"),
    ("decorates", "partial"),
    ("renders", "unsupported"),
    ("tests", "partial"),
    ("references", "partial"),
];

pub(crate) struct GraphSnapshotCache {
    revision: u64,
    sequence: u64,
    committed: Option<CachedSnapshot>,
    dirty_files: FxHashSet<FileId>,
    full_rebuild: bool,
    external_inputs: String,
    external_input_roots: Vec<ExternalGraphInput>,
    external_input_digest: Result<String, String>,
    external_input_revision: u64,
    external_inputs_dirty: bool,
    external_input_watcher: Option<RecommendedWatcher>,
    external_input_event_fence: Arc<StdMutex<u64>>,
    external_input_event_epoch: u64,
    external_input_watcher_stale: bool,
    source_input_watcher: Option<RecommendedWatcher>,
    source_input_event_fence: Arc<StdMutex<u64>>,
    source_input_event_epoch: u64,
    source_input_watcher_stale: bool,
}

impl Default for GraphSnapshotCache {
    fn default() -> Self {
        Self {
            revision: 0,
            sequence: 0,
            committed: None,
            dirty_files: FxHashSet::default(),
            full_rebuild: true,
            external_inputs: String::new(),
            external_input_roots: Vec::new(),
            external_input_digest: Ok(digest_bytes(&[])),
            external_input_revision: 0,
            external_inputs_dirty: true,
            external_input_watcher: None,
            external_input_event_fence: Arc::new(StdMutex::new(0)),
            external_input_event_epoch: 0,
            external_input_watcher_stale: true,
            source_input_watcher: None,
            source_input_event_fence: Arc::new(StdMutex::new(0)),
            source_input_event_epoch: 0,
            source_input_watcher_stale: true,
        }
    }
}

struct CachedSnapshot {
    revision: u64,
    producer: GraphSnapshotProducer,
    universe: GraphSnapshotUniverse,
    sequence: u64,
    generation: String,
    manifest: Vec<GraphSnapshotManifestEntry>,
    phases: GraphSnapshotPhases,
    shards: BTreeMap<String, Arc<GraphSnapshotShard>>,
    interface_fingerprints: BTreeMap<String, String>,
    source_inputs: BTreeMap<FileId, String>,
    node_owners: BTreeMap<String, String>,
    delta_base: Option<String>,
    delta_upserts: Vec<Arc<GraphSnapshotShard>>,
    delta_deletes: Vec<String>,
}

struct SnapshotResponsePlan {
    producer: GraphSnapshotProducer,
    universe: GraphSnapshotUniverse,
    sequence: u64,
    generation: String,
    base_generation: Option<String>,
    upserts: Vec<Arc<GraphSnapshotShard>>,
    deletes: Vec<String>,
    manifest: Vec<GraphSnapshotManifestEntry>,
    phases: GraphSnapshotPhases,
}

impl GraphSnapshotCache {
    pub(crate) fn revision(&self) -> u64 {
        self.revision
    }

    pub(crate) fn invalidate_all(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.dirty_files.clear();
        self.full_rebuild = true;
        self.source_input_watcher_stale = true;
    }

    fn invalidate_universe(&mut self) {
        self.invalidate_all();
        self.committed = None;
    }

    pub(crate) fn invalidate_files(&mut self, files: &[FileId]) {
        self.revision = self.revision.wrapping_add(1);
        if self.committed.is_none() {
            self.full_rebuild = true;
            return;
        }
        if !self.full_rebuild {
            self.dirty_files.extend(files.iter().copied());
        }
    }

    fn current(&self) -> Option<&CachedSnapshot> {
        let cached = self.committed.as_ref()?;
        if cached.revision != self.revision {
            return None;
        }
        Some(cached)
    }
}

pub(crate) fn handle(
    snap: GlobalStateSnapshot,
    params: GraphSnapshotParams,
) -> anyhow::Result<GraphSnapshotResult> {
    if snap.workspaces.is_empty() {
        return Err(retry_error(
            "graph snapshot is unavailable until the workspace has loaded; retry",
        ));
    }
    let started = Instant::now();
    ensure_external_inputs(&snap)?;
    let observed_universe = universe(&snap)?;
    let cached_response = {
        let mut cache = snap.graph_snapshot_cache.lock();
        let external_fence = cache.external_input_event_fence.clone();
        let external_guard = external_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let source_fence = cache.source_input_event_fence.clone();
        let source_guard = source_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        drain_external_input_events_at(&mut cache, *external_guard);
        drain_source_input_events_at(&mut cache, *source_guard);
        if cache.committed.as_ref().is_some_and(|cached| cached.universe != observed_universe) {
            cache.invalidate_universe();
        }
        cache
            .current()
            .map(|cached| response_plan(cached, params.known_generation.as_deref(), true))
    };
    if let Some(response) = cached_response {
        let mut result = response.into_result();
        result.phases.total_millis = elapsed_millis(started);
        return Ok(result);
    }
    validate_project_model_inputs(&snap)?;
    if let Some(checkpoint) = params.checkpoint.as_ref() {
        let source_files =
            StaticIndex::graph_source_files(&snap.analysis, VendoredLibrariesConfig::Excluded);
        let source_epoch = ensure_source_input_watcher(&snap, &source_files)?;
        let source_inputs = validate_source_inputs(&snap, &source_files)?;
        let mut result = checkpoint_response(
            &snap,
            &observed_universe,
            checkpoint,
            &params,
            &source_inputs,
            source_epoch,
        )?;
        result.phases.total_millis = elapsed_millis(started);
        return Ok(result);
    }
    let (
        revision,
        dirty_files,
        requested_full_rebuild,
        cached_universe,
        cached_interface_fingerprints,
        cached_source_inputs,
        cached_node_owners,
    ) = {
        let cache = snap.graph_snapshot_cache.lock();
        (
            snap.graph_snapshot_revision,
            cache.dirty_files.iter().copied().collect::<Vec<_>>(),
            cache.full_rebuild,
            cache.committed.as_ref().map(|cached| cached.universe.clone()),
            cache
                .committed
                .as_ref()
                .map(|cached| cached.interface_fingerprints.clone())
                .unwrap_or_default(),
            cache.committed.as_ref().map(|cached| cached.source_inputs.clone()).unwrap_or_default(),
            cache.committed.as_ref().map(|cached| cached.node_owners.clone()).unwrap_or_default(),
        )
    };

    let semantic_started = Instant::now();
    let mut full_rebuild = requested_full_rebuild;
    let mut index = if full_rebuild {
        StaticIndex::compute_graph(&snap.analysis, VendoredLibrariesConfig::Excluded)
    } else {
        StaticIndex::compute_graph_files(&snap.analysis, &dirty_files)
    };
    if full_rebuild && index.files.is_empty() {
        return Err(retry_error(
            "graph snapshot is unavailable until the crate graph has loaded; retry",
        ));
    }
    let mut interface_fingerprints = compute_interface_fingerprints(&snap, &index)?;
    if !full_rebuild {
        let dirty_sources =
            dirty_files.iter().map(|&file_id| source_path(&snap, file_id)).collect::<BTreeSet<_>>();
        if interfaces_changed(
            &cached_interface_fingerprints,
            &interface_fingerprints,
            &dirty_sources,
        ) {
            full_rebuild = true;
            index = StaticIndex::compute_graph(&snap.analysis, VendoredLibrariesConfig::Excluded);
            if index.files.is_empty() {
                return Err(retry_error(
                    "graph snapshot is unavailable while the crate graph is reloading; retry",
                ));
            }
            interface_fingerprints = compute_interface_fingerprints(&snap, &index)?;
        } else {
            let mut merged = cached_interface_fingerprints;
            for source in &dirty_sources {
                merged.remove(source);
            }
            merged.extend(interface_fingerprints);
            interface_fingerprints = merged;
        }
    }
    let semantic_millis = elapsed_millis(semantic_started);

    let indexed_files = index.files.iter().map(|file| file.file_id).collect::<Vec<_>>();
    let source_epoch = if full_rebuild {
        ensure_source_input_watcher(&snap, &indexed_files)?
    } else {
        current_source_input_epoch(&snap)?
    };
    let validated_source_inputs = validate_source_inputs(&snap, &indexed_files)?;
    let mut source_inputs = if full_rebuild { BTreeMap::new() } else { cached_source_inputs };
    source_inputs.extend(validated_source_inputs);
    let dirty_sources =
        indexed_files.iter().map(|&file_id| source_path(&snap, file_id)).collect::<BTreeSet<_>>();
    let mut retained_node_owners = if full_rebuild { BTreeMap::new() } else { cached_node_owners };
    retained_node_owners.retain(|_, source| !dirty_sources.contains(source));

    let shard_started = Instant::now();
    let snapshot_universe = if requested_full_rebuild {
        observed_universe
    } else {
        let cached = cached_universe
            .ok_or_else(|| anyhow::format_err!("incremental graph snapshot lost its universe"))?;
        if cached != observed_universe {
            return Err(retry_error("graph snapshot universe moved; retry the request"));
        }
        cached
    };
    let built = build_shards(
        &snap,
        index,
        &snapshot_universe,
        &interface_fingerprints,
        &source_inputs,
        retained_node_owners,
    )?;
    let shard_millis = elapsed_millis(shard_started);

    let encode_started = Instant::now();
    ensure_external_inputs(&snap)?;
    validate_project_model_inputs(&snap)?;
    if universe(&snap)? != snapshot_universe {
        snap.graph_snapshot_cache.lock().invalidate_all();
        return Err(retry_error(
            "graph snapshot universe moved while it was being built; retry the request",
        ));
    }
    let mut cache = snap.graph_snapshot_cache.lock();
    let external_fence = cache.external_input_event_fence.clone();
    let external_guard = external_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let source_fence = cache.source_input_event_fence.clone();
    let source_guard = source_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    drain_external_input_events_at(&mut cache, *external_guard);
    if *source_guard != source_epoch {
        cache.invalidate_all();
        return Err(retry_error(
            "graph snapshot source inputs moved while it was being built; retry the request",
        ));
    }
    ensure_cache_revision(
        &cache,
        revision,
        "graph snapshot was invalidated while it was being built; retry the request",
    )?;
    if let Some(committed) = cache.current() {
        let mut response = response_plan(committed, params.known_generation.as_deref(), true);
        response.phases = GraphSnapshotPhases {
            semantic_millis,
            shard_millis,
            encode_millis: elapsed_millis(encode_started),
            total_millis: elapsed_millis(started),
            cache_hit: true,
        };
        drop(cache);
        drop(source_guard);
        drop(external_guard);
        return Ok(response.into_result());
    }
    let mut merged = if full_rebuild {
        BTreeMap::new()
    } else {
        let dirty_sources =
            dirty_files.iter().map(|&file_id| source_path(&snap, file_id)).collect::<BTreeSet<_>>();
        let mut merged =
            cache.committed.as_ref().map(|cached| cached.shards.clone()).unwrap_or_default();
        merged.retain(|_, shard| !dirty_sources.contains(&shard.source));
        merged
    };
    for shard in built.shards {
        merged.insert(shard.key.clone(), Arc::new(shard));
    }

    let old_manifest = cache
        .committed
        .as_ref()
        .map(|cached| {
            cached
                .manifest
                .iter()
                .map(|entry| (entry.key.as_str(), entry.digest.as_str()))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let manifest = merged
        .values()
        .map(|shard| GraphSnapshotManifestEntry {
            key: shard.key.clone(),
            digest: shard.digest.clone(),
        })
        .collect::<Vec<_>>();
    let new_keys = manifest.iter().map(|entry| entry.key.as_str()).collect::<BTreeSet<_>>();
    let delta_upserts = merged
        .values()
        .filter(|shard| {
            old_manifest.get(shard.key.as_str()).copied() != Some(shard.digest.as_str())
        })
        .cloned()
        .collect::<Vec<_>>();
    let delta_deletes = old_manifest
        .keys()
        .filter(|key| !new_keys.contains(**key))
        .map(|key| (*key).to_owned())
        .collect::<Vec<_>>();
    let base_generation = cache.committed.as_ref().map(|cached| cached.generation.clone());
    let generation = snapshot_generation(&snapshot_universe.digest, &manifest)?;
    cache.sequence = cache.sequence.wrapping_add(1);
    let sequence = cache.sequence;
    let producer = producer();
    let encode_millis = elapsed_millis(encode_started);
    let phases = GraphSnapshotPhases {
        semantic_millis,
        shard_millis,
        encode_millis,
        total_millis: elapsed_millis(started),
        cache_hit: false,
    };
    let cached = CachedSnapshot {
        revision,
        producer,
        universe: snapshot_universe,
        sequence,
        generation,
        manifest,
        phases,
        shards: merged,
        interface_fingerprints,
        source_inputs,
        node_owners: built.node_owners,
        delta_base: base_generation,
        delta_upserts,
        delta_deletes,
    };
    let response = response_plan(&cached, params.known_generation.as_deref(), false);
    cache.committed = Some(cached);
    cache.dirty_files.clear();
    cache.full_rebuild = false;
    drop(cache);
    drop(source_guard);
    drop(external_guard);
    Ok(response.into_result())
}

fn checkpoint_response(
    snap: &GlobalStateSnapshot,
    snapshot_universe: &GraphSnapshotUniverse,
    checkpoint: &GraphSnapshotCheckpoint,
    params: &GraphSnapshotParams,
    source_inputs: &BTreeMap<FileId, String>,
    source_epoch: u64,
) -> anyhow::Result<GraphSnapshotResult> {
    if checkpoint.protocol_version != PROTOCOL_VERSION
        || checkpoint.schema_version != SCHEMA_VERSION
        || checkpoint.producer != producer()
        || checkpoint.universe != snapshot_universe.digest
        || params.known_generation.as_deref() != Some(checkpoint.generation.as_str())
    {
        return Err(retry_error(
            "persisted graph checkpoint does not match this producer universe; rebuild and retry",
        ));
    }
    let expected_sources = source_inputs
        .iter()
        .map(|(&file_id, digest)| (source_path(snap, file_id), digest.clone()))
        .collect::<BTreeMap<_, _>>();
    let actual_sources = checkpoint
        .sources
        .iter()
        .map(|source| (source.source.clone(), source.checker_digest.clone()))
        .collect::<BTreeMap<_, _>>();
    if actual_sources.len() != checkpoint.sources.len() || actual_sources != expected_sources {
        return Err(retry_error(
            "persisted graph checkpoint source manifest moved; rebuild and retry",
        ));
    }
    let mut manifest = checkpoint.manifest.clone();
    manifest.sort_by(|left, right| left.key.cmp(&right.key));
    let manifest_count = manifest.len();
    manifest.dedup_by(|left, right| left.key == right.key);
    let manifest_by_key = manifest
        .iter()
        .map(|entry| (entry.key.as_str(), entry.digest.as_str()))
        .collect::<BTreeMap<_, _>>();
    let mut shards = BTreeMap::new();
    let mut interface_fingerprints = BTreeMap::new();
    let mut node_owners = BTreeMap::new();
    for shard in &checkpoint.shards {
        let expected_key = format!("{}\0{}", snapshot_universe.target, shard.source);
        let valid = shard.key == expected_key
            && expected_sources.get(&shard.source) == Some(&shard.checker_digest)
            && is_digest(&shard.interface_fingerprint)
            && manifest_by_key.get(shard.key.as_str()).copied() == Some(shard.digest.as_str())
            && shard_digest(
                &shard.key,
                &shard.source,
                &shard.checker_digest,
                &shard.interface_fingerprint,
                &shard.nodes,
                &shard.edges,
                &shard.diagnostics,
                &shard.coverage,
                &shard.unresolved,
            )? == shard.digest;
        if !valid
            || shards.insert(shard.key.clone(), Arc::new(shard.clone())).is_some()
            || interface_fingerprints
                .insert(shard.source.clone(), shard.interface_fingerprint.clone())
                .is_some()
            || shard
                .nodes
                .iter()
                .any(|node| node_owners.insert(node.id.clone(), shard.source.clone()).is_some())
        {
            return Err(retry_error(
                "persisted graph checkpoint shards are invalid; rebuild and retry",
            ));
        }
    }
    if manifest.len() != manifest_count
        || manifest.len() != expected_sources.len()
        || checkpoint.shards.len() != expected_sources.len()
        || !manifest.iter().all(|entry| is_digest(&entry.digest))
        || snapshot_generation(&snapshot_universe.digest, &manifest)? != checkpoint.generation
    {
        return Err(retry_error(
            "persisted graph checkpoint manifest is invalid; rebuild and retry",
        ));
    }
    ensure_external_inputs(snap)?;
    validate_project_model_inputs(snap)?;
    if universe(snap)? != *snapshot_universe {
        return Err(retry_error(
            "graph snapshot universe moved while validating a checkpoint; retry",
        ));
    }
    let source_files = source_inputs.keys().copied().collect::<Vec<_>>();
    if validate_source_inputs(snap, &source_files)? != *source_inputs {
        return Err(retry_error(
            "graph snapshot source inputs moved while validating a checkpoint; retry",
        ));
    }
    let mut cache = snap.graph_snapshot_cache.lock();
    let external_fence = cache.external_input_event_fence.clone();
    let external_guard = external_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let source_fence = cache.source_input_event_fence.clone();
    let source_guard = source_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    drain_external_input_events_at(&mut cache, *external_guard);
    if *source_guard != source_epoch {
        cache.invalidate_all();
        return Err(retry_error(
            "graph snapshot source inputs moved while validating a checkpoint; retry",
        ));
    }
    if let Some(committed) = cache.current() {
        return Ok(response_plan(committed, params.known_generation.as_deref(), true).into_result());
    }
    ensure_cache_revision(
        &cache,
        snap.graph_snapshot_revision,
        "graph snapshot was invalidated while validating a checkpoint; retry the request",
    )?;
    cache.sequence = cache.sequence.wrapping_add(1);
    let cached = CachedSnapshot {
        revision: snap.graph_snapshot_revision,
        producer: producer(),
        universe: snapshot_universe.clone(),
        sequence: cache.sequence,
        generation: checkpoint.generation.clone(),
        manifest,
        phases: GraphSnapshotPhases { cache_hit: true, ..GraphSnapshotPhases::default() },
        shards,
        interface_fingerprints,
        source_inputs: source_inputs.clone(),
        node_owners,
        delta_base: None,
        delta_upserts: Vec::new(),
        delta_deletes: Vec::new(),
    };
    let response = response_plan(&cached, params.known_generation.as_deref(), true).into_result();
    cache.committed = Some(cached);
    cache.dirty_files.clear();
    cache.full_rebuild = false;
    Ok(response)
}

fn is_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| matches!(byte, b'0'..=b'9' | b'a'..=b'f'))
}

fn ensure_cache_revision(
    cache: &GraphSnapshotCache,
    expected: u64,
    message: &str,
) -> anyhow::Result<()> {
    if cache.revision == expected { Ok(()) } else { Err(retry_error(message)) }
}

fn retry_error(message: &str) -> anyhow::Error {
    crate::lsp::LspError::new(lsp_server::ErrorCode::ServerCancelled as i32, message.to_owned())
        .into()
}

fn validate_project_model_inputs(snap: &GlobalStateSnapshot) -> anyhow::Result<()> {
    for workspace in snap.workspaces.iter() {
        for (path, captured) in &workspace.graph_project_inputs {
            let current = fs::read(path).ok();
            if current.as_deref() != captured.as_deref() {
                return Err(retry_error(&format!(
                    "{} moved beyond the immutable project model; reload and retry",
                    path
                )));
            }
        }
        if !matches!(
            workspace.kind,
            ProjectWorkspaceKind::Cargo { .. }
                | ProjectWorkspaceKind::DetachedFile { cargo: Some(_), .. }
        ) {
            continue;
        }
        let current = fs::read(workspace.workspace_root().join("Cargo.lock")).ok();
        if current.as_deref() != workspace.graph_lockfile.as_deref() {
            return Err(retry_error(
                "Cargo.lock moved beyond the immutable project model; reload and retry",
            ));
        }
    }
    Ok(())
}

fn validate_source_inputs(
    snap: &GlobalStateSnapshot,
    files: &[FileId],
) -> anyhow::Result<BTreeMap<FileId, String>> {
    files
        .iter()
        .copied()
        .map(|file_id| {
            let text = snap.analysis.file_text(file_id)?;
            Ok((file_id, checker_source_digest(snap, file_id, &text)?))
        })
        .collect()
}

fn ensure_source_input_watcher(
    snap: &GlobalStateSnapshot,
    files: &[FileId],
) -> anyhow::Result<u64> {
    {
        let cache = snap.graph_snapshot_cache.lock();
        if cache.source_input_watcher.is_some() && !cache.source_input_watcher_stale {
            drop(cache);
            return current_source_input_epoch(snap);
        }
    }
    let roots = files
        .iter()
        .map(|&file_id| {
            let file = snap.file_id_to_file_path(file_id);
            let path = file.as_path().ok_or_else(|| {
                retry_error(&format!(
                    "graph source {} has no disk identity; retry after saving it",
                    file
                ))
            })?;
            let parent = path.parent().ok_or_else(|| {
                retry_error(&format!("graph source {path} has no watchable parent; retry"))
            })?;
            Ok(PathBuf::from(parent.as_str()))
        })
        .collect::<anyhow::Result<BTreeSet<_>>>()?;
    let event_fence = snap.graph_snapshot_cache.lock().source_input_event_fence.clone();
    let watcher = source_input_watcher(&event_fence, roots)?;
    let mut cache = snap.graph_snapshot_cache.lock();
    let event_guard = event_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    drain_source_input_events_at(&mut cache, *event_guard);
    let retired = cache.source_input_watcher.replace(watcher);
    cache.source_input_watcher_stale = false;
    let epoch = *event_guard;
    drop(event_guard);
    drop(cache);
    drop(retired);
    Ok(epoch)
}

fn current_source_input_epoch(snap: &GlobalStateSnapshot) -> anyhow::Result<u64> {
    let mut cache = snap.graph_snapshot_cache.lock();
    if cache.source_input_watcher.is_none() || cache.source_input_watcher_stale {
        return Err(retry_error(
            "graph source watcher is unavailable for an incremental snapshot; retry",
        ));
    }
    let event_fence = cache.source_input_event_fence.clone();
    let event_guard = event_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    drain_source_input_events_at(&mut cache, *event_guard);
    Ok(*event_guard)
}

fn source_input_watcher(
    event_fence: &Arc<StdMutex<u64>>,
    roots: BTreeSet<PathBuf>,
) -> anyhow::Result<RecommendedWatcher> {
    let event_fence = Arc::clone(event_fence);
    let mut watcher = RecommendedWatcher::new(
        move |event: notify::Result<notify::Event>| {
            if event.as_ref().is_ok_and(|event| matches!(event.kind, EventKind::Access(_))) {
                return;
            }
            let mut epoch = event_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *epoch = epoch.wrapping_add(1);
        },
        NotifyConfig::default().with_follow_symlinks(false),
    )
    .map_err(|error| {
        retry_error(&format!("graph source watcher could not start: {error}; retry the request"))
    })?;
    for root in roots {
        watcher.watch(&root, RecursiveMode::NonRecursive).map_err(|error| {
            retry_error(&format!(
                "graph source root {} could not be watched: {error}; retry the request",
                root.display()
            ))
        })?;
    }
    Ok(watcher)
}

fn drain_source_input_events_at(cache: &mut GraphSnapshotCache, event_epoch: u64) {
    if cache.source_input_event_epoch == event_epoch {
        return;
    }
    cache.source_input_event_epoch = event_epoch;
    let revision_already_moved =
        cache.committed.as_ref().is_some_and(|cached| cached.revision != cache.revision);
    if !revision_already_moved {
        cache.invalidate_all();
    }
}

fn checker_source_digest(
    snap: &GlobalStateSnapshot,
    file_id: FileId,
    checker_text: &str,
) -> anyhow::Result<String> {
    let file = snap.file_id_to_file_path(file_id);
    let path = file.as_path().ok_or_else(|| {
        retry_error(&format!("graph source {} has no disk identity; retry after saving it", file))
    })?;
    checker_disk_digest(path.as_ref(), checker_text)
}

fn checker_disk_digest(path: &Path, checker_text: &str) -> anyhow::Result<String> {
    let bytes = fs::read(path).map_err(|error| {
        retry_error(&format!(
            "graph source {} is unavailable on disk: {error}; retry the request",
            path.display()
        ))
    })?;
    let disk_text = String::from_utf8(bytes.clone()).map_err(|error| {
        retry_error(&format!(
            "graph source {} is not UTF-8: {error}; retry the request",
            path.display()
        ))
    })?;
    let (normalized, _) = crate::line_index::LineEndings::normalize(disk_text);
    if normalized != checker_text {
        return Err(retry_error(&format!(
            "graph source {} moved beyond the analyzer VFS; retry the request",
            path.display()
        )));
    }
    Ok(digest_bytes(&bytes))
}

#[cfg(test)]
fn response_for(
    cached: &CachedSnapshot,
    known_generation: Option<&str>,
    cache_hit: bool,
) -> GraphSnapshotResult {
    response_plan(cached, known_generation, cache_hit).into_result()
}

fn response_plan(
    cached: &CachedSnapshot,
    known_generation: Option<&str>,
    cache_hit: bool,
) -> SnapshotResponsePlan {
    let (base_generation, upserts, deletes) =
        if known_generation == Some(cached.generation.as_str()) {
            (Some(cached.generation.clone()), Vec::new(), Vec::new())
        } else if known_generation == cached.delta_base.as_deref() {
            (cached.delta_base.clone(), cached.delta_upserts.clone(), cached.delta_deletes.clone())
        } else {
            (None, cached.shards.values().cloned().collect(), Vec::new())
        };
    SnapshotResponsePlan {
        producer: cached.producer.clone(),
        universe: cached.universe.clone(),
        sequence: cached.sequence,
        generation: cached.generation.clone(),
        base_generation,
        upserts,
        deletes,
        manifest: cached.manifest.clone(),
        phases: if cache_hit {
            GraphSnapshotPhases { cache_hit: true, ..GraphSnapshotPhases::default() }
        } else {
            cached.phases.clone()
        },
    }
}

impl SnapshotResponsePlan {
    fn into_result(self) -> GraphSnapshotResult {
        GraphSnapshotResult {
            protocol_version: PROTOCOL_VERSION,
            schema_version: SCHEMA_VERSION,
            producer: self.producer,
            universe: self.universe,
            sequence: self.sequence,
            generation: self.generation,
            base_generation: self.base_generation,
            upserts: self.upserts.into_iter().map(|shard| (*shard).clone()).collect(),
            deletes: self.deletes,
            manifest: self.manifest,
            phases: self.phases,
        }
    }
}

fn producer() -> GraphSnapshotProducer {
    let version = crate::version();
    GraphSnapshotProducer {
        name: "samchon-rust-analyzer".to_owned(),
        version: version.version.to_owned(),
        commit: version
            .commit_info
            .as_ref()
            .map(|commit| commit.commit_hash)
            .unwrap_or("development")
            .to_owned(),
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ExternalGraphInput {
    identity: String,
    root: PathBuf,
    optional: bool,
}

fn external_graph_inputs(snap: &GlobalStateSnapshot) -> anyhow::Result<Vec<ExternalGraphInput>> {
    let mut inputs = snap
        .workspaces
        .iter()
        .filter_map(|workspace| match &workspace.kind {
            ProjectWorkspaceKind::Cargo { cargo, .. }
            | ProjectWorkspaceKind::DetachedFile { cargo: Some((cargo, ..)), .. } => Some(cargo),
            ProjectWorkspaceKind::Json(_)
            | ProjectWorkspaceKind::DetachedFile { cargo: None, .. } => None,
        })
        .flat_map(|cargo| {
            cargo.packages().filter_map(|package| {
                let package = &cargo[package];
                (package.is_local && !package.is_member).then(|| {
                    let root: &std::path::Path = package.manifest.parent().as_ref();
                    ExternalGraphInput {
                        identity: format!("path-package={}:{}", package.id, package.manifest),
                        root: root.to_path_buf(),
                        optional: false,
                    }
                })
            })
        })
        .collect::<Vec<_>>();
    for workspace in snap.workspaces.iter() {
        inputs.extend(workspace.graph_project_inputs.iter().map(|(path, _)| ExternalGraphInput {
            identity: format!("project-input={path}"),
            root: PathBuf::from(path.as_str()),
            optional: true,
        }));
        let (rustc_path, cargo_path, environment) = match &workspace.kind {
            ProjectWorkspaceKind::Cargo { cargo, .. }
            | ProjectWorkspaceKind::DetachedFile { cargo: Some((cargo, ..)), .. } => (
                workspace.sysroot.tool_path(Tool::Rustc, cargo.workspace_root(), cargo.env()),
                Some(workspace.sysroot.tool_path(Tool::Cargo, cargo.workspace_root(), cargo.env())),
                Some(cargo.env()),
            ),
            ProjectWorkspaceKind::Json(_)
            | ProjectWorkspaceKind::DetachedFile { cargo: None, .. } => (
                workspace
                    .sysroot
                    .root()
                    .and_then(|root| {
                        let bin = root.join("bin");
                        Tool::Rustc.path_in(bin.as_ref())
                    })
                    .unwrap_or_else(|| Tool::Rustc.path()),
                None,
                None,
            ),
        };
        let workspace_root: &std::path::Path = workspace.workspace_root().as_ref();
        let rustc_path: &std::path::Path = rustc_path.as_ref();
        let rustc_path = resolve_external_tool_path(rustc_path, workspace_root, environment)?;
        inputs.push(ExternalGraphInput {
            identity: format!("rustc={}", rustc_path.to_string_lossy()),
            root: rustc_path,
            optional: false,
        });
        if let Some(cargo_path) = cargo_path {
            let cargo_path: &std::path::Path = cargo_path.as_ref();
            let cargo_path = resolve_external_tool_path(cargo_path, workspace_root, environment)?;
            inputs.push(ExternalGraphInput {
                identity: format!("cargo={}", cargo_path.to_string_lossy()),
                root: cargo_path,
                optional: false,
            });
        }
        if let Some(root) = workspace.sysroot.rust_lib_src_root() {
            inputs.push(ExternalGraphInput {
                identity: format!("rust-lib-src={root}"),
                root: PathBuf::from(root.as_str()),
                optional: false,
            });
        }
        let proc_macro_server = snap
            .config
            .proc_macro_srv()
            .or_else(|| workspace.find_sysroot_proc_macro_srv().and_then(Result::ok));
        if let Some(server) = proc_macro_server {
            inputs.push(ExternalGraphInput {
                identity: format!("proc-macro-server={server}"),
                root: PathBuf::from(server.as_str()),
                optional: false,
            });
        }
    }
    inputs.sort();
    inputs.dedup();
    Ok(inputs)
}

fn resolve_external_tool_path(
    tool_path: &Path,
    workspace_root: &Path,
    environment: Option<&ide_db::base_db::Env>,
) -> anyhow::Result<PathBuf> {
    if tool_path.as_os_str().is_empty() {
        return Err(anyhow::format_err!("external graph tool path is empty"));
    }
    if tool_path.is_absolute() {
        return Ok(tool_path.to_path_buf());
    }
    if tool_path.parent().is_some_and(|parent| !parent.as_os_str().is_empty()) {
        return Ok(workspace_root.join(tool_path));
    }

    let search_path = environment
        .and_then(|environment| {
            environment
                .into_iter()
                .find(|(key, _)| {
                    if cfg!(windows) {
                        key.eq_ignore_ascii_case("PATH")
                    } else {
                        key.as_str() == "PATH"
                    }
                })
                .map(|(_, value)| OsString::from(value))
        })
        .or_else(|| env::var_os("PATH"))
        .ok_or_else(|| {
            anyhow::format_err!(
                "external graph tool `{}` is unavailable because PATH is unset",
                tool_path.display()
            )
        })?;
    env::split_paths(&search_path)
        .map(|directory| {
            if directory.as_os_str().is_empty() {
                workspace_root.to_path_buf()
            } else if directory.is_absolute() {
                directory
            } else {
                workspace_root.join(directory)
            }
        })
        .find_map(|directory| probe_external_tool(directory.join(tool_path)))
        .ok_or_else(|| {
            anyhow::format_err!(
                "external graph tool `{}` is unavailable on PATH",
                tool_path.display()
            )
        })
}

fn probe_external_tool(path: PathBuf) -> Option<PathBuf> {
    if path.is_file() {
        return Some(path);
    }
    let extension = env::consts::EXE_EXTENSION;
    (!extension.is_empty()).then(|| path.with_extension(extension)).filter(|path| path.is_file())
}

fn ensure_external_inputs(snap: &GlobalStateSnapshot) -> anyhow::Result<()> {
    let input_fingerprint = digest_json(&json!({
        "workspaces": snap
            .workspaces
            .iter()
            .map(|workspace| workspace.graph_semantic_descriptor())
            .collect::<Vec<_>>(),
        "procMacroServer": snap.config.proc_macro_srv().map(|path| path.to_string()),
    }))?;

    let cached_inputs = {
        let mut cache = snap.graph_snapshot_cache.lock();
        drain_external_input_events(&mut cache);
        (cache.external_inputs == input_fingerprint
            && cache.external_input_watcher.is_some()
            && !cache.external_input_watcher_stale)
            .then(|| cache.external_input_roots.clone())
    };
    let (inputs, held_watcher) = if let Some(inputs) = cached_inputs {
        (inputs, None)
    } else {
        let inputs = external_graph_inputs(snap)?;
        let event_fence = snap.graph_snapshot_cache.lock().external_input_event_fence.clone();
        let watcher = external_input_watcher(
            &event_fence,
            external_input_watch_roots(&inputs, &BTreeMap::new())?,
        )?;
        let mut cache = snap.graph_snapshot_cache.lock();
        let held_watcher = cache.external_input_watcher.replace(watcher);
        cache.external_inputs = input_fingerprint.clone();
        cache.external_input_roots = inputs.clone();
        cache.external_inputs_dirty = true;
        cache.external_input_watcher_stale = false;
        (inputs, held_watcher)
    };

    let revision = {
        let cache = snap.graph_snapshot_cache.lock();
        if !cache.external_inputs_dirty {
            return cache.external_input_digest.clone().map(|_| ()).map_err(|error| {
                retry_error(&format!(
                    "external graph inputs are unavailable: {error}; retry the request"
                ))
            });
        }
        cache.external_input_revision
    };
    let first_capture = capture_external_inputs(&inputs);
    let (capture, replacement) = match first_capture {
        Ok(first_capture) => {
            let roots = external_input_watch_roots(&inputs, &first_capture.symlink_watch_roots)?;
            let event_fence = snap.graph_snapshot_cache.lock().external_input_event_fence.clone();
            let replacement = external_input_watcher(&event_fence, roots)?;
            match capture_external_inputs(&inputs) {
                Ok(fenced_capture) => {
                    if fenced_capture.symlink_watch_roots != first_capture.symlink_watch_roots {
                        return Err(retry_error(
                            "external graph-input symlinks moved while their watcher was being fenced; retry the request",
                        ));
                    }
                    (Ok(fenced_capture), Some(replacement))
                }
                Err(error) => (Err(error), None),
            }
        }
        Err(error) => (Err(error), None),
    };
    let mut cache = snap.graph_snapshot_cache.lock();
    let event_fence = cache.external_input_event_fence.clone();
    let event_guard = event_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    drain_external_input_events_at(&mut cache, *event_guard);
    if cache.external_inputs != input_fingerprint || cache.external_input_revision != revision {
        return Err(retry_error(
            "external graph inputs moved while they were being captured; retry the request",
        ));
    }
    match capture {
        Ok(capture) => {
            cache.external_input_digest = Ok(capture.digest);
            cache.external_inputs_dirty = false;
            let retired_watcher = cache.external_input_watcher.replace(
                replacement.expect("a successful capture always builds a replacement watcher"),
            );
            cache.external_input_watcher_stale = false;
            drop(event_guard);
            drop(cache);
            drop(retired_watcher);
            drop(held_watcher);
            Ok(())
        }
        Err(error) => {
            cache.external_input_digest = Err(error.to_string());
            cache.external_inputs_dirty = true;
            Err(retry_error(&format!(
                "external graph inputs could not be captured: {error}; retry the request"
            )))
        }
    }
}

fn drain_external_input_events(cache: &mut GraphSnapshotCache) {
    let event_fence = cache.external_input_event_fence.clone();
    let event_guard = event_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    drain_external_input_events_at(cache, *event_guard);
}

fn drain_external_input_events_at(cache: &mut GraphSnapshotCache, event_epoch: u64) {
    if cache.external_input_event_epoch != event_epoch {
        cache.external_input_event_epoch = event_epoch;
        cache.external_input_revision = cache.external_input_revision.wrapping_add(1);
        cache.external_inputs_dirty = true;
        cache.external_input_watcher_stale = true;
        cache.invalidate_all();
    }
}

fn external_input_watcher(
    event_fence: &Arc<StdMutex<u64>>,
    roots: BTreeMap<PathBuf, bool>,
) -> anyhow::Result<RecommendedWatcher> {
    let event_fence = Arc::clone(event_fence);
    let mut watcher = RecommendedWatcher::new(
        move |event: notify::Result<notify::Event>| {
            if event.as_ref().is_ok_and(|event| matches!(event.kind, EventKind::Access(_))) {
                return;
            }
            let mut epoch = event_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *epoch = epoch.wrapping_add(1);
        },
        NotifyConfig::default().with_follow_symlinks(false),
    )
    .map_err(|error| {
        retry_error(&format!(
            "external graph-input watcher could not start: {error}; retry the request"
        ))
    })?;
    for (root, recursive) in roots {
        let mode = if recursive { RecursiveMode::Recursive } else { RecursiveMode::NonRecursive };
        watcher.watch(&root, mode).map_err(|error| {
            retry_error(&format!(
                "external graph input {} could not be watched: {error}; retry the request",
                root.display()
            ))
        })?;
    }
    Ok(watcher)
}

fn external_input_watch_roots(
    inputs: &[ExternalGraphInput],
    symlink_watch_roots: &BTreeMap<PathBuf, bool>,
) -> anyhow::Result<BTreeMap<PathBuf, bool>> {
    let mut roots = BTreeMap::new();
    for input in inputs {
        let metadata = match fs::metadata(&input.root) {
            Ok(metadata) => metadata,
            Err(error)
                if input.optional
                    && matches!(
                        error.kind(),
                        std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                    ) =>
            {
                insert_missing_target_watch_roots(&mut roots, input.root.clone());
                continue;
            }
            Err(error) => {
                return Err(retry_error(&format!(
                    "external graph input {} is unavailable: {error}; retry the request",
                    input.root.display()
                )));
            }
        };
        insert_watch_root(&mut roots, input.root.clone(), metadata.is_dir());
        if let Some(parent) = input.root.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            insert_watch_root(&mut roots, parent.to_path_buf(), false);
        }
    }
    for (root, recursive) in symlink_watch_roots {
        insert_watch_root(&mut roots, root.clone(), *recursive);
    }
    Ok(roots)
}

fn insert_watch_root(roots: &mut BTreeMap<PathBuf, bool>, root: PathBuf, recursive: bool) {
    roots.entry(root).and_modify(|prior| *prior |= recursive).or_insert(recursive);
}

struct ExternalInputCapture {
    digest: String,
    symlink_watch_roots: BTreeMap<PathBuf, bool>,
}

fn capture_external_inputs(inputs: &[ExternalGraphInput]) -> anyhow::Result<ExternalInputCapture> {
    let mut content = BTreeMap::<Vec<u8>, Vec<u8>>::new();
    let mut symlink_watch_roots = BTreeMap::new();
    for input in inputs {
        if input.optional
            && fs::symlink_metadata(&input.root).is_err_and(|error| {
                matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                )
            })
        {
            content.insert(
                external_input_key_suffix(&external_input_key(input, Path::new("")), b"missing"),
                Vec::new(),
            );
            insert_missing_target_watch_roots(&mut symlink_watch_roots, input.root.clone());
            continue;
        }
        for entry in WalkDir::new(&input.root).follow_links(true) {
            let entry = match entry {
                Ok(entry) => entry,
                Err(error) => {
                    let Some(path) = error.path() else {
                        return Err(error.into());
                    };
                    if fs::symlink_metadata(path)
                        .is_ok_and(|metadata| metadata.file_type().is_symlink())
                    {
                        capture_symlink(input, path, &mut content, &mut symlink_watch_roots)?;
                        continue;
                    }
                    return Err(error.into());
                }
            };
            let path = entry.path();
            let relative = path.strip_prefix(&input.root).unwrap_or(path);
            let key = external_input_key(input, relative);
            if entry.path_is_symlink() {
                capture_symlink(input, path, &mut content, &mut symlink_watch_roots)?;
                if fs::metadata(path).is_ok_and(|metadata| metadata.is_file()) {
                    content.insert(external_input_key_suffix(&key, b"file"), fs::read(path)?);
                }
            }
            if entry.file_type().is_file() {
                content.insert(external_input_key_suffix(&key, b"file"), fs::read(path)?);
            }
        }
    }
    let mut digest = Sha256::new();
    for (key, bytes) in content {
        digest.update((key.len() as u64).to_le_bytes());
        digest.update(key);
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    Ok(ExternalInputCapture { digest: format!("{:x}", digest.finalize()), symlink_watch_roots })
}

fn capture_symlink(
    input: &ExternalGraphInput,
    path: &Path,
    content: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    watch_roots: &mut BTreeMap<PathBuf, bool>,
) -> anyhow::Result<()> {
    let target = fs::read_link(path)?;
    let relative = path.strip_prefix(&input.root).unwrap_or(path);
    let key = external_input_key(input, relative);
    content.insert(external_input_key_suffix(&key, b"symlink"), path_bytes(&target));
    let link = canonical_link_identity(path);
    if let Some(parent) = link.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        insert_watch_root(watch_roots, parent.to_path_buf(), false);
    }
    let target =
        if target.is_absolute() { target } else { link.parent().unwrap_or(&link).join(target) };
    capture_symlink_target(
        target,
        &key,
        content,
        watch_roots,
        &mut BTreeSet::from([(link, PathBuf::new())]),
    )
}

fn capture_symlink_target(
    mut target: PathBuf,
    key: &[u8],
    content: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    watch_roots: &mut BTreeMap<PathBuf, bool>,
    seen: &mut BTreeSet<(PathBuf, PathBuf)>,
) -> anyhow::Result<()> {
    let mut hops = 0;
    loop {
        match inspect_symlink_path(&target)? {
            SymlinkPath::Link { path, suffix } => {
                hops += 1;
                if hops > MAX_SYMLINK_HOPS {
                    return Ok(());
                }
                let link = canonical_link_identity(&path);
                if let Some(parent) = link.parent().filter(|parent| !parent.as_os_str().is_empty())
                {
                    insert_watch_root(watch_roots, parent.to_path_buf(), false);
                }
                if !seen.insert((link.clone(), suffix.clone())) {
                    return Ok(());
                }
                let raw_target = fs::read_link(&path)?;
                let mut hop_key = external_input_key_suffix(key, b"symlink-hop");
                append_key_part(&mut hop_key, &path_bytes(&link));
                content.insert(hop_key, path_bytes(&raw_target));
                let next = if raw_target.is_absolute() {
                    raw_target
                } else {
                    link.parent().unwrap_or(&link).join(raw_target)
                };
                target = next.join(suffix);
            }
            SymlinkPath::Existing { path, directory } => {
                let path = fs::canonicalize(&path).unwrap_or(path);
                insert_watch_root(watch_roots, path.clone(), directory);
                if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty())
                {
                    insert_watch_root(watch_roots, parent.to_path_buf(), false);
                }
                return Ok(());
            }
            SymlinkPath::Missing { path } => {
                insert_missing_target_watch_roots(watch_roots, path);
                return Ok(());
            }
        }
    }
}

enum SymlinkPath {
    Link { path: PathBuf, suffix: PathBuf },
    Existing { path: PathBuf, directory: bool },
    Missing { path: PathBuf },
}

fn inspect_symlink_path(path: &Path) -> anyhow::Result<SymlinkPath> {
    let mut prefix = PathBuf::new();
    let mut components = path.components();
    while let Some(component) = components.next() {
        match component {
            std::path::Component::CurDir => continue,
            std::path::Component::ParentDir => {
                prefix.pop();
                continue;
            }
            std::path::Component::Prefix(_) | std::path::Component::RootDir => {
                prefix.push(component.as_os_str());
                continue;
            }
            std::path::Component::Normal(_) => prefix.push(component.as_os_str()),
        }
        match fs::symlink_metadata(&prefix) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Ok(SymlinkPath::Link {
                    path: prefix,
                    suffix: components.as_path().to_path_buf(),
                });
            }
            Ok(metadata) => {
                if !components.as_path().as_os_str().is_empty() && !metadata.is_dir() {
                    return Ok(SymlinkPath::Missing { path: path.to_path_buf() });
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::NotFound | std::io::ErrorKind::NotADirectory
                ) =>
            {
                return Ok(SymlinkPath::Missing { path: path.to_path_buf() });
            }
            Err(error) => return Err(error.into()),
        }
    }
    match fs::metadata(&prefix) {
        Ok(metadata) => Ok(SymlinkPath::Existing { path: prefix, directory: metadata.is_dir() }),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            Ok(SymlinkPath::Missing { path: path.to_path_buf() })
        }
        Err(error) => Err(error.into()),
    }
}

fn insert_missing_target_watch_roots(watch_roots: &mut BTreeMap<PathBuf, bool>, target: PathBuf) {
    let Some(root) = nearest_existing_watch_root(target) else { return };
    insert_watch_root(watch_roots, root.clone(), false);
    if let Some(parent) = root.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        insert_watch_root(watch_roots, parent.to_path_buf(), false);
    }
}

fn canonical_link_identity(path: &Path) -> PathBuf {
    let Some(parent) = path.parent() else { return path.to_path_buf() };
    let parent = fs::canonicalize(parent).unwrap_or_else(|_| parent.to_path_buf());
    match path.file_name() {
        Some(name) => parent.join(name),
        None => parent,
    }
}

fn external_input_key(input: &ExternalGraphInput, relative: &Path) -> Vec<u8> {
    let mut key = Vec::new();
    append_key_part(&mut key, input.identity.as_bytes());
    append_key_part(&mut key, &path_bytes(&input.root));
    append_key_part(&mut key, &path_bytes(relative));
    key
}

fn external_input_key_suffix(key: &[u8], suffix: &[u8]) -> Vec<u8> {
    let mut output = key.to_vec();
    append_key_part(&mut output, suffix);
    output
}

fn append_key_part(output: &mut Vec<u8>, part: &[u8]) {
    output.extend_from_slice(&(part.len() as u64).to_le_bytes());
    output.extend_from_slice(part);
}

#[cfg(unix)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::unix::ffi::OsStrExt;
    path.as_os_str().as_bytes().to_vec()
}

#[cfg(windows)]
fn path_bytes(path: &Path) -> Vec<u8> {
    use std::os::windows::ffi::OsStrExt;
    path.as_os_str().encode_wide().flat_map(u16::to_le_bytes).collect()
}

#[cfg(not(any(unix, windows)))]
fn path_bytes(path: &Path) -> Vec<u8> {
    path.to_string_lossy().as_bytes().to_vec()
}

fn nearest_existing_watch_root(mut path: PathBuf) -> Option<PathBuf> {
    loop {
        if fs::metadata(&path).is_ok() {
            return Some(path);
        }
        if !path.pop() {
            return None;
        }
    }
}

fn universe(snap: &GlobalStateSnapshot) -> anyhow::Result<GraphSnapshotUniverse> {
    let mut workspace_roots = snap
        .workspaces
        .iter()
        .map(|workspace| normalize_path(workspace.workspace_root().as_str()))
        .collect::<Vec<_>>();
    workspace_roots.sort();
    workspace_roots.dedup();

    let mut toolchains = snap
        .workspaces
        .iter()
        .filter_map(|workspace| workspace.toolchain.as_ref().map(ToString::to_string))
        .collect::<Vec<_>>();
    toolchains.sort();
    toolchains.dedup();

    let cargo_config = snap.config.cargo(None);
    let profile = cargo_profile(&cargo_config.extra_args);
    let mut ignored_proc_macros = snap
        .config
        .ignored_proc_macros(None)
        .iter()
        .map(|(crate_name, macros)| {
            let mut macros = macros.iter().map(ToString::to_string).collect::<Vec<_>>();
            macros.sort();
            (crate_name.to_string(), macros)
        })
        .collect::<Vec<_>>();
    ignored_proc_macros.sort();
    let mut environment = std::env::vars_os()
        .map(|(key, value)| format!("{}={}", key.to_string_lossy(), value.to_string_lossy()))
        .collect::<Vec<_>>();
    environment.sort();

    let mut configurations = vec![
        format!("profile={profile}"),
        format!("cargo-config-sha256={}", digest_bytes(format!("{cargo_config:#?}").as_bytes())),
        format!("environment-sha256={}", digest_bytes(environment.join("\0").as_bytes())),
        format!("expand-proc-macros={}", snap.config.expand_proc_macros()),
        format!("proc-macros-loaded={}", snap.proc_macros_loaded),
        format!("run-build-scripts={}", snap.config.run_build_scripts(None)),
        format!(
            "ignored-proc-macros-sha256={}",
            digest_bytes(format!("{ignored_proc_macros:?}").as_bytes())
        ),
        format!(
            "proc-macro-server={}",
            snap.config
                .proc_macro_srv()
                .map(|path| normalize_path(path.as_str()))
                .unwrap_or_else(|| "sysroot-or-unavailable".to_owned())
        ),
    ];
    let external_input_digest =
        snap.graph_snapshot_cache.lock().external_input_digest.clone().map_err(|error| {
            retry_error(&format!(
                "external graph inputs are unavailable: {error}; retry the request"
            ))
        })?;
    configurations.push(format!("external-inputs-sha256={external_input_digest}"));
    for workspace in snap.workspaces.iter() {
        configurations.push(format!(
            "workspace-descriptor-sha256={}",
            digest_bytes(workspace.graph_semantic_descriptor().as_bytes())
        ));
        configurations.extend(workspace.graph_project_inputs.iter().map(|(path, bytes)| {
            format!(
                "project-input={};sha256={}",
                normalize_path(path.as_str()),
                bytes.as_deref().map(digest_bytes).unwrap_or_else(|| "missing".to_owned())
            )
        }));
        configurations.push(format!(
            "cargo-lock-sha256={}",
            workspace
                .graph_lockfile
                .as_deref()
                .map(digest_bytes)
                .unwrap_or_else(|| "missing".to_owned())
        ));
        configurations.push(format!("target={:?}", workspace.target));
        configurations.extend(workspace.rustc_cfg.iter().map(|cfg| format!("cfg={cfg:?}")));
        configurations.push(format!("cfg-overrides={:?}", workspace.cfg_overrides));
        configurations.push(format!("set-test={}", workspace.set_test));
        let rustc_version = workspace.graph_rustc_version.as_ref().map_err(|error| {
            retry_error(&format!(
                "exact rustc identity was unavailable in the project model: {error}; reload and retry"
            ))
        })?;
        configurations.push(format!("rustc-version={rustc_version}"));
        if let ProjectWorkspaceKind::Cargo { cargo, .. }
        | ProjectWorkspaceKind::DetachedFile { cargo: Some((cargo, ..)), .. } = &workspace.kind
        {
            for package in cargo.packages() {
                let package = &cargo[package];
                let mut features = package.active_features.clone();
                features.sort();
                let mut dependencies = package
                    .dependencies
                    .iter()
                    .map(|dependency| {
                        format!(
                            "{}:{}:{:?}",
                            dependency.name, cargo[dependency.pkg].id, dependency.kind
                        )
                    })
                    .collect::<Vec<_>>();
                dependencies.sort();
                configurations.push(format!(
                    "package-id={};name={}@{};manifest={};edition={:?};features={};dependencies={}",
                    package.id,
                    package.name,
                    package.version,
                    normalize_path(&package.manifest.to_string()),
                    package.edition,
                    features.join(","),
                    dependencies.join(",")
                ));
                for &target in &package.targets {
                    let target = &cargo[target];
                    let mut required_features = target.required_features.clone();
                    required_features.sort();
                    configurations.push(format!(
                        "target={};kind={:?};root={};required-features={}",
                        target.name,
                        target.kind,
                        normalize_path(target.root.as_str()),
                        required_features.join(",")
                    ));
                }
            }
        }
    }
    configurations.sort();
    configurations.dedup();
    let digest = digest_json(&json!({
        "workspaceRoots": workspace_roots,
        "toolchains": toolchains,
        "configurations": configurations,
        "producer": producer(),
    }))?;
    Ok(GraphSnapshotUniverse {
        target: format!("rust:{digest}"),
        digest,
        workspace_roots,
        toolchains,
        configurations,
    })
}

fn cargo_profile(extra_args: &[String]) -> String {
    if extra_args.iter().any(|arg| arg == "--release") {
        return "release".to_owned();
    }
    extra_args
        .windows(2)
        .find_map(|args| (args[0] == "--profile").then(|| args[1].clone()))
        .or_else(|| {
            extra_args.iter().find_map(|arg| arg.strip_prefix("--profile=").map(str::to_owned))
        })
        .unwrap_or_else(|| "dev".to_owned())
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn compute_interface_fingerprints(
    snap: &GlobalStateSnapshot,
    index: &StaticIndex<'_>,
) -> anyhow::Result<BTreeMap<String, String>> {
    let mut entries = index
        .files
        .iter()
        .map(|file| (source_path(snap, file.file_id), Vec::<String>::new()))
        .collect::<BTreeMap<_, _>>();

    for (_, token) in index.tokens.iter_ref() {
        if !token.local
            && !token.external
            && let Some(definition) = token.definition
        {
            let source = source_path(snap, definition.file_id);
            if let Some(source_entries) = entries.get_mut(&source) {
                let semantic_body = if token.const_function
                    || matches!(
                        token.kind,
                        SymbolInformationKind::Constant
                            | SymbolInformationKind::EnumMember
                            | SymbolInformationKind::Macro
                            | SymbolInformationKind::StaticVariable
                    ) {
                    token.definition_body.and_then(|body| {
                        let text = snap.analysis.file_text(body.file_id).ok()?;
                        text.get(usize::from(body.range.start())..usize::from(body.range.end()))
                            .map(str::to_owned)
                    })
                } else {
                    None
                };
                source_entries.push(canonical_json(&json!({
                    "kind": format!("{:?}", token.kind),
                    "id": token.stable_id,
                    "qualifiedName": token.qualified_name,
                    "signature": token.signature,
                    "exported": token.exported,
                    "semanticBody": semantic_body,
                }))?);
            }
        }

        for reference in &token.references {
            if reference.is_definition
                || !matches!(
                    reference.role,
                    StaticReferenceRole::Export | StaticReferenceRole::Import
                )
            {
                continue;
            }
            let source = source_path(snap, reference.range.file_id);
            if let Some(source_entries) = entries.get_mut(&source) {
                source_entries.push(canonical_json(&json!({
                    "role": reference_kind(reference.role),
                    "target": token.stable_id,
                    "spelling": reference.interface_spelling,
                    "export": reference.export.as_ref().map(|export| json!({
                        "exporter": export.exporter,
                        "aliasId": export.alias_id,
                        "alias": export.alias,
                        "qualifiedName": export.qualified_name,
                    })),
                }))?);
            }
        }
    }

    for relation in &index.relations {
        let source = source_path(snap, relation.file_id);
        if let Some(source_entries) = entries.get_mut(&source) {
            source_entries.push(canonical_json(&json!({
                "relation": format!("{:?}", relation.kind),
                "from": relation.from,
                "to": relation.to,
                "toDisplayName": relation.to_display_name,
                "toQualifiedName": relation.to_qualified_name,
                "toSignature": relation.to_signature,
                "toKind": format!("{:?}", relation.to_kind),
                "toExternal": relation.to_external,
                "toExported": relation.to_exported,
            }))?);
        }
    }

    entries
        .into_iter()
        .map(|(source, mut source_entries)| {
            source_entries.sort();
            Ok((source, digest_json(&source_entries)?))
        })
        .collect()
}

fn interfaces_changed(
    cached: &BTreeMap<String, String>,
    current: &BTreeMap<String, String>,
    dirty_sources: &BTreeSet<String>,
) -> bool {
    dirty_sources
        .iter()
        .any(|source| !cached.contains_key(source) || cached.get(source) != current.get(source))
}

#[derive(Default)]
struct MutableShard {
    source: String,
    nodes: Vec<GraphSnapshotNode>,
    edges: Vec<GraphSnapshotEdge>,
    diagnostics: Vec<GraphSnapshotDiagnostic>,
    unresolved: Vec<GraphSnapshotUnresolved>,
}

struct BuiltShards {
    shards: Vec<GraphSnapshotShard>,
    node_owners: BTreeMap<String, String>,
}

fn build_shards(
    snap: &GlobalStateSnapshot,
    index: StaticIndex<'_>,
    universe: &GraphSnapshotUniverse,
    interface_fingerprints: &BTreeMap<String, String>,
    source_inputs: &BTreeMap<FileId, String>,
    mut id_sources: BTreeMap<String, String>,
) -> anyhow::Result<BuiltShards> {
    let files = index.files;
    let relations = index.relations;
    let tokens = index.tokens.iter().map(|(_, token)| token).collect::<Vec<_>>();
    let mut sources = BTreeMap::<FileId, String>::new();
    let mut source_digests = BTreeMap::<String, String>::new();
    let conditional_files = files
        .iter()
        .filter(|file| file.conditional_build)
        .map(|file| file.file_id)
        .collect::<BTreeSet<_>>();
    for file in &files {
        let source = source_path(snap, file.file_id);
        source_digests.insert(
            source.clone(),
            source_inputs.get(&file.file_id).cloned().ok_or_else(|| {
                retry_error("graph source set moved beyond its validated disk inputs; retry")
            })?,
        );
        sources.insert(file.file_id, source);
    }

    let mut shards = sources
        .values()
        .map(|source| {
            (source.clone(), MutableShard { source: source.clone(), ..Default::default() })
        })
        .collect::<BTreeMap<_, _>>();
    let dependency_source = "bundled:///rust/dependencies".to_owned();
    let mut node_sources = Vec::with_capacity(tokens.len());
    let mut node_present = Vec::with_capacity(tokens.len());
    for token in &tokens {
        let definition = token.definition;
        let external = token.external || definition.is_none();
        let definition_source = (!external)
            .then(|| definition.map(|definition| source_path(snap, definition.file_id)))
            .flatten();
        let file = if external {
            dependency_source.clone()
        } else {
            definition
                .map(|definition| source_path(snap, definition.file_id))
                .unwrap_or_else(|| dependency_source.clone())
        };
        let owner_source =
            id_sources.get(&token.stable_id).cloned().or(definition_source).or_else(|| {
                token
                    .references
                    .iter()
                    .filter(|reference| !reference.is_definition)
                    .filter_map(|reference| sources.get(&reference.range.file_id).cloned())
                    .next()
            });
        let name = token
            .display_name
            .clone()
            .or_else(|| token.qualified_name.as_deref()?.rsplit("::").next().map(str::to_owned));
        let node = name.map(|name| GraphSnapshotNode {
            id: token.stable_id.clone(),
            kind: graph_node_kind(token.kind).to_owned(),
            name,
            qualified_name: token.qualified_name.clone(),
            file,
            external,
            exported: token.exported,
            signature: token.signature.clone(),
            evidence: (!external)
                .then(|| definition.and_then(|range| evidence(snap, range).ok()))
                .flatten(),
        });
        if let (Some(node), Some(source)) = (node.as_ref(), owner_source.as_ref()) {
            id_sources.entry(node.id.clone()).or_insert_with(|| source.clone());
            if let Some(shard) = shards.get_mut(source) {
                shard.nodes.push(node.clone());
            }
        }
        node_present.push(id_sources.contains_key(&token.stable_id));
        node_sources.push(owner_source);
    }

    let mut file_nodes = BTreeMap::new();
    for (&file_id, source) in &sources {
        let id = format!("rust-file-v1|{}:{}", source.len(), source);
        file_nodes.insert(file_id, id.clone());
        id_sources.insert(id.clone(), source.clone());
        shards.get_mut(source).unwrap().nodes.push(GraphSnapshotNode {
            id,
            kind: "file".to_owned(),
            name: source.rsplit('/').next().unwrap_or(source).to_owned(),
            qualified_name: None,
            file: source.clone(),
            external: false,
            exported: false,
            signature: None,
            evidence: None,
        });
    }

    let definitions = tokens
        .iter()
        .enumerate()
        .filter_map(|(token_id, token)| {
            token
                .definition_body
                .filter(|body| sources.contains_key(&body.file_id))
                .map(|body| (token_id, body, token.stable_id.as_str()))
        })
        .collect::<Vec<_>>();
    let mut edge_keys = BTreeSet::new();
    for (token_id, token) in tokens.iter().enumerate() {
        if !node_present[token_id] {
            continue;
        }
        for reference in &token.references {
            if reference.is_definition {
                continue;
            }
            let Some(source) = sources.get(&reference.range.file_id) else {
                continue;
            };
            if let Some(export) = &reference.export {
                id_sources.insert(export.alias_id.clone(), source.clone());
                shards.get_mut(source).unwrap().nodes.push(GraphSnapshotNode {
                    id: export.alias_id.clone(),
                    kind: graph_node_kind(token.kind).to_owned(),
                    name: export.alias.clone(),
                    qualified_name: Some(export.qualified_name.clone()),
                    file: source.clone(),
                    external: false,
                    exported: true,
                    signature: token.signature.clone(),
                    evidence: evidence(snap, export.alias_range).ok(),
                });
                let exports_key = (
                    export.exporter.clone(),
                    export.alias_id.clone(),
                    "exports".to_owned(),
                    source.clone(),
                );
                if edge_keys.insert(exports_key) {
                    shards.get_mut(source).unwrap().edges.push(GraphSnapshotEdge {
                        from: export.exporter.clone(),
                        to: export.alias_id.clone(),
                        kind: "exports".to_owned(),
                        evidence: evidence(snap, export.alias_range).ok(),
                    });
                }
                let target_key = (
                    export.alias_id.clone(),
                    token.stable_id.clone(),
                    "references".to_owned(),
                    source.clone(),
                );
                if edge_keys.insert(target_key) {
                    shards.get_mut(source).unwrap().edges.push(GraphSnapshotEdge {
                        from: export.alias_id.clone(),
                        to: token.stable_id.clone(),
                        kind: "references".to_owned(),
                        evidence: evidence(snap, reference.range).ok(),
                    });
                }
                continue;
            }
            let owner = if reference.role == StaticReferenceRole::Export {
                file_nodes.get(&reference.range.file_id).cloned().unwrap()
            } else {
                enclosing_owner(&definitions, reference.range)
                    .map(str::to_owned)
                    .or_else(|| file_nodes.get(&reference.range.file_id).cloned())
                    .unwrap()
            };
            let kind = reference_kind(reference.role);
            let key = (owner.clone(), token.stable_id.clone(), kind.to_owned(), source.clone());
            if edge_keys.insert(key) {
                shards.get_mut(source).unwrap().edges.push(GraphSnapshotEdge {
                    from: owner,
                    to: token.stable_id.clone(),
                    kind: kind.to_owned(),
                    evidence: evidence(snap, reference.range).ok(),
                });
            }
        }
    }

    for relation in relations {
        let (Some(source), true) =
            (sources.get(&relation.file_id), id_sources.contains_key(&relation.from))
        else {
            continue;
        };
        if !id_sources.contains_key(&relation.to) {
            let Some(name) = relation.to_display_name.clone() else {
                continue;
            };
            let target_source = (!relation.to_external)
                .then(|| relation.to_definition_file.map(|file_id| source_path(snap, file_id)))
                .flatten();
            let owner_source = target_source.clone().unwrap_or_else(|| source.clone());
            if let Some(shard) = shards.get_mut(&owner_source) {
                shard.nodes.push(GraphSnapshotNode {
                    id: relation.to.clone(),
                    kind: graph_node_kind(relation.to_kind).to_owned(),
                    name,
                    qualified_name: relation.to_qualified_name.clone(),
                    file: target_source.clone().unwrap_or_else(|| dependency_source.clone()),
                    external: relation.to_external || target_source.is_none(),
                    exported: relation.to_exported,
                    signature: Some(relation.to_signature.clone()),
                    evidence: None,
                });
            }
            id_sources.insert(relation.to.clone(), owner_source);
        }
        let kind = match relation.kind {
            StaticRelationKind::Extends => "extends",
            StaticRelationKind::Implements => "implements",
            StaticRelationKind::Overrides => "overrides",
        };
        let key = (relation.from.clone(), relation.to.clone(), kind.to_owned(), source.clone());
        if edge_keys.insert(key) {
            shards.get_mut(source).unwrap().edges.push(GraphSnapshotEdge {
                from: relation.from,
                to: relation.to,
                kind: kind.to_owned(),
                evidence: None,
            });
        }
    }

    for (token_id, body, node_id) in &definitions {
        let Some(source) = node_sources.get(*token_id).and_then(Option::as_ref) else {
            continue;
        };
        let parent = enclosing_owner_excluding(&definitions, *body, *token_id)
            .map(str::to_owned)
            .or_else(|| file_nodes.get(&body.file_id).cloned())
            .unwrap();
        let key = (parent.clone(), (*node_id).to_owned(), "contains".to_owned(), source.clone());
        if edge_keys.insert(key) {
            shards.get_mut(source).unwrap().edges.push(GraphSnapshotEdge {
                from: parent,
                to: (*node_id).to_owned(),
                kind: "contains".to_owned(),
                evidence: evidence(snap, tokens[*token_id].definition.unwrap_or(*body)).ok(),
            });
        }
        if tokens[*token_id].exported {
            let exporter = file_nodes.get(&body.file_id).cloned().unwrap();
            let key =
                (exporter.clone(), (*node_id).to_owned(), "exports".to_owned(), source.clone());
            if edge_keys.insert(key) {
                shards.get_mut(source).unwrap().edges.push(GraphSnapshotEdge {
                    from: exporter,
                    to: (*node_id).to_owned(),
                    kind: "exports".to_owned(),
                    evidence: evidence(snap, tokens[*token_id].definition.unwrap_or(*body)).ok(),
                });
            }
        }
    }

    for (&file_id, source) in &sources {
        let source_root = snap.analysis.source_root_id(file_id)?;
        let config = snap.config.diagnostics(Some(source_root));
        for diagnostic in
            snap.analysis.full_diagnostics(&config, AssistResolveStrategy::None, file_id)?
        {
            let position = evidence(snap, diagnostic.range)?;
            shards.get_mut(source).unwrap().diagnostics.push(GraphSnapshotDiagnostic {
                file: position.file,
                line: position.start_line,
                column: Some(position.start_column),
                code: diagnostic.code.as_str().to_owned(),
                message: diagnostic.message,
                severity: Some(
                    match diagnostic.severity {
                        Severity::Error => "error",
                        Severity::Warning | Severity::WeakWarning => "warning",
                        Severity::Allow => "hint",
                    }
                    .to_owned(),
                ),
            });
        }
        shards
            .get_mut(source)
            .unwrap()
            .unresolved
            .extend(collect_unresolved(snap, file_id, &tokens)?);
        if conditional_files.contains(&file_id) {
            let range = snap.analysis.parse(file_id)?.syntax().text_range();
            let position = evidence(snap, FileRange { file_id, range })?;
            shards.get_mut(source).unwrap().unresolved.extend(
                COVERAGE.iter().filter(|(_, state)| *state != "unsupported").map(|(family, _)| {
                    GraphSnapshotUnresolved {
                        family: (*family).to_owned(),
                        evidence: position.clone(),
                        reason: "conditional-build".to_owned(),
                        candidates: Vec::new(),
                    }
                }),
            );
        }
        let range = snap.analysis.parse(file_id)?.syntax().text_range();
        let position = evidence(snap, FileRange { file_id, range })?;
        let shard = shards.get_mut(source).unwrap();
        for (family, state) in COVERAGE {
            if state == "partial"
                && !shard.unresolved.iter().any(|unresolved| unresolved.family == family)
            {
                shard.unresolved.push(GraphSnapshotUnresolved {
                    family: family.to_owned(),
                    evidence: position.clone(),
                    reason: "provider-gap".to_owned(),
                    candidates: Vec::new(),
                });
            }
        }
    }

    let test_attributes = tokens
        .iter()
        .filter(|token| {
            token.kind == SymbolInformationKind::Attribute
                && token.display_name.as_deref() == Some("test")
        })
        .map(|token| token.stable_id.as_str())
        .collect::<BTreeSet<_>>();
    let test_owners = shards
        .values()
        .flat_map(|shard| &shard.edges)
        .filter(|edge| edge.kind == "decorates" && test_attributes.contains(edge.to.as_str()))
        .map(|edge| edge.from.clone())
        .collect::<BTreeSet<_>>();
    let tested = shards
        .iter()
        .flat_map(|(source, shard)| {
            shard
                .edges
                .iter()
                .filter(|edge| {
                    test_owners.contains(&edge.from)
                        && matches!(edge.kind.as_str(), "calls" | "instantiates")
                })
                .map(|edge| {
                    (source.clone(), edge.from.clone(), edge.to.clone(), edge.evidence.clone())
                })
        })
        .collect::<Vec<_>>();
    for (source, from, to, evidence) in tested {
        let key = (from.clone(), to.clone(), "tests".to_owned(), source.clone());
        if edge_keys.insert(key) {
            shards.get_mut(&source).unwrap().edges.push(GraphSnapshotEdge {
                from,
                to,
                kind: "tests".to_owned(),
                evidence,
            });
        }
    }

    let mut result = Vec::new();
    for (_, mut shard) in shards {
        shard.nodes.sort_by(|a, b| a.id.cmp(&b.id));
        shard.nodes.dedup_by(|a, b| a.id == b.id);
        shard.edges.sort_by(|a, b| (&a.from, &a.to, &a.kind).cmp(&(&b.from, &b.to, &b.kind)));
        shard.diagnostics.sort_by(|a, b| {
            (&a.file, a.line, a.column, &a.code, &a.message)
                .cmp(&(&b.file, b.line, b.column, &b.code, &b.message))
        });
        shard.unresolved.sort_by(|a, b| {
            (&a.family, &a.evidence.file, a.evidence.start_line, a.evidence.start_column, &a.reason)
                .cmp(&(
                    &b.family,
                    &b.evidence.file,
                    b.evidence.start_line,
                    b.evidence.start_column,
                    &b.reason,
                ))
        });
        shard.unresolved.dedup_by(|a, b| {
            a.family == b.family && a.evidence == b.evidence && a.reason == b.reason
        });
        let coverage = COVERAGE
            .into_iter()
            .map(|(family, state)| GraphSnapshotCoverage {
                family: family.to_owned(),
                state: state.to_owned(),
            })
            .collect::<Vec<_>>();
        let key = format!("{}\0{}", universe.target, shard.source);
        let checker_digest = source_digests
            .get(&shard.source)
            .cloned()
            .ok_or_else(|| anyhow::format_err!("graph shard lost its checker source digest"))?;
        let interface_fingerprint = interface_fingerprints
            .get(&shard.source)
            .cloned()
            .ok_or_else(|| anyhow::format_err!("graph shard lost its interface fingerprint"))?;
        let digest = shard_digest(
            &key,
            &shard.source,
            &checker_digest,
            &interface_fingerprint,
            &shard.nodes,
            &shard.edges,
            &shard.diagnostics,
            &coverage,
            &shard.unresolved,
        )?;
        result.push(GraphSnapshotShard {
            key,
            source: shard.source,
            checker_digest,
            interface_fingerprint,
            digest,
            nodes: shard.nodes,
            edges: shard.edges,
            diagnostics: shard.diagnostics,
            coverage,
            unresolved: shard.unresolved,
        });
    }
    result.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(BuiltShards { shards: result, node_owners: id_sources })
}

fn enclosing_owner<'a>(
    definitions: &'a [(usize, FileRange, &'a str)],
    occurrence: FileRange,
) -> Option<&'a str> {
    definitions
        .iter()
        .filter(|(_, body, _)| {
            body.file_id == occurrence.file_id && body.range.contains_range(occurrence.range)
        })
        .min_by_key(|(_, body, _)| body.range.len())
        .map(|(_, _, id)| *id)
}

fn enclosing_owner_excluding<'a>(
    definitions: &'a [(usize, FileRange, &'a str)],
    occurrence: FileRange,
    excluded: usize,
) -> Option<&'a str> {
    definitions
        .iter()
        .filter(|(token_id, body, _)| {
            *token_id != excluded
                && body.file_id == occurrence.file_id
                && body.range.contains_range(occurrence.range)
        })
        .min_by_key(|(_, body, _)| body.range.len())
        .map(|(_, _, id)| *id)
}

fn reference_kind(role: StaticReferenceRole) -> &'static str {
    match role {
        StaticReferenceRole::Access => "accesses",
        StaticReferenceRole::Call => "calls",
        StaticReferenceRole::Decorate => "decorates",
        StaticReferenceRole::Export => "exports",
        StaticReferenceRole::Import => "imports",
        StaticReferenceRole::Instantiate => "instantiates",
        StaticReferenceRole::Reference => "references",
        StaticReferenceRole::Type => "type_ref",
    }
}

fn collect_unresolved(
    snap: &GlobalStateSnapshot,
    file_id: FileId,
    tokens: &[ide::TokenStaticData],
) -> anyhow::Result<Vec<GraphSnapshotUnresolved>> {
    let resolved = tokens
        .iter()
        .flat_map(|token| &token.references)
        .filter(|reference| !reference.is_definition && reference.range.file_id == file_id)
        .map(|reference| (reference.range.range, reference_kind(reference.role)))
        .collect::<Vec<_>>();
    let file = snap.analysis.parse(file_id)?;
    let root = file.syntax();
    let mut sites = Vec::<(&'static str, TextRange, &'static str)>::new();

    sites.extend(
        root.descendants().filter_map(ast::CallExpr::cast).filter_map(|call| {
            Some(("calls", call.expr()?.syntax().text_range(), "analysis-error"))
        }),
    );
    sites.extend(root.descendants().filter_map(ast::MethodCallExpr::cast).filter_map(|call| {
        Some(("calls", call.name_ref()?.syntax().text_range(), "analysis-error"))
    }));
    sites.extend(
        root.descendants()
            .filter_map(ast::MacroCall::cast)
            .filter_map(|call| Some(("calls", path_site(call.path()?)?, "macro-or-generated"))),
    );
    sites.extend(
        root.descendants().filter_map(ast::RecordExpr::cast).filter_map(|record| {
            Some(("instantiates", path_site(record.path()?)?, "analysis-error"))
        }),
    );
    sites.extend(
        root.descendants()
            .filter_map(ast::PathType::cast)
            .filter_map(|ty| Some(("type_ref", path_site(ty.path()?)?, "analysis-error"))),
    );
    sites.extend(
        root.descendants()
            .filter_map(ast::UseTree::cast)
            .filter_map(|tree| Some(("imports", path_site(tree.path()?)?, "analysis-error"))),
    );

    let mut unresolved = Vec::new();
    for token in tokens.iter().filter(|token| token.trait_member) {
        for reference in token.references.iter().filter(|reference| {
            !reference.is_definition
                && reference.range.file_id == file_id
                && reference.role == StaticReferenceRole::Call
        }) {
            unresolved.push(GraphSnapshotUnresolved {
                family: "dispatches".to_owned(),
                evidence: evidence(snap, reference.range)?,
                reason: "dynamic".to_owned(),
                candidates: Vec::new(),
            });
        }
    }
    for (family, site, reason) in sites {
        if resolved.iter().any(|(range, resolved_family)| {
            *resolved_family == family && site.contains_range(*range)
        }) {
            continue;
        }
        unresolved.push(GraphSnapshotUnresolved {
            family: family.to_owned(),
            evidence: evidence(snap, FileRange { file_id, range: site })?,
            reason: reason.to_owned(),
            candidates: Vec::new(),
        });
    }
    Ok(unresolved)
}

fn path_site(path: ast::Path) -> Option<TextRange> {
    Some(path.segment()?.syntax().text_range())
}

fn graph_node_kind(kind: SymbolInformationKind) -> &'static str {
    match kind {
        SymbolInformationKind::Module => "module",
        SymbolInformationKind::Function | SymbolInformationKind::Macro => "function",
        SymbolInformationKind::Method
        | SymbolInformationKind::StaticMethod
        | SymbolInformationKind::TraitMethod => "method",
        SymbolInformationKind::Struct | SymbolInformationKind::Union => "class",
        SymbolInformationKind::Trait => "interface",
        SymbolInformationKind::Enum => "enum",
        SymbolInformationKind::Type
        | SymbolInformationKind::TypeAlias
        | SymbolInformationKind::AssociatedType
        | SymbolInformationKind::TypeParameter => "type",
        SymbolInformationKind::Field => "field",
        SymbolInformationKind::Parameter | SymbolInformationKind::SelfParameter => "parameter",
        SymbolInformationKind::Attribute => "function",
        SymbolInformationKind::Constant
        | SymbolInformationKind::EnumMember
        | SymbolInformationKind::StaticVariable
        | SymbolInformationKind::Variable => "variable",
    }
}

fn source_path(snap: &GlobalStateSnapshot, file_id: FileId) -> String {
    let path = snap.file_id_to_file_path(file_id);
    let Some(path) = path.as_path() else {
        return path.to_string();
    };
    match path.strip_prefix(snap.config.default_root_path()) {
        Some(relative) => normalize_path(relative.as_str()),
        None => normalize_path(path.as_str()),
    }
}

fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

fn evidence(snap: &GlobalStateSnapshot, range: FileRange) -> anyhow::Result<GraphSnapshotEvidence> {
    let file = source_path(snap, range.file_id);
    let index = snap.file_line_index(range.file_id)?;
    let range = to_proto::range(&index, range.range);
    Ok(GraphSnapshotEvidence {
        file,
        start_line: range.start.line + 1,
        start_column: range.start.character + 1,
        end_line: range.end.line + 1,
        end_column: range.end.character + 1,
    })
}

fn shard_digest(
    key: &str,
    source: &str,
    checker_digest: &str,
    interface_fingerprint: &str,
    nodes: &[GraphSnapshotNode],
    edges: &[GraphSnapshotEdge],
    diagnostics: &[GraphSnapshotDiagnostic],
    coverage: &[GraphSnapshotCoverage],
    unresolved: &[GraphSnapshotUnresolved],
) -> anyhow::Result<String> {
    digest_json(&json!({
        "key": key,
        "source": source,
        "checkerDigest": checker_digest,
        "interfaceFingerprint": interface_fingerprint,
        "nodes": nodes,
        "edges": edges,
        "diagnostics": diagnostics,
        "coverage": coverage,
        "unresolved": unresolved,
    }))
}

fn snapshot_generation(
    universe: &str,
    manifest: &[GraphSnapshotManifestEntry],
) -> anyhow::Result<String> {
    digest_json(&json!({ "universe": universe, "manifest": manifest }))
}

fn digest_json<T: Serialize>(value: &T) -> anyhow::Result<String> {
    let value = serde_json::to_value(value)?;
    let mut canonical = String::new();
    write_canonical_json(&value, &mut canonical)?;
    Ok(format!("{:x}", Sha256::digest(canonical.as_bytes())))
}

fn canonical_json<T: Serialize>(value: &T) -> anyhow::Result<String> {
    let value = serde_json::to_value(value)?;
    let mut output = String::new();
    write_canonical_json(&value, &mut output)?;
    Ok(output)
}

fn write_canonical_json(value: &Value, output: &mut String) -> anyhow::Result<()> {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => output.push_str(&serde_json::to_string(value)?),
        Value::Array(values) => {
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(']');
        }
        Value::Object(values) => {
            output.push('{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort();
            for (index, key) in keys.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                output.push_str(&serde_json::to_string(key)?);
                output.push(':');
                write_canonical_json(&values[key], output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn elapsed_millis(started: Instant) -> u64 {
    started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, BTreeSet},
        fs,
        path::Path,
        sync::Arc,
    };

    use ide::FileId;
    use serde_json::json;

    use crate::lsp_ext::{
        GraphSnapshotCoverage, GraphSnapshotManifestEntry, GraphSnapshotPhases,
        GraphSnapshotProducer, GraphSnapshotShard, GraphSnapshotUniverse,
    };

    use super::{
        CachedSnapshot, ExternalGraphInput, GraphSnapshotCache, capture_external_inputs,
        checker_disk_digest, digest_json, drain_external_input_events,
        drain_source_input_events_at, ensure_cache_revision, external_input_watch_roots,
        interfaces_changed, nearest_existing_watch_root, resolve_external_tool_path, response_for,
        shard_digest, write_canonical_json,
    };

    #[test]
    fn canonical_json_sorts_keys_and_preserves_json_string_semantics() {
        let value = json!({ "z": "<>&\u{2028}\u{2029}", "a": { "d": 2, "b": 1 } });
        let mut actual = String::new();
        write_canonical_json(&value, &mut actual).unwrap();
        assert_eq!(
            actual,
            format!("{{\"a\":{{\"b\":1,\"d\":2}},\"z\":\"<>&{}{}\"}}", '\u{2028}', '\u{2029}')
        );
        assert_eq!(digest_json(&value).unwrap().len(), 64);
    }

    #[test]
    fn shard_identity_includes_the_exact_checker_source_digest() {
        let left = shard_digest(
            "target\0src/lib.rs",
            "src/lib.rs",
            "left",
            "interface",
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();
        let right = shard_digest(
            "target\0src/lib.rs",
            "src/lib.rs",
            "right",
            "interface",
            &[],
            &[],
            &[],
            &[],
            &[],
        )
        .unwrap();

        assert_ne!(left, right);
    }

    #[test]
    fn cached_snapshot_selects_noop_delta_full_and_incremental_responses() {
        let shard = GraphSnapshotShard {
            key: "target\0src/lib.rs".to_owned(),
            source: "src/lib.rs".to_owned(),
            checker_digest: "checker".to_owned(),
            interface_fingerprint: "interface".to_owned(),
            digest: "digest".to_owned(),
            nodes: Vec::new(),
            edges: Vec::new(),
            diagnostics: Vec::new(),
            coverage: vec![GraphSnapshotCoverage {
                family: "calls".to_owned(),
                state: "complete".to_owned(),
            }],
            unresolved: Vec::new(),
        };
        let cached = CachedSnapshot {
            revision: 3,
            producer: GraphSnapshotProducer::default(),
            universe: GraphSnapshotUniverse::default(),
            sequence: 4,
            generation: "new".to_owned(),
            manifest: vec![GraphSnapshotManifestEntry {
                key: shard.key.clone(),
                digest: shard.digest.clone(),
            }],
            phases: GraphSnapshotPhases {
                semantic_millis: 11,
                shard_millis: 12,
                encode_millis: 13,
                total_millis: 36,
                cache_hit: false,
            },
            shards: BTreeMap::from([(shard.key.clone(), Arc::new(shard.clone()))]),
            interface_fingerprints: BTreeMap::new(),
            source_inputs: BTreeMap::new(),
            node_owners: BTreeMap::new(),
            delta_base: Some("old".to_owned()),
            delta_upserts: vec![Arc::new(shard)],
            delta_deletes: vec!["deleted".to_owned()],
        };
        let shard_arc = cached.shards.values().next().unwrap();
        let strong_count = Arc::strong_count(shard_arc);

        let noop = response_for(&cached, Some("new"), true);
        assert!(noop.upserts.is_empty());
        assert!(noop.deletes.is_empty());
        assert_eq!(noop.base_generation.as_deref(), Some("new"));
        assert!(noop.phases.cache_hit);
        assert_eq!(Arc::strong_count(shard_arc), strong_count);

        let delta = response_for(&cached, Some("old"), false);
        assert_eq!(delta.upserts.len(), 1);
        assert_eq!(delta.deletes, ["deleted"]);
        assert_eq!(delta.base_generation.as_deref(), Some("old"));
        assert_eq!(delta.phases.total_millis, 36);
        assert!(!delta.phases.cache_hit);

        let full = response_for(&cached, Some("unrelated"), false);
        assert_eq!(full.upserts.len(), 1);
        assert!(full.deletes.is_empty());
        assert_eq!(full.base_generation, None);
    }

    #[test]
    fn invalidation_distinguishes_file_edits_from_universe_changes() {
        let mut cache = GraphSnapshotCache {
            full_rebuild: false,
            committed: Some(CachedSnapshot {
                revision: 0,
                producer: GraphSnapshotProducer::default(),
                universe: GraphSnapshotUniverse::default(),
                sequence: 0,
                generation: String::new(),
                manifest: Vec::new(),
                phases: GraphSnapshotPhases::default(),
                shards: BTreeMap::new(),
                interface_fingerprints: BTreeMap::new(),
                source_inputs: BTreeMap::new(),
                node_owners: BTreeMap::new(),
                delta_base: None,
                delta_upserts: Vec::new(),
                delta_deletes: Vec::new(),
            }),
            ..GraphSnapshotCache::default()
        };
        let file = FileId::from_raw(7);

        cache.invalidate_files(&[file]);
        assert!(!cache.full_rebuild);
        assert!(cache.dirty_files.contains(&file));
        assert!(cache.current().is_none());

        drain_source_input_events_at(&mut cache, 1);
        assert!(!cache.full_rebuild, "a VFS-invalidated edit retains its dirty-file delta");

        cache.invalidate_all();
        assert!(cache.full_rebuild);
        assert!(cache.dirty_files.is_empty());
    }

    #[test]
    fn interface_changes_are_limited_to_dirty_sources() {
        let cached = BTreeMap::from([
            ("src/a.rs".to_owned(), "same".to_owned()),
            ("src/b.rs".to_owned(), "old".to_owned()),
        ]);
        let current = BTreeMap::from([
            ("src/a.rs".to_owned(), "same".to_owned()),
            ("src/b.rs".to_owned(), "new".to_owned()),
        ]);

        assert!(!interfaces_changed(&cached, &current, &BTreeSet::from(["src/a.rs".to_owned()]),));
        assert!(interfaces_changed(&cached, &current, &BTreeSet::from(["src/b.rs".to_owned()]),));
        assert!(interfaces_changed(
            &cached,
            &current,
            &BTreeSet::from(["path-dependency/src/lib.rs".to_owned()]),
        ));
    }

    #[test]
    fn checkpoint_install_refuses_a_revision_invalidated_after_snapshot_capture() {
        let mut cache = GraphSnapshotCache::default();
        let captured = cache.revision();
        cache.revision = cache.revision.wrapping_add(1);
        cache.dirty_files.insert(FileId::from_raw(7));
        cache.full_rebuild = false;
        let dirty = cache.dirty_files.clone();
        let full_rebuild = cache.full_rebuild;

        let error =
            ensure_cache_revision(&cache, captured, "checkpoint raced an edit").unwrap_err();
        assert!(error.to_string().contains("checkpoint raced an edit"));
        assert_eq!(cache.dirty_files, dirty);
        assert_eq!(cache.full_rebuild, full_rebuild);
    }

    #[test]
    fn external_input_events_invalidate_resident_snapshots_once_drained() {
        let mut cache = GraphSnapshotCache {
            full_rebuild: false,
            external_inputs_dirty: false,
            external_input_watcher_stale: false,
            ..GraphSnapshotCache::default()
        };
        let revision = cache.revision();
        *cache.external_input_event_fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) =
            1;

        drain_external_input_events(&mut cache);

        assert_eq!(cache.revision(), revision.wrapping_add(1));
        assert_eq!(cache.external_input_revision, 1);
        assert!(cache.external_inputs_dirty);
        assert!(cache.external_input_watcher_stale);
        assert!(cache.full_rebuild);
    }

    #[test]
    fn external_input_digest_covers_hidden_generated_and_identity_inputs() {
        let directory = temp_dir::TempDir::new().unwrap();
        let root = directory.path().to_path_buf();
        fs::create_dir_all(root.join(".git")).unwrap();
        fs::create_dir_all(root.join("target")).unwrap();
        fs::write(root.join("src.rs"), "pub fn source() {}\n").unwrap();
        fs::write(root.join(".git/config"), "semantic-input-a\n").unwrap();
        fs::write(root.join("target/generated.rs"), "pub const GENERATED: u8 = 1;\n").unwrap();

        let dependency = |identity: &str| ExternalGraphInput {
            identity: identity.to_owned(),
            root: root.clone(),
            optional: false,
        };
        let initial = capture_external_inputs(&[dependency("package-a")]).unwrap().digest;
        fs::write(root.join(".git/config"), "semantic-input-b\n").unwrap();
        let hidden = capture_external_inputs(&[dependency("package-a")]).unwrap().digest;
        fs::write(root.join("target/generated.rs"), "pub const GENERATED: u8 = 2;\n").unwrap();
        let generated = capture_external_inputs(&[dependency("package-a")]).unwrap().digest;
        let renamed_package = capture_external_inputs(&[dependency("package-b")]).unwrap().digest;

        assert_ne!(initial, hidden);
        assert_ne!(hidden, generated);
        assert_ne!(generated, renamed_package);
    }

    #[test]
    fn optional_external_inputs_fence_missing_creation_and_removal() {
        let directory = temp_dir::TempDir::new().unwrap();
        let input = directory.path().join("missing/config.toml");
        let inputs = [ExternalGraphInput {
            identity: "cargo-config".to_owned(),
            root: input.clone(),
            optional: true,
        }];
        let missing = capture_external_inputs(&inputs).unwrap();
        assert_eq!(
            external_input_watch_roots(&inputs, &missing.symlink_watch_roots)
                .unwrap()
                .get(directory.path()),
            Some(&false),
        );

        fs::create_dir_all(input.parent().unwrap()).unwrap();
        fs::write(&input, "[build]\ntarget-dir = 'target'\n").unwrap();
        let created = capture_external_inputs(&inputs).unwrap();
        assert_ne!(created.digest, missing.digest);

        fs::remove_file(&input).unwrap();
        let removed = capture_external_inputs(&inputs).unwrap();
        assert_eq!(removed.digest, missing.digest);
    }

    #[test]
    fn external_tool_paths_preserve_absolute_and_anchor_explicit_relative_paths() {
        let directory = temp_dir::TempDir::new().unwrap();
        let workspace = directory.path();
        let absolute = workspace.join("toolchain/rustc");

        assert_eq!(resolve_external_tool_path(&absolute, workspace, None).unwrap(), absolute,);
        assert_eq!(
            resolve_external_tool_path(Path::new("tools/cargo"), workspace, None).unwrap(),
            workspace.join("tools/cargo"),
        );
    }

    #[test]
    fn external_tool_paths_resolve_bare_commands_from_the_workspace_environment() {
        let directory = temp_dir::TempDir::new().unwrap();
        let workspace = directory.path();
        let bin = workspace.join("toolchain/bin");
        fs::create_dir_all(&bin).unwrap();
        let cargo = bin.join(format!("cargo{}", std::env::consts::EXE_SUFFIX));
        fs::write(&cargo, "tool").unwrap();
        let mut environment = ide_db::base_db::Env::default();
        environment.set("PATH", "toolchain/bin");

        assert_eq!(
            resolve_external_tool_path(Path::new("cargo"), workspace, Some(&environment)).unwrap(),
            cargo,
        );
    }

    #[test]
    fn external_tool_paths_fail_closed_for_missing_bare_commands() {
        let directory = temp_dir::TempDir::new().unwrap();
        let workspace = directory.path();
        let mut environment = ide_db::base_db::Env::default();
        environment.set("PATH", "missing");

        let error = resolve_external_tool_path(
            Path::new("definitely-missing-rust-tool"),
            workspace,
            Some(&environment),
        )
        .unwrap_err();

        assert!(error.to_string().contains("unavailable on PATH"));
        assert!(!error.to_string().contains(&workspace.display().to_string()));
    }

    #[test]
    fn checker_source_digests_preserve_exact_disk_line_endings() {
        let directory = temp_dir::TempDir::new().unwrap();
        let source = directory.path().join("lib.rs");
        let bytes = b"pub fn answer() -> u8 {\r\n    42\r\n}\r\n";
        fs::write(&source, bytes).unwrap();

        assert_eq!(
            checker_disk_digest(&source, "pub fn answer() -> u8 {\n    42\n}\n").unwrap(),
            super::digest_bytes(bytes),
        );
        assert!(checker_disk_digest(&source, "pub fn answer() -> u8 { 0 }\n").is_err());
    }

    #[test]
    fn a_missing_symlink_target_watches_its_nearest_existing_ancestor() {
        let directory = temp_dir::TempDir::new().unwrap();
        let root = directory.path().to_path_buf();
        assert_eq!(
            nearest_existing_watch_root(root.join("missing/dependency/generated.rs")),
            Some(root),
        );
    }

    #[test]
    fn external_input_watch_roots_fence_atomic_replacement_without_recursive_ancestors() {
        let directory = temp_dir::TempDir::new().unwrap();
        let root = directory.path().join("input");
        fs::create_dir_all(&root).unwrap();
        let tool = root.join("rustc");
        fs::write(&tool, "tool").unwrap();
        let inputs = [ExternalGraphInput {
            identity: "rustc".to_owned(),
            root: tool.clone(),
            optional: false,
        }];
        let missing_ancestor = directory.path().to_path_buf();
        let supplemental = BTreeMap::from([(missing_ancestor.clone(), false)]);

        let roots = external_input_watch_roots(&inputs, &supplemental).unwrap();

        assert_eq!(roots.get(&tool), Some(&false));
        assert_eq!(roots.get(&root), Some(&false));
        assert_eq!(roots.get(&missing_ancestor), Some(&false));
    }

    #[cfg(unix)]
    #[test]
    fn external_input_digest_follows_symlinked_inputs() {
        use std::os::unix::fs::symlink;

        let directory = temp_dir::TempDir::new().unwrap();
        let root = directory.path().to_path_buf();
        fs::create_dir_all(root.join("real")).unwrap();
        fs::write(root.join("real/generated.rs"), "pub const GENERATED: u8 = 1;\n").unwrap();
        symlink(root.join("real"), root.join("linked")).unwrap();
        let dependencies = [ExternalGraphInput {
            identity: "package".to_owned(),
            root: root.clone(),
            optional: false,
        }];
        let initial = capture_external_inputs(&dependencies).unwrap().digest;
        fs::write(root.join("real/generated.rs"), "pub const GENERATED: u8 = 2;\n").unwrap();
        let changed = capture_external_inputs(&dependencies).unwrap().digest;
        assert_ne!(initial, changed);
    }

    #[cfg(unix)]
    #[test]
    fn external_input_watcher_follows_symlink_chains_and_indirect_dangling_targets() {
        use std::{os::unix::fs::symlink, sync::Mutex as StdMutex, thread, time::Duration};

        use super::external_input_watcher;

        let directory = temp_dir::TempDir::new().unwrap();
        let input = directory.path().join("input");
        let outside = directory.path().join("outside");
        let real = directory.path().join("real");
        let dependency = real.join("dependency");
        fs::create_dir_all(&input).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::create_dir_all(&dependency).unwrap();
        fs::write(dependency.join("lib.rs"), "pub fn before() {}\n").unwrap();
        symlink(&dependency, outside.join("alias")).unwrap();
        symlink(outside.join("alias"), input.join("link")).unwrap();
        let inputs = [ExternalGraphInput {
            identity: "package".to_owned(),
            root: input.clone(),
            optional: false,
        }];
        let capture = capture_external_inputs(&inputs).unwrap();
        assert_eq!(capture.symlink_watch_roots.get(&dependency), Some(&true));
        let fence = Arc::new(StdMutex::new(0));
        let watcher = external_input_watcher(
            &fence,
            external_input_watch_roots(&inputs, &capture.symlink_watch_roots).unwrap(),
        )
        .unwrap();
        fs::write(dependency.join("lib.rs"), "pub fn after() {}\n").unwrap();
        wait_for_external_event(&fence);
        drop(watcher);

        fs::remove_file(input.join("link")).unwrap();
        fs::remove_file(outside.join("alias")).unwrap();
        symlink(real.join("missing.rs"), outside.join("alias")).unwrap();
        symlink(outside.join("alias"), input.join("link")).unwrap();
        let capture = capture_external_inputs(&inputs).unwrap();
        assert_eq!(capture.symlink_watch_roots.get(&real), Some(&false));
        let fence = Arc::new(StdMutex::new(0));
        let watcher = external_input_watcher(
            &fence,
            external_input_watch_roots(&inputs, &capture.symlink_watch_roots).unwrap(),
        )
        .unwrap();
        fs::write(real.join("missing.rs"), "pub fn appeared() {}\n").unwrap();
        wait_for_external_event(&fence);
        drop(watcher);

        fs::remove_file(input.join("link")).unwrap();
        fs::remove_file(outside.join("alias")).unwrap();
        let deep = outside.join("deep");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("nested.rs"), "pub fn before() {}\n").unwrap();
        symlink(".", outside.join("repeat")).unwrap();
        symlink(outside.join("repeat/repeat/deep"), input.join("link")).unwrap();
        let capture = capture_external_inputs(&inputs).unwrap();
        assert_eq!(capture.symlink_watch_roots.get(&deep), Some(&true));
        let fence = Arc::new(StdMutex::new(0));
        let watcher = external_input_watcher(
            &fence,
            external_input_watch_roots(&inputs, &capture.symlink_watch_roots).unwrap(),
        )
        .unwrap();
        fs::write(deep.join("nested.rs"), "pub fn after() {}\n").unwrap();
        wait_for_external_event(&fence);
        drop(watcher);

        fs::remove_file(input.join("link")).unwrap();
        let blocker = outside.join("blocker");
        fs::write(&blocker, "not a directory").unwrap();
        symlink(outside.join("blocker/child"), input.join("link")).unwrap();
        let capture = capture_external_inputs(&inputs).unwrap();
        assert_eq!(capture.symlink_watch_roots.get(&blocker), Some(&false));
        assert_eq!(capture.symlink_watch_roots.get(&outside), Some(&false));
        fs::remove_file(input.join("link")).unwrap();
        symlink(outside.join("blocker/../deep"), input.join("link")).unwrap();
        let capture = capture_external_inputs(&inputs).unwrap();
        assert_ne!(capture.symlink_watch_roots.get(&deep), Some(&true));

        fn wait_for_external_event(fence: &Arc<StdMutex<u64>>) {
            for _ in 0..100 {
                if *fence.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) != 0 {
                    return;
                }
                thread::sleep(Duration::from_millis(20));
            }
            panic!("external graph-input watcher did not observe the symlink target change");
        }
    }

    #[cfg(unix)]
    #[test]
    fn external_input_digest_fences_broken_and_cyclic_symlinks_without_failing() {
        use std::os::unix::fs::symlink;

        let directory = temp_dir::TempDir::new().unwrap();
        let root = directory.path().to_path_buf();
        symlink("missing.rs", root.join("broken.rs")).unwrap();
        symlink(".", root.join("cycle")).unwrap();
        let inputs = [ExternalGraphInput {
            identity: "package".to_owned(),
            root: root.clone(),
            optional: false,
        }];

        let initial = capture_external_inputs(&inputs).unwrap();
        fs::remove_file(root.join("broken.rs")).unwrap();
        symlink("other-missing.rs", root.join("broken.rs")).unwrap();
        let changed = capture_external_inputs(&inputs).unwrap();

        assert_ne!(initial.digest, changed.digest);
        assert!(initial.symlink_watch_roots.contains_key(&root));
    }

    #[cfg(unix)]
    #[test]
    fn external_input_digest_distinguishes_non_utf8_paths() {
        use std::{ffi::OsString, os::unix::ffi::OsStringExt};

        let directory = temp_dir::TempDir::new().unwrap();
        let root = directory.path().to_path_buf();
        let first = root.join(OsString::from_vec(vec![b'f', 0x80]));
        let second = root.join(OsString::from_vec(vec![b'f', 0x81]));
        fs::write(&first, "first").unwrap();
        fs::write(&second, "second").unwrap();
        let inputs = [ExternalGraphInput { identity: "package".to_owned(), root, optional: false }];

        let initial = capture_external_inputs(&inputs).unwrap().digest;
        fs::write(&first, "changed-first").unwrap();
        let changed_first = capture_external_inputs(&inputs).unwrap().digest;
        fs::write(&first, "first").unwrap();
        fs::write(&second, "changed-second").unwrap();
        let changed_second = capture_external_inputs(&inputs).unwrap().digest;

        assert_ne!(initial, changed_first);
        assert_ne!(initial, changed_second);
    }
}
