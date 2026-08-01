//! Resident HIR-backed graph snapshot export.

use std::{
    collections::{BTreeMap, BTreeSet},
    time::Instant,
};

use ide::{
    AssistResolveStrategy, FileId, FileRange, Severity, StaticIndex, StaticReferenceRole,
    StaticRelationKind, SymbolInformationKind, VendoredLibrariesConfig,
};
use project_model::ProjectWorkspaceKind;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use serde_json::{Value, json};
use sha2::{Digest as _, Sha256};
use syntax::{AstNode, TextRange, ast};
use toolchain::Tool;

use crate::{
    global_state::GlobalStateSnapshot,
    lsp::to_proto,
    lsp_ext::{
        GraphSnapshotCoverage, GraphSnapshotDiagnostic, GraphSnapshotEdge, GraphSnapshotEvidence,
        GraphSnapshotManifestEntry, GraphSnapshotNode, GraphSnapshotParams, GraphSnapshotPhases,
        GraphSnapshotProducer, GraphSnapshotResult, GraphSnapshotShard, GraphSnapshotUniverse,
        GraphSnapshotUnresolved,
    },
};

const PROTOCOL_VERSION: u32 = 1;
const SCHEMA_VERSION: u32 = 1;
const COVERAGE: [(&str, &str); 15] = [
    ("contains", "complete"),
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
    ("references", "complete"),
];

pub(crate) struct GraphSnapshotCache {
    revision: u64,
    sequence: u64,
    committed: Option<CachedSnapshot>,
    dirty_files: FxHashSet<FileId>,
    full_rebuild: bool,
}

impl Default for GraphSnapshotCache {
    fn default() -> Self {
        Self {
            revision: 0,
            sequence: 0,
            committed: None,
            dirty_files: FxHashSet::default(),
            full_rebuild: true,
        }
    }
}

#[derive(Clone)]
struct CachedSnapshot {
    revision: u64,
    full: GraphSnapshotResult,
    interface_fingerprints: BTreeMap<String, String>,
    delta_base: Option<String>,
    delta_upserts: Vec<GraphSnapshotShard>,
    delta_deletes: Vec<String>,
}

impl GraphSnapshotCache {
    pub(crate) fn invalidate_all(&mut self) {
        self.revision = self.revision.wrapping_add(1);
        self.dirty_files.clear();
        self.full_rebuild = true;
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

    fn response(
        &self,
        known_generation: Option<&str>,
        cache_hit: bool,
    ) -> Option<GraphSnapshotResult> {
        let cached = self.committed.as_ref()?;
        if cached.revision != self.revision {
            return None;
        }
        Some(response_for(cached, known_generation, cache_hit))
    }
}

pub(crate) fn handle(
    snap: GlobalStateSnapshot,
    params: GraphSnapshotParams,
) -> anyhow::Result<GraphSnapshotResult> {
    if snap.workspaces.is_empty() {
        anyhow::bail!("graph snapshot is unavailable until the workspace has loaded; retry");
    }
    let started = Instant::now();
    let (
        revision,
        dirty_files,
        requested_full_rebuild,
        cached_universe,
        cached_interface_fingerprints,
        cached_relation_edges,
    ) = {
        let cache = snap.graph_snapshot_cache.lock();
        if let Some(response) = cache.response(params.known_generation.as_deref(), true) {
            return Ok(response);
        }
        (
            cache.revision,
            cache.dirty_files.iter().copied().collect::<Vec<_>>(),
            cache.full_rebuild,
            cache.committed.as_ref().map(|cached| cached.full.universe.clone()),
            cache
                .committed
                .as_ref()
                .map(|cached| cached.interface_fingerprints.clone())
                .unwrap_or_default(),
            cache
                .committed
                .as_ref()
                .map(|cached| relation_edges_by_source(&cached.full))
                .unwrap_or_default(),
        )
    };

    let semantic_started = Instant::now();
    let mut full_rebuild = requested_full_rebuild;
    let mut index = if full_rebuild {
        StaticIndex::compute(&snap.analysis, VendoredLibrariesConfig::Excluded)
    } else {
        StaticIndex::compute_files(&snap.analysis, &dirty_files)
    };
    if full_rebuild && index.files.is_empty() {
        anyhow::bail!("graph snapshot is unavailable until the crate graph has loaded; retry");
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
            index = StaticIndex::compute(&snap.analysis, VendoredLibrariesConfig::Excluded);
            if index.files.is_empty() {
                anyhow::bail!(
                    "graph snapshot is unavailable while the crate graph is reloading; retry"
                );
            }
            interface_fingerprints = compute_interface_fingerprints(&snap, &index)?;
        }
    }
    let semantic_millis = elapsed_millis(semantic_started);

    let shard_started = Instant::now();
    let universe = if full_rebuild {
        universe(&snap)?
    } else {
        cached_universe
            .ok_or_else(|| anyhow::format_err!("incremental graph snapshot lost its universe"))?
    };
    let preserved_relation_edges = (!full_rebuild).then_some(&cached_relation_edges);
    let mut shards = build_shards(&snap, index, &universe, preserved_relation_edges)?;
    let shard_millis = elapsed_millis(shard_started);

    let encode_started = Instant::now();
    let mut cache = snap.graph_snapshot_cache.lock();
    if cache.revision != revision {
        anyhow::bail!("graph snapshot was invalidated while it was being built; retry the request");
    }
    if !full_rebuild {
        let dirty_sources =
            dirty_files.iter().map(|&file_id| source_path(&snap, file_id)).collect::<BTreeSet<_>>();
        let mut merged = cache
            .committed
            .as_ref()
            .map(|cached| cached.full.upserts.clone())
            .unwrap_or_default()
            .into_iter()
            .filter(|shard| !dirty_sources.contains(&shard.source))
            .map(|shard| (shard.key.clone(), shard))
            .collect::<BTreeMap<_, _>>();
        for shard in shards {
            merged.insert(shard.key.clone(), shard);
        }
        shards = merged.into_values().collect();
    }

    let previous = cache.committed.as_ref().map(|cached| &cached.full);
    let old_manifest = previous
        .map(|result| {
            result
                .manifest
                .iter()
                .map(|entry| (entry.key.as_str(), entry.digest.as_str()))
                .collect::<BTreeMap<_, _>>()
        })
        .unwrap_or_default();
    let manifest = shards
        .iter()
        .map(|shard| GraphSnapshotManifestEntry {
            key: shard.key.clone(),
            digest: shard.digest.clone(),
        })
        .collect::<Vec<_>>();
    let new_keys = manifest.iter().map(|entry| entry.key.as_str()).collect::<BTreeSet<_>>();
    let delta_upserts = shards
        .iter()
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
    let base_generation = previous.map(|result| result.generation.clone());
    let generation = snapshot_generation(&universe.digest, &manifest)?;
    cache.sequence = cache.sequence.wrapping_add(1);
    let sequence = cache.sequence;
    let producer = producer();
    let encode_millis = elapsed_millis(encode_started);
    let full = GraphSnapshotResult {
        protocol_version: PROTOCOL_VERSION,
        schema_version: SCHEMA_VERSION,
        producer,
        universe,
        sequence,
        generation,
        base_generation: None,
        upserts: shards,
        deletes: Vec::new(),
        manifest,
        phases: GraphSnapshotPhases {
            semantic_millis,
            shard_millis,
            encode_millis,
            total_millis: elapsed_millis(started),
            cache_hit: false,
        },
    };
    let cached = CachedSnapshot {
        revision,
        full,
        interface_fingerprints,
        delta_base: base_generation,
        delta_upserts,
        delta_deletes,
    };
    let response = response_for(&cached, params.known_generation.as_deref(), false);
    cache.committed = Some(cached);
    cache.dirty_files.clear();
    cache.full_rebuild = false;
    Ok(response)
}

fn response_for(
    cached: &CachedSnapshot,
    known_generation: Option<&str>,
    cache_hit: bool,
) -> GraphSnapshotResult {
    let mut response = cached.full.clone();
    if cache_hit {
        response.phases = GraphSnapshotPhases { cache_hit: true, ..GraphSnapshotPhases::default() };
    }
    if known_generation == Some(response.generation.as_str()) {
        response.base_generation = Some(response.generation.clone());
        response.upserts.clear();
        response.deletes.clear();
    } else if known_generation == cached.delta_base.as_deref() {
        response.base_generation = cached.delta_base.clone();
        response.upserts.clone_from(&cached.delta_upserts);
        response.deletes.clone_from(&cached.delta_deletes);
    }
    response
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

    let mut configurations = Vec::new();
    for workspace in snap.workspaces.iter() {
        configurations.push(format!("target={:?}", workspace.target));
        configurations.extend(workspace.rustc_cfg.iter().map(|cfg| format!("cfg={cfg:?}")));
        configurations.push(format!("cfg-overrides={:?}", workspace.cfg_overrides));
        configurations.push(format!("set-test={}", workspace.set_test));
        if let ProjectWorkspaceKind::Cargo { cargo, .. } = &workspace.kind {
            let cargo_path =
                workspace.sysroot.tool_path(Tool::Cargo, cargo.workspace_root(), cargo.env());
            let cargo_env = cargo
                .env()
                .into_iter()
                .map(|(key, value)| (key.clone(), Some(value.clone())))
                .collect::<FxHashMap<_, _>>();
            let cargo_version = toolchain::command(&cargo_path, cargo.workspace_root(), &cargo_env)
                .arg("--version")
                .output()
                .ok()
                .filter(|output| output.status.success())
                .and_then(|output| String::from_utf8(output.stdout).ok())
                .map(|version| version.trim().to_owned())
                .unwrap_or_else(|| "unavailable".to_owned());
            configurations.push(format!("cargo-path={}", normalize_path(cargo_path.as_str())));
            configurations.push(format!("cargo-version={cargo_version}"));
            for package in cargo.packages() {
                let package = &cargo[package];
                let mut features = package.active_features.clone();
                features.sort();
                configurations.push(format!(
                    "package={}@{};manifest={};edition={:?};features={}",
                    package.name,
                    package.version,
                    normalize_path(&package.manifest.to_string()),
                    package.edition,
                    features.join(",")
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
                let macro_body = if token.kind == SymbolInformationKind::Macro {
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
                    "macroBody": macro_body,
                }))?);
            }
        }

        for reference in &token.references {
            if reference.is_definition || reference.role != StaticReferenceRole::Import {
                continue;
            }
            let source = source_path(snap, reference.range.file_id);
            if let Some(source_entries) = entries.get_mut(&source) {
                source_entries.push(canonical_json(&json!({
                    "import": token.stable_id,
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
    dirty_sources.iter().any(|source| cached.get(source) != current.get(source))
}

fn relation_edges_by_source(
    snapshot: &GraphSnapshotResult,
) -> BTreeMap<String, Vec<GraphSnapshotEdge>> {
    snapshot
        .upserts
        .iter()
        .filter_map(|shard| {
            let edges = shard
                .edges
                .iter()
                .filter(|edge| is_relation_kind(&edge.kind))
                .cloned()
                .collect::<Vec<_>>();
            (!edges.is_empty()).then(|| (shard.source.clone(), edges))
        })
        .collect()
}

fn is_relation_kind(kind: &str) -> bool {
    matches!(kind, "extends" | "implements" | "overrides" | "dispatches")
}

#[derive(Default)]
struct MutableShard {
    source: String,
    nodes: Vec<GraphSnapshotNode>,
    edges: Vec<GraphSnapshotEdge>,
    diagnostics: Vec<GraphSnapshotDiagnostic>,
    unresolved: Vec<GraphSnapshotUnresolved>,
}

fn build_shards(
    snap: &GlobalStateSnapshot,
    index: StaticIndex<'_>,
    universe: &GraphSnapshotUniverse,
    preserved_relation_edges: Option<&BTreeMap<String, Vec<GraphSnapshotEdge>>>,
) -> anyhow::Result<Vec<GraphSnapshotShard>> {
    let files = index.files;
    let relations = index.relations;
    let tokens = index.tokens.iter().map(|(_, token)| token).collect::<Vec<_>>();
    let mut sources = BTreeMap::<FileId, String>::new();
    for file in &files {
        sources.insert(file.file_id, source_path(snap, file.file_id));
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
    let mut id_sources = BTreeMap::new();

    for token in &tokens {
        let definition = token.definition;
        let external = token.external || definition.is_none();
        let definition_source =
            definition.and_then(|definition| sources.get(&definition.file_id).cloned());
        let file = if external {
            dependency_source.clone()
        } else {
            definition
                .map(|definition| source_path(snap, definition.file_id))
                .unwrap_or_else(|| dependency_source.clone())
        };
        let definition_owned = definition.is_some() && !external;
        let mut owner_sources = definition_source.into_iter().collect::<BTreeSet<_>>();
        if owner_sources.is_empty() && !definition_owned {
            owner_sources.extend(
                token
                    .references
                    .iter()
                    .filter(|reference| !reference.is_definition)
                    .filter_map(|reference| sources.get(&reference.range.file_id).cloned()),
            );
        }
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
            evidence: definition.and_then(|range| evidence(snap, range).ok()),
        });
        node_present.push(node.is_some());
        node_sources.push(node.as_ref().and_then(|_| owner_sources.first().cloned()));
        if let Some(node) = node {
            for source in owner_sources {
                id_sources.entry(node.id.clone()).or_insert_with(|| source.clone());
                shards
                    .entry(source.clone())
                    .or_insert_with(|| MutableShard { source, ..Default::default() })
                    .nodes
                    .push(node.clone());
            }
        }
    }

    let mut file_nodes = BTreeMap::new();
    for (&file_id, source) in &sources {
        let id = format!("rust-file-v1|{}:{}", source.len(), source);
        file_nodes.insert(file_id, id.clone());
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
            let owner = enclosing_owner(&definitions, reference.range)
                .map(str::to_owned)
                .or_else(|| file_nodes.get(&reference.range.file_id).cloned())
                .unwrap();
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
        let kind = match relation.kind {
            StaticRelationKind::Extends => "extends",
            StaticRelationKind::Implements => "implements",
            StaticRelationKind::Overrides => "overrides",
            StaticRelationKind::Dispatches => "dispatches",
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

    if let Some(preserved_relation_edges) = preserved_relation_edges {
        for (source, edges) in preserved_relation_edges {
            let Some(shard) = shards.get_mut(source) else {
                continue;
            };
            for edge in edges {
                let key = (edge.from.clone(), edge.to.clone(), edge.kind.clone(), source.clone());
                if edge_keys.insert(key) {
                    shard.edges.push(edge.clone());
                }
            }
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
        let digest = shard_digest(
            &key,
            &shard.source,
            &shard.nodes,
            &shard.edges,
            &shard.diagnostics,
            &coverage,
            &shard.unresolved,
        )?;
        result.push(GraphSnapshotShard {
            key,
            source: shard.source,
            digest,
            nodes: shard.nodes,
            edges: shard.edges,
            diagnostics: shard.diagnostics,
            coverage,
            unresolved: shard.unresolved,
        });
    }
    result.sort_by(|a, b| a.key.cmp(&b.key));
    Ok(result)
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
    nodes: &[GraphSnapshotNode],
    edges: &[GraphSnapshotEdge],
    diagnostics: &[GraphSnapshotDiagnostic],
    coverage: &[GraphSnapshotCoverage],
    unresolved: &[GraphSnapshotUnresolved],
) -> anyhow::Result<String> {
    digest_json(&json!({
        "key": key,
        "source": source,
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
    use std::collections::{BTreeMap, BTreeSet};

    use ide::FileId;
    use serde_json::json;

    use crate::lsp_ext::{
        GraphSnapshotCoverage, GraphSnapshotManifestEntry, GraphSnapshotPhases,
        GraphSnapshotProducer, GraphSnapshotResult, GraphSnapshotShard, GraphSnapshotUniverse,
    };

    use super::{
        CachedSnapshot, GraphSnapshotCache, digest_json, interfaces_changed, response_for,
        write_canonical_json,
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
    fn cached_snapshot_selects_noop_delta_full_and_incremental_responses() {
        let shard = GraphSnapshotShard {
            key: "target\0src/lib.rs".to_owned(),
            source: "src/lib.rs".to_owned(),
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
            full: GraphSnapshotResult {
                protocol_version: 1,
                schema_version: 1,
                producer: GraphSnapshotProducer::default(),
                universe: GraphSnapshotUniverse::default(),
                sequence: 4,
                generation: "new".to_owned(),
                base_generation: None,
                upserts: vec![shard.clone()],
                deletes: Vec::new(),
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
            },
            interface_fingerprints: BTreeMap::new(),
            delta_base: Some("old".to_owned()),
            delta_upserts: vec![shard],
            delta_deletes: vec!["deleted".to_owned()],
        };

        let noop = response_for(&cached, Some("new"), true);
        assert!(noop.upserts.is_empty());
        assert!(noop.deletes.is_empty());
        assert_eq!(noop.base_generation.as_deref(), Some("new"));
        assert!(noop.phases.cache_hit);

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
                full: GraphSnapshotResult::default(),
                interface_fingerprints: BTreeMap::new(),
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
        assert!(cache.response(None, true).is_none());

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
    }
}
