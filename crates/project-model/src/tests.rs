use std::env::temp_dir;

use base_db::{CrateGraphBuilder, ProcMacroPaths};
use cargo_metadata::Metadata;
use cfg::{CfgAtom, CfgDiff};
use expect_test::{ExpectFile, expect_file};
use intern::sym;
use paths::{AbsPath, AbsPathBuf, Utf8Path, Utf8PathBuf};
use rustc_hash::FxHashMap;
use serde::de::DeserializeOwned;
use span::FileId;
use triomphe::Arc;

use crate::{
    CargoWorkspace, CfgOverrides, ManifestPath, ProjectJson, ProjectJsonData, ProjectWorkspace,
    RustSourceWorkspaceConfig, Sysroot, WorkspaceBuildScripts, sysroot::RustLibSrcWorkspace,
    workspace::ProjectWorkspaceKind,
};

fn load_cargo(file: &str) -> (CrateGraphBuilder, ProcMacroPaths) {
    let project_workspace = load_workspace_from_metadata(file);
    to_crate_graph(project_workspace, &mut Default::default())
}

fn load_cargo_with_overrides(
    file: &str,
    cfg_overrides: CfgOverrides,
) -> (CrateGraphBuilder, ProcMacroPaths) {
    let project_workspace =
        ProjectWorkspace { cfg_overrides, ..load_workspace_from_metadata(file) };
    to_crate_graph(project_workspace, &mut Default::default())
}

fn load_workspace_from_metadata(file: &str) -> ProjectWorkspace {
    let meta: Metadata = get_test_json_file(file);
    let manifest_path =
        ManifestPath::try_from(AbsPathBuf::try_from(meta.workspace_root.clone()).unwrap()).unwrap();
    let cargo_workspace = CargoWorkspace::new(meta, manifest_path, Default::default(), false);
    ProjectWorkspace {
        kind: ProjectWorkspaceKind::Cargo {
            cargo: cargo_workspace,
            build_scripts: WorkspaceBuildScripts::default(),
            rustc: Err(None),
            error: None,
        },
        graph_lockfile: None,
        graph_project_inputs: Vec::new(),
        graph_rustc_version: Ok("test-rustc".into()),
        cfg_overrides: Default::default(),
        sysroot: Sysroot::empty(),
        rustc_cfg: Vec::new(),
        toolchain: None,
        target: Err("target_data_layout not loaded".into()),
        extra_includes: Vec::new(),
        set_test: true,
    }
}

#[test]
fn graph_project_inputs_participate_in_workspace_identity() {
    let left = load_workspace_from_metadata("hello-world-metadata.json");
    let mut right = left.clone();
    right.graph_project_inputs.push((
        AbsPathBuf::assert_utf8(std::env::current_dir().unwrap().join("Cargo.toml")),
        Some(Arc::from(&b"[workspace]\n"[..])),
    ));

    assert!(!left.eq_ignore_build_data(&right));
    assert_ne!(left.graph_semantic_descriptor(), right.graph_semantic_descriptor());
}

fn load_rust_project(file: &str) -> (CrateGraphBuilder, ProcMacroPaths) {
    let data = get_test_json_file(file);
    let project = rooted_project_json(data);
    let sysroot = Sysroot::empty();
    let project_workspace = ProjectWorkspace {
        kind: ProjectWorkspaceKind::Json(project),
        graph_lockfile: None,
        graph_project_inputs: Vec::new(),
        graph_rustc_version: Ok("test-rustc".into()),
        sysroot,
        rustc_cfg: Vec::new(),
        toolchain: None,
        target: Err("test has no target data".into()),
        cfg_overrides: Default::default(),
        extra_includes: Vec::new(),
        set_test: true,
    };
    to_crate_graph(project_workspace, &mut Default::default())
}

#[test]
fn non_cargo_crates_receive_workspace_distinct_graph_identities() {
    let (project_graph, _) = load_rust_project("hello-world-project.json");
    let project_identities = project_graph
        .iter()
        .filter_map(|krate| project_graph[krate].extra.graph_identity.as_deref())
        .collect::<Vec<_>>();
    assert!(!project_identities.is_empty());
    assert!(project_identities.iter().all(|identity| identity.starts_with("project-json=")));

    let directory = temp_dir::TempDir::new().unwrap();
    let detached =
        ManifestPath::try_from(AbsPathBuf::assert_utf8(directory.path().join("standalone.rs")))
            .unwrap();
    let detached_workspace = ProjectWorkspace {
        kind: ProjectWorkspaceKind::DetachedFile { file: detached.clone(), cargo: None },
        graph_lockfile: None,
        graph_project_inputs: Vec::new(),
        graph_rustc_version: Ok("test-rustc".into()),
        sysroot: Sysroot::empty(),
        rustc_cfg: Vec::new(),
        toolchain: None,
        target: Err("test has no target data".into()),
        cfg_overrides: Default::default(),
        extra_includes: Vec::new(),
        set_test: true,
    };
    let (detached_graph, _) = to_crate_graph(detached_workspace, &mut Default::default());
    let identities = detached_graph
        .iter()
        .filter_map(|krate| {
            detached_graph[krate].extra.graph_identity.as_deref().map(str::to_owned)
        })
        .collect::<Vec<_>>();
    assert_eq!(identities, [format!("detached-file={detached}")]);
}

fn get_test_json_file<T: DeserializeOwned>(file: &str) -> T {
    let file = get_test_path(file);
    let data = std::fs::read_to_string(file).unwrap();
    let mut json = data.parse::<serde_json::Value>().unwrap();
    fixup_paths(&mut json);
    return serde_json::from_value(json).unwrap();

    fn fixup_paths(val: &mut serde_json::Value) {
        match val {
            serde_json::Value::String(s) => replace_root(s, true),
            serde_json::Value::Array(vals) => vals.iter_mut().for_each(fixup_paths),
            serde_json::Value::Object(kvals) => kvals.values_mut().for_each(fixup_paths),
            serde_json::Value::Null | serde_json::Value::Bool(_) | serde_json::Value::Number(_) => {
            }
        }
    }
}

fn replace_cargo(s: &mut String) {
    let path = toolchain::Tool::Cargo.path().to_string().escape_debug().collect::<String>();
    *s = s.replace(&path, "$CARGO$");
}

fn replace_root(s: &mut String, direction: bool) {
    if direction {
        let root = if cfg!(windows) { r#"C:\\ROOT\"# } else { "/ROOT/" };
        *s = s.replace("$ROOT$", root)
    } else {
        let root = if cfg!(windows) { r#"C:\\\\ROOT\\"# } else { "/ROOT/" };
        *s = s.replace(root, "$ROOT$")
    }
}

fn get_test_path(file: &str) -> Utf8PathBuf {
    let base = Utf8PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    base.join("test_data").join(file)
}

fn rooted_project_json(data: ProjectJsonData) -> ProjectJson {
    let mut root = "$ROOT$".to_owned();
    replace_root(&mut root, true);
    let path = Utf8Path::new(&root);
    let base = AbsPath::assert(path);
    ProjectJson::new(None, base, data)
}

fn to_crate_graph(
    project_workspace: ProjectWorkspace,
    file_map: &mut FxHashMap<AbsPathBuf, FileId>,
) -> (CrateGraphBuilder, ProcMacroPaths) {
    project_workspace.to_crate_graph(
        &mut {
            |path| {
                let len = file_map.len() + 1;
                Some(*file_map.entry(path.to_path_buf()).or_insert(FileId::from_raw(len as u32)))
            }
        },
        &Default::default(),
    )
}

fn check_crate_graph(crate_graph: CrateGraphBuilder, expect: ExpectFile) {
    let mut crate_graph = format!("{crate_graph:#?}");

    replace_root(&mut crate_graph, false);
    replace_cargo(&mut crate_graph);
    expect.assert_eq(&crate_graph);
}

#[test]
fn cargo_hello_world_project_model_with_wildcard_overrides() {
    let cfg_overrides = CfgOverrides {
        global: CfgDiff::new(Vec::new(), vec![CfgAtom::Flag(sym::test)]),
        selective: Default::default(),
    };
    let (crate_graph, _proc_macros) =
        load_cargo_with_overrides("hello-world-metadata.json", cfg_overrides);
    check_crate_graph(
        crate_graph,
        expect_file![
            "../test_data/output/cargo_hello_world_project_model_with_wildcard_overrides.txt"
        ],
    )
}

#[test]
fn cargo_hello_world_project_model_with_selective_overrides() {
    let cfg_overrides = CfgOverrides {
        global: Default::default(),
        selective: std::iter::once((
            "libc".to_owned(),
            CfgDiff::new(Vec::new(), vec![CfgAtom::Flag(sym::test)]),
        ))
        .collect(),
    };
    let (crate_graph, _proc_macros) =
        load_cargo_with_overrides("hello-world-metadata.json", cfg_overrides);
    check_crate_graph(
        crate_graph,
        expect_file![
            "../test_data/output/cargo_hello_world_project_model_with_selective_overrides.txt"
        ],
    )
}

#[test]
fn cargo_hello_world_project_model() {
    let (crate_graph, _proc_macros) = load_cargo("hello-world-metadata.json");
    check_crate_graph(
        crate_graph,
        expect_file!["../test_data/output/cargo_hello_world_project_model.txt"],
    )
}

#[test]
fn rust_project_hello_world_project_model() {
    let (crate_graph, _proc_macros) = load_rust_project("hello-world-project.json");
    check_crate_graph(
        crate_graph,
        expect_file!["../test_data/output/rust_project_hello_world_project_model.txt"],
    );
}

#[test]
fn rust_project_labeled_project_model() {
    // This just needs to parse.
    _ = load_rust_project("labeled-project.json");
}

#[test]
fn rust_project_cfg_groups() {
    let (crate_graph, _proc_macros) = load_rust_project("cfg-groups.json");
    check_crate_graph(crate_graph, expect_file!["../test_data/output/rust_project_cfg_groups.txt"]);
}

#[test]
fn rust_project_crate_attrs() {
    let (crate_graph, _proc_macros) = load_rust_project("crate-attrs.json");
    check_crate_graph(
        crate_graph,
        expect_file!["../test_data/output/rust_project_crate_attrs.txt"],
    );
}

#[test]
fn crate_graph_dedup_identical() {
    let (mut crate_graph, proc_macros) = load_cargo("regex-metadata.json");

    let (d_crate_graph, mut d_proc_macros) = (crate_graph.clone(), proc_macros.clone());

    crate_graph.extend(d_crate_graph.clone(), &mut d_proc_macros);
    assert!(crate_graph.iter().eq(d_crate_graph.iter()));
    assert_eq!(proc_macros, d_proc_macros);
}

#[test]
fn crate_graph_dedup() {
    let mut file_map = Default::default();

    let ripgrep_workspace = load_workspace_from_metadata("ripgrep-metadata.json");
    let (mut crate_graph, _proc_macros) = to_crate_graph(ripgrep_workspace, &mut file_map);
    assert_eq!(crate_graph.iter().count(), 71);

    let regex_workspace = load_workspace_from_metadata("regex-metadata.json");
    let (regex_crate_graph, mut regex_proc_macros) = to_crate_graph(regex_workspace, &mut file_map);
    assert_eq!(regex_crate_graph.iter().count(), 50);

    crate_graph.extend(regex_crate_graph, &mut regex_proc_macros);
    assert_eq!(crate_graph.iter().count(), 108);
}

#[test]
fn smoke_test_real_sysroot_cargo() {
    let file_map = &mut FxHashMap::<AbsPathBuf, FileId>::default();
    let meta: Metadata = get_test_json_file("hello-world-metadata.json");
    let manifest_path =
        ManifestPath::try_from(AbsPathBuf::try_from(meta.workspace_root.clone()).unwrap()).unwrap();
    let cargo_workspace = CargoWorkspace::new(meta, manifest_path, Default::default(), false);
    let mut sysroot = Sysroot::discover(
        AbsPath::assert(Utf8Path::new(env!("CARGO_MANIFEST_DIR"))),
        &Default::default(),
    );
    let cwd = AbsPathBuf::assert_utf8(temp_dir().join("smoke_test_real_sysroot_cargo"));
    std::fs::create_dir_all(&cwd).unwrap();
    let loaded_sysroot =
        sysroot.load_workspace(&RustSourceWorkspaceConfig::default_cargo(), false, &|_| ());
    if let Some(loaded_sysroot) = loaded_sysroot {
        sysroot.set_workspace(loaded_sysroot);
    }
    assert!(
        matches!(sysroot.workspace(), RustLibSrcWorkspace::Workspace { .. }),
        "got {}",
        sysroot.workspace()
    );
    let project_workspace = ProjectWorkspace {
        kind: ProjectWorkspaceKind::Cargo {
            cargo: cargo_workspace,
            build_scripts: WorkspaceBuildScripts::default(),
            rustc: Err(None),
            error: None,
        },
        graph_lockfile: None,
        graph_project_inputs: Vec::new(),
        graph_rustc_version: Ok("test-rustc".into()),
        sysroot,
        rustc_cfg: Vec::new(),
        cfg_overrides: Default::default(),
        toolchain: None,
        target: Err("target_data_layout not loaded".into()),
        extra_includes: Vec::new(),
        set_test: true,
    };
    project_workspace.to_crate_graph(
        &mut {
            |path| {
                let len = file_map.len();
                Some(*file_map.entry(path.to_path_buf()).or_insert(FileId::from_raw(len as u32)))
            }
        },
        &Default::default(),
    );
}
