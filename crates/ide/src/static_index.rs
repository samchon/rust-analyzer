//! This module provides `StaticIndex` which is used for powering
//! read-only code browsers and emitting LSIF

use std::{collections::BTreeSet, fmt::Write as _};

use arrayvec::ArrayVec;
use either::Either;
use hir::{
    AsAssocItem, AssocItem, AssocItemContainer, Crate, HirDisplay, Impl, Module, Semantics,
    db::HirDatabase,
};
use ide_db::{
    FileId, FileRange, FxHashMap, FxHashSet, RootDatabase,
    base_db::{SourceDatabase, VfsPath},
    defs::{Definition, IdentClass},
    documentation::Documentation,
    famous_defs::FamousDefs,
    ra_fixture::RaFixtureConfig,
};
use sha2::{Digest as _, Sha256};
use syntax::{
    AstNode, AstToken, NodeOrToken, SyntaxKind, SyntaxNode, SyntaxToken, TextRange,
    ast::{self, HasName, HasVisibility},
};

use crate::navigation_target::{NavigationTarget, UpmappingResult};
use crate::{
    Analysis, Fold, HoverConfig, HoverResult, TryToNav,
    hover::{SubstTyLen, hover_for_definition},
    moniker::{MonikerResult, SymbolInformationKind, def_to_kind, def_to_moniker},
    parent_module::crates_for,
};

/// A static representation of fully analyzed source code.
///
/// The intended use-case is powering read-only code browsers and emitting LSIF/SCIP.
#[derive(Debug)]
pub struct StaticIndex<'a> {
    pub files: Vec<StaticIndexedFile>,
    pub tokens: TokenStore,
    pub relations: Vec<StaticRelation>,
    analysis: &'a Analysis,
    db: &'a RootDatabase,
    def_map: FxHashMap<Definition<'a>, TokenId>,
    graph: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct StaticRelation {
    pub file_id: FileId,
    pub from: String,
    pub to: String,
    pub to_display_name: Option<String>,
    pub to_qualified_name: Option<String>,
    pub to_signature: String,
    pub to_kind: SymbolInformationKind,
    pub to_external: bool,
    pub to_exported: bool,
    pub to_definition_file: Option<FileId>,
    pub kind: StaticRelationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StaticRelationKind {
    Extends,
    Implements,
    Overrides,
}

#[derive(Debug)]
pub struct ReferenceData {
    pub range: FileRange,
    pub is_definition: bool,
    pub role: StaticReferenceRole,
    /// Exact use-tree syntax that makes import/export alias spelling part of
    /// the graph interface invalidation fence.
    pub interface_spelling: Option<String>,
    pub export: Option<StaticExportData>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StaticExportData {
    pub exporter: String,
    pub alias_id: String,
    pub alias: String,
    pub qualified_name: String,
    pub alias_range: FileRange,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StaticReferenceRole {
    Access,
    Call,
    Decorate,
    Export,
    Import,
    Instantiate,
    Reference,
    Type,
}

#[derive(Debug)]
pub struct TokenStaticData {
    /// Versioned semantic identity used by graph-style whole-project exports.
    ///
    /// Non-local definitions are keyed by rust-analyzer's HIR-derived moniker.
    /// Locals add a syntax-structural path beneath their enclosing semantic
    /// definition. Declaration coordinates are deliberately excluded.
    pub stable_id: String,
    pub qualified_name: Option<String>,
    pub exported: bool,
    pub external: bool,
    pub local: bool,
    /// Whether this definition is an associated item declared by a trait.
    pub trait_member: bool,
    /// Whether this definition is a const function whose body is part of its semantic interface.
    pub const_function: bool,
    // FIXME: Make this have the lifetime of the database.
    pub documentation: Option<Documentation<'static>>,
    pub hover: Option<HoverResult>,
    /// The position of the token itself.
    ///
    /// For example, in `fn foo() {}` this is the position of `foo`.
    pub definition: Option<FileRange>,
    /// The position of the entire definition that this token belongs to.
    ///
    /// For example, in `fn foo() {}` this is the position from `fn`
    /// to the closing brace.
    ///
    /// This excludes trivia (whitespace/comments) other than doc
    /// comments. This differs from LSP, which includes trivia.
    ///
    /// SCIP:
    ///
    /// > source range of the nearest non-trivial enclosing AST node.
    ///
    /// <https://github.com/scip-code/scip/blob/20459645420419b3c2a10d6a9f57436abeeb273b/scip.proto#L747-L796>
    ///
    /// LSP:
    ///
    /// > range enclosing this symbol not including leading/trailing
    /// > whitespace but everything else like comments.
    ///
    /// <https://microsoft.github.io/language-server-protocol/specifications/lsp/3.18/specification/#locationLink>
    pub definition_body: Option<FileRange>,
    pub references: Vec<ReferenceData>,
    pub moniker: Option<MonikerResult>,
    pub display_name: Option<String>,
    pub signature: Option<String>,
    pub kind: SymbolInformationKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(usize);

impl TokenId {
    pub fn raw(self) -> usize {
        self.0
    }
}

#[derive(Default, Debug)]
pub struct TokenStore(Vec<TokenStaticData>);

impl TokenStore {
    pub fn insert(&mut self, data: TokenStaticData) -> TokenId {
        let id = TokenId(self.0.len());
        self.0.push(data);
        id
    }

    pub fn get_mut(&mut self, id: TokenId) -> Option<&mut TokenStaticData> {
        self.0.get_mut(id.0)
    }

    pub fn get(&self, id: TokenId) -> Option<&TokenStaticData> {
        self.0.get(id.0)
    }

    pub fn iter(self) -> impl Iterator<Item = (TokenId, TokenStaticData)> {
        self.0.into_iter().enumerate().map(|(id, data)| (TokenId(id), data))
    }

    pub fn iter_ref(&self) -> impl Iterator<Item = (TokenId, &TokenStaticData)> {
        self.0.iter().enumerate().map(|(id, data)| (TokenId(id), data))
    }
}

#[derive(Debug)]
pub struct StaticIndexedFile {
    pub file_id: FileId,
    pub folds: Vec<Fold>,
    pub tokens: Vec<(TextRange, TokenId)>,
    /// The same source file participates in more than one HIR crate/configuration context.
    pub conditional_build: bool,
}

fn all_modules(db: &dyn HirDatabase) -> Vec<Module> {
    let mut worklist: Vec<_> =
        Crate::all(db).into_iter().map(|krate| krate.root_module(db)).collect();
    let mut modules = Vec::new();

    while let Some(module) = worklist.pop() {
        modules.push(module);
        worklist.extend(module.children(db));
    }

    modules
}

fn documentation_for_definition(
    sema: &Semantics<'_, RootDatabase>,
    def: Definition<'_>,
    scope_node: &SyntaxNode,
) -> Option<Documentation<'static>> {
    let famous_defs = match &def {
        Definition::BuiltinType(_) => Some(FamousDefs(sema, sema.scope(scope_node)?.krate())),
        _ => None,
    };

    def.docs(sema.db, famous_defs.as_ref(), def.krate(sema.db)?.to_display_target(sema.db))
        .map(Documentation::into_owned)
}

// FIXME: This is a weird function
fn get_definitions<'db>(
    sema: &Semantics<'db, RootDatabase>,
    token: SyntaxToken,
    graph: bool,
) -> Option<
    ArrayVec<(Definition<'db>, Option<hir::GenericSubstitution<'db>>, StaticReferenceRole), 2>,
> {
    for token in sema.descend_into_macros_exact(token) {
        let def = IdentClass::classify_token(sema, &token).map(IdentClass::definitions);
        if let Some(defs) = def
            && !defs.is_empty()
        {
            return Some(
                defs.into_iter()
                    .map(|(def, substitution)| {
                        let role = if graph {
                            reference_role(sema.db, def, &token)
                        } else {
                            StaticReferenceRole::Reference
                        };
                        (def, substitution, role)
                    })
                    .collect(),
            );
        }
    }
    None
}

fn reference_role(
    db: &RootDatabase,
    def: Definition<'_>,
    token: &SyntaxToken,
) -> StaticReferenceRole {
    let range = token.text_range();
    let target_kind = def_to_kind(db, def);
    let mut node = token.parent();
    while let Some(current) = node {
        match current.kind() {
            SyntaxKind::ATTR => return StaticReferenceRole::Decorate,
            SyntaxKind::USE_TREE => {
                return if public_use(token) && is_terminal_path_segment(token) {
                    StaticReferenceRole::Export
                } else {
                    StaticReferenceRole::Import
                };
            }
            SyntaxKind::PATH_TYPE if is_terminal_path_segment(token) => {
                return StaticReferenceRole::Type;
            }
            SyntaxKind::MACRO_CALL if target_kind == SymbolInformationKind::Macro => {
                return StaticReferenceRole::Call;
            }
            SyntaxKind::PATH_EXPR
                if matches!(
                    target_kind,
                    SymbolInformationKind::Struct | SymbolInformationKind::EnumMember
                ) && current.text_range().end() == range.end()
                    && is_terminal_path_segment(token) =>
            {
                return StaticReferenceRole::Instantiate;
            }
            SyntaxKind::RECORD_EXPR
                if ast::RecordExpr::cast(current.clone())
                    .and_then(|expr| expr.path())
                    .is_some_and(|path| path.syntax().text_range().contains_range(range))
                    && is_terminal_path_segment(token) =>
            {
                return StaticReferenceRole::Instantiate;
            }
            SyntaxKind::CALL_EXPR
                if ast::CallExpr::cast(current.clone())
                    .and_then(|expr| expr.expr())
                    .is_some_and(|callee| callee.syntax().text_range().contains_range(range))
                    && is_terminal_path_segment(token) =>
            {
                return StaticReferenceRole::Call;
            }
            SyntaxKind::METHOD_CALL_EXPR => {
                let Some(expr) = ast::MethodCallExpr::cast(current.clone()) else {
                    node = current.parent();
                    continue;
                };
                if expr
                    .name_ref()
                    .is_some_and(|name| name.syntax().text_range().contains_range(range))
                {
                    return StaticReferenceRole::Call;
                }
                if matches!(
                    target_kind,
                    SymbolInformationKind::Struct | SymbolInformationKind::EnumMember
                ) && expr
                    .receiver()
                    .is_some_and(|receiver| receiver.syntax().text_range().contains_range(range))
                {
                    return StaticReferenceRole::Instantiate;
                }
            }
            SyntaxKind::FIELD_EXPR | SyntaxKind::RECORD_EXPR_FIELD => {
                return StaticReferenceRole::Access;
            }
            _ => {}
        }
        node = current.parent();
    }
    StaticReferenceRole::Reference
}

fn public_use(token: &SyntaxToken) -> bool {
    token
        .parent_ancestors()
        .find_map(ast::Use::cast)
        .and_then(|use_| use_.visibility())
        .is_some_and(|visibility| visibility.syntax().text() == "pub")
}

fn is_terminal_path_segment(token: &SyntaxToken) -> bool {
    let Some(segment) = token.parent_ancestors().find_map(ast::PathSegment::cast) else {
        return true;
    };
    segment
        .syntax()
        .parent()
        .and_then(|path| path.parent())
        .is_none_or(|parent| parent.kind() != SyntaxKind::PATH)
}

fn qualified_name(
    moniker: Option<&MonikerResult>,
    def: Definition<'_>,
    db: &RootDatabase,
    edition: crate::Edition,
) -> Option<String> {
    let (mut parts, local) = match moniker {
        Some(MonikerResult::Moniker(moniker)) => (
            moniker.identifier.description.iter().map(|part| part.name.clone()).collect::<Vec<_>>(),
            false,
        ),
        Some(MonikerResult::Local { enclosing_moniker }) => (
            enclosing_moniker
                .iter()
                .flat_map(|moniker| &moniker.identifier.description)
                .map(|part| part.name.clone())
                .collect::<Vec<_>>(),
            true,
        ),
        None => (Vec::new(), true),
    };
    if local {
        parts.push(def.name(db)?.display(db, edition).to_string());
    }
    (!parts.is_empty()).then(|| parts.join("::"))
}

fn stable_definition_id(
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    def: Definition<'_>,
    nav: Option<&NavigationTarget>,
    moniker: Option<&MonikerResult>,
) -> String {
    let mut id = String::from("rust-hir-v1");
    append_component(&mut id, "crate-context", &crate_context_digest(db, def));
    match moniker {
        Some(MonikerResult::Moniker(moniker)) => append_moniker(&mut id, moniker),
        Some(MonikerResult::Local { enclosing_moniker }) => {
            if let Some(moniker) = enclosing_moniker {
                append_moniker(&mut id, moniker);
            } else {
                append_component(&mut id, "scope", "unresolved");
            }
            append_local_identity(&mut id, db, sema, def, nav);
        }
        None => {
            append_component(&mut id, "scope", "builtin-or-unresolved");
            if let Some(owner) = semantic_owner(db, def)
                && let Some(krate) = owner.krate(db)
            {
                append_component(
                    &mut id,
                    "enclosing-owner",
                    &stable_id_for_definition(db, sema, krate, owner),
                );
            }
            append_local_identity(&mut id, db, sema, def, nav);
        }
    }
    append_assoc_identity(&mut id, db, sema, def);
    id
}

fn append_assoc_identity(
    id: &mut String,
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    def: Definition<'_>,
) {
    let Some(item) = def.as_assoc_item(db) else {
        return;
    };
    match item.container(db) {
        AssocItemContainer::Trait(trait_) => {
            let krate = trait_.module(db).krate(db);
            append_component(
                id,
                "assoc-owner",
                &stable_id_for_definition(db, sema, krate, trait_.into()),
            );
        }
        AssocItemContainer::Impl(impl_) => {
            let module = impl_.module(db);
            let display_target = module.krate(db).to_display_target(db);
            append_component(
                id,
                "impl-self",
                &impl_.self_ty(db).display(db, display_target).to_string(),
            );
            if let Some(trait_) = impl_.trait_(db) {
                append_component(
                    id,
                    "implemented-trait",
                    &stable_id_for_definition(db, sema, module.krate(db), trait_.into()),
                );
            } else {
                append_component(id, "implemented-trait", "inherent");
            }
        }
    }
}

fn crate_context_digest(db: &RootDatabase, def: Definition<'_>) -> String {
    let Some(krate) = def.krate(db).or_else(|| semantic_owner(db, def)?.krate(db)) else {
        return "builtin".to_owned();
    };
    crate_context_digest_for_crate(db, krate)
}

fn semantic_owner<'db>(db: &'db RootDatabase, def: Definition<'db>) -> Option<Definition<'db>> {
    match def {
        Definition::TupleField(field) => field.parent(db).try_into().ok(),
        _ => def.enclosing_definition(db),
    }
}

fn crate_context_digest_for_crate(db: &RootDatabase, krate: Crate) -> String {
    let origin = krate.origin(db);
    let root = crate_root_key(db, krate, origin.is_local());
    let mut dependencies = krate
        .dependencies(db)
        .into_iter()
        .map(|dependency| {
            format!(
                "{:?}={:?}@{:?}:{:?}:{}:{:?}",
                dependency.name,
                dependency.krate.display_name(db),
                dependency.krate.version(db),
                dependency.krate.origin(db),
                crate_root_key(db, dependency.krate, dependency.krate.origin(db).is_local(),),
                dependency.krate.cfg(db),
            )
        })
        .collect::<Vec<_>>();
    dependencies.sort();
    let context = format!(
        "root={root};name={:?};version={:?};origin={:?};edition={:?};cfg={:?};dependencies={dependencies:?}",
        krate.display_name(db),
        krate.version(db),
        origin,
        krate.edition(db),
        krate.cfg(db),
    );
    format!("{:x}", Sha256::digest(context.as_bytes()))
}

fn crate_root_key(db: &RootDatabase, krate: Crate, local: bool) -> String {
    if !local {
        return "dependency".to_owned();
    }
    let root_file = krate.root_file(db);
    let source_root = db.file_source_root(root_file).source_root_id(db);
    let root_path = db.source_root(source_root).source_root(db).path_for_file(&root_file).cloned();
    let Some(root_path) = root_path else {
        return "unavailable".to_owned();
    };
    let Some(path) = root_path.as_path() else {
        return root_path.to_string();
    };
    let cwd = &krate.base().data(db).proc_macro_cwd;
    if let Some(identity) = &krate.base().extra_data(db).graph_identity {
        let relative = path
            .strip_prefix(cwd.as_path())
            .map(|path| path.as_str().replace('\\', "/"))
            .unwrap_or_else(|| path.as_str().replace('\\', "/"));
        return format!("{identity};root={relative}");
    }
    let workspace = cwd.file_name().map(str::to_owned).unwrap_or_else(|| "workspace".to_owned());
    if let Some(relative) = path.strip_prefix(cwd.as_path()) {
        return format!("{workspace}/{}", relative.as_str().replace('\\', "/"));
    }
    let mut suffix = path
        .components()
        .rev()
        .take(3)
        .map(|component| component.as_str().to_owned())
        .collect::<Vec<_>>();
    suffix.reverse();
    format!("{workspace}/{}", suffix.join("/"))
}

fn is_effectively_exported(db: &RootDatabase, def: Definition<'_>) -> bool {
    if !has_public_visibility_chain(db, def) {
        return false;
    }
    if let Some(item) = def.as_assoc_item(db)
        && let AssocItemContainer::Impl(impl_) = item.container(db)
        && let Some(adt) = impl_.self_ty(db).as_adt()
        && !has_public_visibility_chain(db, Definition::Adt(adt))
    {
        return false;
    }
    true
}

fn has_public_visibility_chain(db: &RootDatabase, def: Definition<'_>) -> bool {
    if !matches!(def.visibility(db), Some(hir::Visibility::Public)) {
        return false;
    }
    let mut owner = def.enclosing_definition(db);
    while let Some(current) = owner {
        if current.visibility(db).is_some_and(|visibility| visibility != hir::Visibility::Public) {
            return false;
        }
        owner = current.enclosing_definition(db);
    }
    true
}

fn export_data(
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    current_crate: Option<Crate>,
    target: Definition<'_>,
    target_id: &str,
    file_id: FileId,
    range: TextRange,
    scope_node: &SyntaxNode,
) -> Option<StaticExportData> {
    let krate = current_crate?;
    let module = sema.scope(scope_node)?.module();
    let exporter = stable_id_for_definition(db, sema, krate, module.into());
    let use_tree = scope_node
        .ancestors()
        .filter_map(ast::UseTree::cast)
        .find(|tree| tree.syntax().text_range().contains_range(range))?;
    let (alias, alias_range) = if let Some(rename) = use_tree.rename()
        && let Some(name) = rename.name()
    {
        (name.text().to_string(), name.syntax().text_range())
    } else {
        (target.name(db)?.display(db, crate::Edition::CURRENT).to_string(), range)
    };
    let module_name = qualified_name(
        def_to_moniker(db, module.into(), krate).as_ref(),
        module.into(),
        db,
        crate::Edition::CURRENT,
    )
    .unwrap_or_else(|| krate.display_name(db).map(|name| name.to_string()).unwrap_or_default());
    let qualified_name =
        if module_name.is_empty() { alias.clone() } else { format!("{module_name}::{alias}") };
    let mut alias_id = String::from("rust-export-v1");
    append_component(&mut alias_id, "exporter", &exporter);
    append_component(&mut alias_id, "alias", &alias);
    append_component(&mut alias_id, "target", target_id);
    Some(StaticExportData {
        exporter,
        alias_id,
        alias,
        qualified_name,
        alias_range: FileRange { file_id, range: alias_range },
    })
}

fn append_moniker(id: &mut String, moniker: &crate::Moniker) {
    append_component(id, "package", &moniker.package_information.name);
    append_component(id, "version", moniker.package_information.version.as_deref().unwrap_or(""));
    append_component(id, "repository", moniker.package_information.repo.as_deref().unwrap_or(""));
    append_component(id, "crate", &moniker.identifier.crate_name);
    for descriptor in &moniker.identifier.description {
        append_component(id, "kind", &format!("{:?}", descriptor.desc));
        append_component(id, "name", &descriptor.name);
    }
}

fn append_local_identity(
    id: &mut String,
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    def: Definition<'_>,
    nav: Option<&NavigationTarget>,
) {
    append_component(id, "local-kind", &format!("{:?}", def_to_kind(db, def)));
    append_component(
        id,
        "local-name",
        &def.name(db)
            .map(|name| name.display_no_db(crate::Edition::CURRENT).to_string())
            .unwrap_or_default(),
    );

    let Some(nav) = nav else {
        append_component(id, "syntax", "unavailable");
        return;
    };
    let root = sema.parse_guess_edition(nav.file_id).syntax().clone();
    let range = nav.focus_or_full_range();
    let mut node = match root.covering_element(range) {
        NodeOrToken::Node(node) => Some(node),
        NodeOrToken::Token(token) => token.parent(),
    };
    if let Some(name) = node.clone().and_then(|node| node.ancestors().find_map(ast::Name::cast)) {
        let name_text = name.syntax().text().to_string();
        let parent_kind = name.syntax().parent().map(|parent| parent.kind());
        let scope = semantic_owner(db, def)
            .and_then(|owner| owner.try_to_nav(sema))
            .map(UpmappingResult::call_site)
            .filter(|owner| owner.file_id == nav.file_id)
            .map(|owner| owner.full_range)
            .unwrap_or_else(|| root.text_range());
        let ordinal = root
            .descendants()
            .filter_map(ast::Name::cast)
            .filter(|candidate| scope.contains_range(candidate.syntax().text_range()))
            .filter(|candidate| candidate.syntax().text().to_string() == name_text)
            .filter(|candidate| {
                candidate.syntax().parent().map(|parent| parent.kind()) == parent_kind
            })
            .take_while(|candidate| candidate.syntax().text_range() != name.syntax().text_range())
            .count();
        append_component(
            id,
            "syntax",
            &format!("{:?}:{ordinal}", parent_kind.unwrap_or(SyntaxKind::NAME)),
        );
        return;
    }

    let scope = semantic_owner(db, def)
        .and_then(|owner| owner.try_to_nav(sema))
        .map(UpmappingResult::call_site)
        .filter(|owner| owner.file_id == nav.file_id)
        .map(|owner| owner.full_range)
        .unwrap_or_else(|| root.text_range());
    let mut path = Vec::new();
    while let Some(current) = node {
        if current == root || current.text_range() == scope {
            break;
        }
        let kind = current.kind();
        let ordinal = std::iter::successors(current.prev_sibling(), SyntaxNode::prev_sibling)
            .filter(|sibling| sibling.kind() == kind)
            .count();
        path.push(format!("{kind:?}:{ordinal}"));
        node = current.parent();
    }
    path.reverse();
    append_component(id, "syntax", &path.join("/"));
}

fn append_component(id: &mut String, label: &str, value: &str) {
    write!(id, "|{}:{}={}:{}", label.len(), label, value.len(), value).unwrap();
}

fn semantic_relations(db: &RootDatabase, files: Option<&FxHashSet<FileId>>) -> Vec<StaticRelation> {
    let sema = Semantics::new(db);
    let mut relations = BTreeSet::new();
    for module in all_modules(db) {
        let file_id = module.definition_source_file_id(db).original_file(db).file_id(db);
        if files.is_some_and(|files| !files.contains(&file_id)) {
            continue;
        }
        let source_root = db.file_source_root(file_id).source_root_id(db);
        if db.source_root(source_root).source_root(db).is_library {
            continue;
        }
        let krate = module.krate(db);
        for trait_ in
            module.declarations(db).into_iter().filter_map(|declaration| match declaration {
                hir::ModuleDef::Trait(trait_) => Some(trait_),
                _ => None,
            })
        {
            for supertrait in trait_.direct_supertraits(db) {
                relations.insert(relation(
                    db,
                    &sema,
                    krate,
                    file_id,
                    trait_.into(),
                    supertrait.into(),
                    StaticRelationKind::Extends,
                ));
            }
        }
        for impl_ in Impl::all_in_module(db, module) {
            let Some(trait_) = impl_.trait_(db) else {
                continue;
            };
            if let Some(adt) = impl_.self_ty(db).as_adt() {
                relations.insert(relation(
                    db,
                    &sema,
                    krate,
                    file_id,
                    adt.into(),
                    trait_.into(),
                    StaticRelationKind::Implements,
                ));
            }
            for item in impl_.items(db) {
                let Some(name) = item.name(db) else {
                    continue;
                };
                let Some(trait_item) = trait_.items(db).into_iter().find(|candidate| {
                    same_assoc_item_kind(item, *candidate)
                        && candidate.name(db).as_ref() == Some(&name)
                }) else {
                    continue;
                };
                let implementation = assoc_item_definition(item);
                let declaration = assoc_item_definition(trait_item);
                relations.insert(relation(
                    db,
                    &sema,
                    krate,
                    file_id,
                    implementation,
                    declaration,
                    StaticRelationKind::Overrides,
                ));
            }
        }
    }
    relations.into_iter().collect()
}

fn relation(
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    from_crate: Crate,
    file_id: FileId,
    from: Definition<'_>,
    to: Definition<'_>,
    kind: StaticRelationKind,
) -> StaticRelation {
    let nav = to.try_to_nav(sema).map(UpmappingResult::call_site);
    let moniker = def_to_moniker(db, to, from_crate);
    let edition = from_crate.edition(db);
    let display_target = from_crate.to_display_target(db);
    StaticRelation {
        file_id,
        from: stable_id_for_definition(db, sema, from_crate, from),
        to: stable_definition_id(db, sema, to, nav.as_ref(), moniker.as_ref()),
        to_display_name: to.name(db).map(|name| name.display(db, edition).to_string()),
        to_qualified_name: qualified_name(moniker.as_ref(), to, db, edition),
        to_signature: to.label(db, display_target),
        to_kind: def_to_kind(db, to),
        to_external: nav.as_ref().is_none_or(|nav| {
            let source_root = db.file_source_root(nav.file_id).source_root_id(db);
            db.source_root(source_root).source_root(db).is_library
        }),
        to_exported: is_effectively_exported(db, to),
        to_definition_file: nav.as_ref().map(|nav| nav.file_id),
        kind,
    }
}

fn stable_id_for_definition(
    db: &RootDatabase,
    sema: &Semantics<'_, RootDatabase>,
    from_crate: Crate,
    def: Definition<'_>,
) -> String {
    let nav = def.try_to_nav(sema).map(UpmappingResult::call_site);
    let moniker = def_to_moniker(db, def, from_crate);
    stable_definition_id(db, sema, def, nav.as_ref(), moniker.as_ref())
}

fn same_assoc_item_kind(left: AssocItem, right: AssocItem) -> bool {
    matches!(
        (left, right),
        (AssocItem::Function(_), AssocItem::Function(_))
            | (AssocItem::Const(_), AssocItem::Const(_))
            | (AssocItem::TypeAlias(_), AssocItem::TypeAlias(_))
    )
}

fn assoc_item_definition<'db>(item: AssocItem) -> Definition<'db> {
    match item {
        AssocItem::Function(item) => item.into(),
        AssocItem::Const(item) => item.into(),
        AssocItem::TypeAlias(item) => item.into(),
    }
}

#[derive(Clone, Copy)]
pub enum VendoredLibrariesConfig<'a> {
    Included { workspace_root: &'a VfsPath },
    Excluded,
}

impl<'a> StaticIndex<'a> {
    fn add_file(&mut self, file_id: FileId) {
        let graph = self.graph;
        let mut crates = crates_for(self.db, file_id);
        let conditional_build = graph && crates.len() > 1;
        let current_crate = if graph {
            crates
                .into_iter()
                .map(Into::into)
                .min_by_key(|krate| crate_context_digest_for_crate(self.db, *krate))
        } else {
            crates.pop().map(Into::into)
        };
        let folds = self.analysis.folding_ranges(file_id, true).unwrap();
        // hovers
        let sema = hir::Semantics::new(self.db);
        let root = sema.parse_guess_edition(file_id).syntax().clone();
        let edition = sema.attach_first_edition(file_id).edition(sema.db);
        let display_target = match sema.first_crate(file_id) {
            Some(krate) => krate.to_display_target(sema.db),
            None => return,
        };
        let tokens = root.descendants_with_tokens().filter_map(|it| match it {
            syntax::NodeOrToken::Node(_) => None,
            syntax::NodeOrToken::Token(it) => Some(it),
        });
        let hover_config = HoverConfig {
            links_in_hover: true,
            memory_layout: None,
            documentation: true,
            keywords: true,
            format: crate::HoverDocFormat::Markdown,
            max_trait_assoc_items_count: None,
            max_fields_count: Some(5),
            max_enum_variants_count: Some(5),
            max_subst_ty_len: SubstTyLen::Unlimited,
            show_drop_glue: true,
            ra_fixture: RaFixtureConfig::default(),
        };
        let mut result = StaticIndexedFile { file_id, folds, tokens: vec![], conditional_build };

        let mut add_token = |def: Definition<'a>,
                             range: TextRange,
                             scope_node: &SyntaxNode,
                             role: StaticReferenceRole| {
            let id = if let Some(it) = self.def_map.get(&def) {
                *it
            } else {
                let nav = def.try_to_nav(&sema).map(UpmappingResult::call_site);
                let moniker = current_crate.and_then(|cc| def_to_moniker(self.db, def, cc));
                let local = matches!(moniker, Some(MonikerResult::Local { .. }));
                let it = self.tokens.insert(TokenStaticData {
                    stable_id: if graph {
                        stable_definition_id(self.db, &sema, def, nav.as_ref(), moniker.as_ref())
                    } else {
                        String::new()
                    },
                    qualified_name: graph
                        .then(|| qualified_name(moniker.as_ref(), def, self.db, edition))
                        .flatten(),
                    exported: graph && is_effectively_exported(self.db, def),
                    external: graph
                        && nav.as_ref().is_some_and(|nav| {
                            let source_root =
                                self.db.file_source_root(nav.file_id).source_root_id(self.db);
                            self.db.source_root(source_root).source_root(self.db).is_library
                        }),
                    local: graph && local,
                    trait_member: graph
                        && def.as_assoc_item(self.db).is_some_and(|item| {
                            matches!(item.container(self.db), AssocItemContainer::Trait(_))
                        }),
                    const_function: graph
                        && matches!(def, Definition::Function(function) if function.is_const(self.db)),
                    documentation: documentation_for_definition(&sema, def, scope_node),
                    hover: Some(hover_for_definition(
                        &sema,
                        file_id,
                        def,
                        None,
                        scope_node,
                        None,
                        false,
                        &hover_config,
                        edition,
                        display_target,
                    )),
                    definition: nav.as_ref().map(|it| FileRange {
                        file_id: it.file_id,
                        range: it.focus_or_full_range(),
                    }),
                    definition_body: nav.as_ref().map(|it| FileRange {
                        file_id: it.file_id,
                        range: definition_range_excluding_trivia(&sema, it.file_id, it.full_range),
                    }),
                    references: vec![],
                    moniker,
                    display_name: def
                        .name(self.db)
                        .map(|name| name.display(self.db, edition).to_string()),
                    signature: Some(def.label(self.db, display_target)),
                    kind: def_to_kind(self.db, def),
                });
                self.def_map.insert(def, it);
                it
            };
            let token = self.tokens.get_mut(id).unwrap();
            let export = (graph && role == StaticReferenceRole::Export)
                .then(|| {
                    export_data(
                        self.db,
                        &sema,
                        current_crate,
                        def,
                        &token.stable_id,
                        file_id,
                        range,
                        scope_node,
                    )
                })
                .flatten();
            token.references.push(ReferenceData {
                range: FileRange { range, file_id },
                is_definition: match def.try_to_nav(&sema).map(UpmappingResult::call_site) {
                    Some(it) => it.file_id == file_id && it.focus_or_full_range() == range,
                    None => false,
                },
                role,
                interface_spelling: (graph
                    && matches!(role, StaticReferenceRole::Export | StaticReferenceRole::Import))
                .then(|| {
                    scope_node
                        .ancestors()
                        .find_map(ast::UseTree::cast)
                        .map(|tree| tree.syntax().text().to_string())
                })
                .flatten(),
                export,
            });
            result.tokens.push((range, id));
        };

        if let Some(module) = sema.file_to_module_def(file_id) {
            let def = Definition::Module(module);
            let range = root.text_range();
            add_token(def, range, &root, StaticReferenceRole::Reference);
        }

        for token in tokens {
            let range = token.text_range();
            let node = token.parent().unwrap();
            match hir::attach_db(self.db, || get_definitions(&sema, token.clone(), graph)) {
                Some(defs) => {
                    for (def, _, role) in defs {
                        add_token(def, range, &node, role);
                    }
                }
                None => continue,
            };
        }
        self.files.push(result);
    }

    pub fn compute(
        analysis: &'a Analysis,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) -> StaticIndex<'a> {
        Self::compute_inner(analysis, vendored_libs_config, false)
    }

    /// Computes the HIR-enriched graph index used by resident graph snapshots.
    pub fn compute_graph(
        analysis: &'a Analysis,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) -> StaticIndex<'a> {
        Self::compute_inner(analysis, vendored_libs_config, true)
    }

    fn compute_inner(
        analysis: &'a Analysis,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
        graph: bool,
    ) -> StaticIndex<'a> {
        let db = &analysis.db;
        hir::attach_db(db, || {
            let work = all_modules(db).into_iter().filter(|module| {
                let file_id = module.definition_source_file_id(db).original_file(db);
                let source_root =
                    db.file_source_root(file_id.file_id(&analysis.db)).source_root_id(db);
                let source_root = db.source_root(source_root).source_root(db);
                let is_vendored = match vendored_libs_config {
                    VendoredLibrariesConfig::Included { workspace_root } => source_root
                        .path_for_file(&file_id.file_id(&analysis.db))
                        .is_some_and(|module_path| module_path.starts_with(workspace_root)),
                    VendoredLibrariesConfig::Excluded => false,
                };

                !source_root.is_library || is_vendored
            });
            let mut this = StaticIndex {
                files: vec![],
                tokens: Default::default(),
                relations: Vec::new(),
                analysis,
                db,
                def_map: Default::default(),
                graph,
            };
            let mut visited_files = FxHashSet::default();
            for module in work {
                let file_id =
                    module.definition_source_file_id(db).original_file(db).file_id(&analysis.db);
                if visited_files.contains(&file_id) {
                    continue;
                }
                this.add_file(file_id);
                visited_files.insert(file_id);
            }
            if graph {
                this.relations = semantic_relations(db, None);
            }
            this
        })
    }

    /// Computes semantic index data only for the requested source files.
    ///
    /// Referenced definitions are still resolved through the same immutable
    /// database revision, so callers can rebuild source-owned shards without
    /// fanning out through LSP requests or walking unrelated files.
    pub fn compute_graph_files(analysis: &'a Analysis, file_ids: &[FileId]) -> StaticIndex<'a> {
        let db = &analysis.db;
        hir::attach_db(db, || {
            let mut this = StaticIndex {
                files: vec![],
                tokens: Default::default(),
                relations: Vec::new(),
                analysis,
                db,
                def_map: Default::default(),
                graph: true,
            };
            let mut visited_files = FxHashSet::default();
            for &file_id in file_ids {
                if visited_files.insert(file_id) {
                    this.add_file(file_id);
                }
            }
            this.relations = semantic_relations(db, Some(&visited_files));
            this
        })
    }

    /// Source files owned by the same graph traversal, without indexing their
    /// declarations or references. Used to validate persisted checkpoints.
    pub fn graph_source_files(
        analysis: &'a Analysis,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) -> Vec<FileId> {
        let db = &analysis.db;
        hir::attach_db(db, || {
            let mut files = all_modules(db)
                .into_iter()
                .filter_map(|module| {
                    let file_id = module.definition_source_file_id(db).original_file(db);
                    let file_id = file_id.file_id(db);
                    let source_root = db.file_source_root(file_id).source_root_id(db);
                    let source_root = db.source_root(source_root).source_root(db);
                    let is_vendored = match vendored_libs_config {
                        VendoredLibrariesConfig::Included { workspace_root } => source_root
                            .path_for_file(&file_id)
                            .is_some_and(|module_path| module_path.starts_with(workspace_root)),
                        VendoredLibrariesConfig::Excluded => false,
                    };
                    (!source_root.is_library || is_vendored).then_some(file_id)
                })
                .collect::<Vec<_>>();
            files.sort_by_key(|file_id| file_id.index());
            files.dedup();
            files
        })
    }
}

fn definition_range_excluding_trivia(
    sema: &Semantics<'_, RootDatabase>,
    file_id: FileId,
    range: TextRange,
) -> TextRange {
    let root = sema.parse_guess_edition(file_id).syntax().clone();
    if range == root.text_range() {
        return range;
    }
    if !root.text_range().contains_range(range) {
        return range;
    }

    let element = root.covering_element(range);
    let tokens = match element {
        NodeOrToken::Node(node) => Either::Left(node.descendants_with_tokens().filter_map(|it| {
            let token = it.into_token()?;
            range.contains_range(token.text_range()).then_some(token)
        })),
        NodeOrToken::Token(token) => Either::Right(std::iter::once(token)),
    };

    let mut first = None;
    let mut last = None;
    for token in tokens {
        if first.is_none() && !is_leading_trivia_excluding_docs(&token) {
            first = Some(token.clone());
        }
        if !is_trailing_trivia(&token) {
            last = Some(token);
        }
    }

    match (first, last) {
        (Some(first), Some(last)) => {
            TextRange::new(first.text_range().start(), last.text_range().end())
        }
        _ => range,
    }
}

fn is_leading_trivia_excluding_docs(token: &SyntaxToken) -> bool {
    match token.kind() {
        SyntaxKind::WHITESPACE => true,
        SyntaxKind::COMMENT => ast::Comment::cast(token.clone()).is_none_or(|it| !it.is_outer()),
        _ => false,
    }
}

fn is_trailing_trivia(token: &SyntaxToken) -> bool {
    matches!(token.kind(), SyntaxKind::WHITESPACE | SyntaxKind::COMMENT)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{StaticIndex, fixture};
    use ide_db::{FileRange, FxHashMap, FxHashSet, base_db::VfsPath};
    use syntax::TextSize;

    use super::{StaticReferenceRole, StaticRelationKind, VendoredLibrariesConfig};

    fn graph_identities(ra_fixture: &str) -> Vec<(Option<String>, String)> {
        let (analysis, _) = fixture::annotations_without_marker(ra_fixture);
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let mut identities = index
            .tokens
            .iter()
            .filter_map(|(_, token)| {
                token.definition?;
                Some((token.qualified_name, token.stable_id))
            })
            .collect::<Vec<_>>();
        identities.sort();
        identities
    }

    fn check_all_ranges(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) {
        let (analysis, ranges) = fixture::annotations_without_marker(ra_fixture);
        let s = StaticIndex::compute(&analysis, vendored_libs_config);
        let mut range_set: FxHashSet<_> = ranges.iter().map(|it| it.0).collect();
        for f in s.files {
            for (range, _) in f.tokens {
                if range.start() == TextSize::from(0) {
                    // ignore whole file range corresponding to module definition
                    continue;
                }
                let it = FileRange { file_id: f.file_id, range };
                if !range_set.contains(&it) {
                    panic!("additional range {it:?}");
                }
                range_set.remove(&it);
            }
        }
        if !range_set.is_empty() {
            panic!("unfound ranges {range_set:?}");
        }
    }

    #[track_caller]
    fn check_definitions(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) {
        let (analysis, ranges) = fixture::annotations_without_marker(ra_fixture);
        let s = StaticIndex::compute(&analysis, vendored_libs_config);
        let mut range_set: FxHashSet<_> = ranges.iter().map(|it| it.0).collect();
        for (_, t) in s.tokens.iter() {
            if let Some(t) = t.definition {
                if t.range.start() == TextSize::from(0) {
                    // ignore definitions that are whole of file
                    continue;
                }
                if !range_set.contains(&t) {
                    panic!("additional definition {t:?}");
                }
                range_set.remove(&t);
            }
        }
        if !range_set.is_empty() {
            panic!("unfound definitions {range_set:?}");
        }
    }

    #[track_caller]
    fn check_references(
        #[rust_analyzer::rust_fixture] ra_fixture: &str,
        vendored_libs_config: VendoredLibrariesConfig<'_>,
    ) {
        let (analysis, ranges) = fixture::annotations_without_marker(ra_fixture);
        let s = StaticIndex::compute(&analysis, vendored_libs_config);
        let mut range_set: FxHashMap<_, i32> = ranges.iter().map(|it| (it.0, 0)).collect();

        // Make sure that all references have at least one range. We use a HashMap instead of a
        // a HashSet so that we can have more than one reference at the same range.
        for (_, t) in s.tokens.iter() {
            for r in &t.references {
                if r.is_definition {
                    continue;
                }
                if r.range.range.start() == TextSize::from(0) {
                    // ignore whole file range corresponding to module definition
                    continue;
                }
                match range_set.entry(r.range) {
                    std::collections::hash_map::Entry::Occupied(mut entry) => {
                        let count = entry.get_mut();
                        *count += 1;
                    }
                    std::collections::hash_map::Entry::Vacant(_) => {
                        panic!("additional reference {r:?}");
                    }
                }
            }
        }
        for (range, count) in range_set.iter() {
            if *count == 0 {
                panic!("unfound reference {range:?}");
            }
        }
    }

    #[test]
    fn field_initialization() {
        check_references(
            r#"
struct Point {
    x: f64,
     //^^^
    y: f64,
     //^^^
}
    fn foo() {
        let x = 5.;
        let y = 10.;
        let mut p = Point { x, y };
                  //^^^^^   ^  ^
        p.x = 9.;
      //^ ^
        p.y = 10.;
      //^ ^
    }
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn struct_and_enum() {
        check_all_ranges(
            r#"
struct Foo;
     //^^^
enum E { X(Foo) }
   //^   ^ ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
        check_definitions(
            r#"
struct Foo;
     //^^^
enum E { X(Foo) }
   //^   ^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );

        check_references(
            r#"
struct Foo;
enum E { X(Foo) }
   //      ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn multi_crate() {
        check_definitions(
            r#"
//- /workspace/main.rs crate:main deps:foo


use foo::func;

fn main() {
 //^^^^
    func();
}
//- /workspace/foo/lib.rs crate:foo

pub func() {

}
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn vendored_crate() {
        check_all_ranges(
            r#"
//- /workspace/main.rs crate:main deps:external,vendored
struct Main(i32);
     //^^^^ ^^^

//- /external/lib.rs new_source_root:library crate:external@0.1.0,https://a.b/foo.git library
struct ExternalLibrary(i32);

//- /workspace/vendored/lib.rs new_source_root:library crate:vendored@0.1.0,https://a.b/bar.git library
struct VendoredLibrary(i32);
     //^^^^^^^^^^^^^^^ ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn vendored_crate_excluded() {
        check_all_ranges(
            r#"
//- /workspace/main.rs crate:main deps:external,vendored
struct Main(i32);
     //^^^^ ^^^

//- /external/lib.rs new_source_root:library crate:external@0.1.0,https://a.b/foo.git library
struct ExternalLibrary(i32);

//- /workspace/vendored/lib.rs new_source_root:library crate:vendored@0.1.0,https://a.b/bar.git library
struct VendoredLibrary(i32);
"#,
            VendoredLibrariesConfig::Excluded,
        )
    }

    #[test]
    fn derives() {
        check_all_ranges(
            r#"
//- minicore:derive
#[rustc_builtin_macro]
//^^^^^^^^^^^^^^^^^^^
pub macro Copy {}
        //^^^^
#[derive(Copy)]
//^^^^^^ ^^^^
struct Hello(i32);
     //^^^^^ ^^^
"#,
            VendoredLibrariesConfig::Included {
                workspace_root: &VfsPath::new_virtual_path("/workspace".to_owned()),
            },
        );
    }

    #[test]
    fn graph_identity_is_semantic_and_position_invariant() {
        let compact = graph_identities(
            r#"
//- /workspace/lib.rs crate:main@1.2.3,https://example.com/main.git
pub struct Service;
impl Service {
    pub fn run(&self, input: i32) -> i32 {
        let output = input + 1;
        let scratch = 0;
        let _ = scratch;
        output
    }
}
"#,
        );
        let moved = graph_identities(
            r#"
//- /workspace/lib.rs crate:main@1.2.3,https://example.com/main.git


pub struct Service;

impl Service {
    pub fn run(
        &self,
        input: i32,
    ) -> i32 {
        let scratch = 0;
        let _ = scratch;
        let output = input + 1;
        output
    }
}
"#,
        );

        assert_eq!(compact, moved);
        assert!(compact.iter().all(|(_, id)| id.starts_with("rust-hir-v1|")));
        assert!(
            compact.iter().any(|(name, _)| name.as_deref() == Some("impl::Service::run")),
            "{compact:#?}"
        );
        assert!(
            compact.iter().any(|(name, _)| name.as_deref() == Some("impl::Service::run::input")),
            "{compact:#?}"
        );
    }

    #[test]
    fn graph_relations_come_from_hir_traits_and_impls() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
pub trait Parent {}
pub trait Child: Parent {}

pub struct Service;
impl Child for Service {}

pub trait Render {
    fn render(&self) -> String;
}
impl Render for Service {
    fn render(&self) -> String { String::new() }
}
"#,
        );
        let relations =
            StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded).relations;

        assert!(relations.iter().any(|relation| relation.kind == StaticRelationKind::Extends));
        assert!(relations.iter().any(|relation| relation.kind == StaticRelationKind::Implements));
        assert!(relations.iter().any(|relation| relation.kind == StaticRelationKind::Overrides));
        assert!(relations.iter().all(|relation| {
            relation.from.starts_with("rust-hir-v1|") && relation.to.starts_with("rust-hir-v1|")
        }));
        assert!(relations.iter().all(|relation| {
            relation.to_display_name.is_some()
                && relation.to_qualified_name.is_some()
                && !relation.to_signature.is_empty()
                && !relation.to_external
                && relation.to_exported
                && relation.to_definition_file.is_some()
        }));
    }

    #[test]
    fn graph_relations_describe_external_endpoints() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- minicore: fmt
//- /workspace/lib.rs crate:main
struct Value;
impl core::fmt::Debug for Value {}
"#,
        );
        let relations =
            StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded).relations;
        let relation = relations
            .iter()
            .find(|relation| relation.kind == StaticRelationKind::Implements)
            .unwrap();

        assert_eq!(relation.to_display_name.as_deref(), Some("Debug"));
        assert_eq!(relation.to_kind, crate::SymbolInformationKind::Trait);
        assert!(relation.to_qualified_name.as_deref().is_some_and(|name| name.ends_with("Debug")));
        assert!(relation.to_signature.contains("Debug"));
        assert!(relation.to_external);
        assert!(relation.to_exported);
        assert!(relation.to_definition_file.is_some());
    }

    #[test]
    fn graph_identity_includes_crate_root_context() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/one/lib.rs crate:one@1.0.0,https://example.com/shared.git
pub fn execute() {}
//- /workspace/two/lib.rs crate:two@1.0.0,https://example.com/shared.git
pub fn execute() {}
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let ids = index
            .tokens
            .iter()
            .filter(|(_, token)| {
                token.definition.is_some() && token.display_name.as_deref() == Some("execute")
            })
            .map(|(_, token)| token.stable_id.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), 2, "{ids:#?}");
        assert!(ids.iter().all(|id| id.contains("crate-context")));
    }

    #[test]
    fn graph_identity_does_not_change_when_a_local_target_is_added() {
        fn execute_id(fixture_text: &str) -> String {
            let (analysis, _) = fixture::annotations_without_marker(fixture_text);
            StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded)
                .tokens
                .iter()
                .find(|(_, token)| {
                    token.definition.is_some() && token.display_name.as_deref() == Some("execute")
                })
                .unwrap()
                .1
                .stable_id
                .clone()
        }

        let before = execute_id(
            r#"
//- /workspace/src/lib.rs crate:main
pub fn execute() {}
"#,
        );
        let after = execute_id(
            r#"
//- /workspace/src/lib.rs crate:main
pub fn execute() {}
//- /workspace/src/main.rs crate:bin
fn main() {}
"#,
        );

        assert_eq!(before, after);
    }

    #[test]
    fn graph_identity_namespaces_tuple_fields_by_their_owner() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
pub fn first(value: (u8,)) -> u8 { value.0 }
pub fn second(value: (u8,)) -> u8 { value.0 }
"#,
        );
        let ids = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded)
            .tokens
            .iter()
            .filter(|(_, token)| {
                token.display_name.as_deref() == Some("0")
                    && token.kind == super::SymbolInformationKind::Field
            })
            .map(|(_, token)| token.stable_id.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), 2, "{ids:#?}");
        assert!(ids.iter().all(|id| id.contains("enclosing-owner")), "{ids:#?}");
    }

    #[test]
    fn graph_exports_follow_effective_visibility_and_public_uses() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
mod private { pub fn hidden() {} }
pub fn exposed() {}
pub use private::hidden as alias;
use private::hidden as local_alias;
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let tokens = index.tokens.iter().map(|(_, token)| token).collect::<Vec<_>>();
        let hidden = tokens.iter().find(|token| token.display_name.as_deref() == Some("hidden"));
        let exposed = tokens.iter().find(|token| token.display_name.as_deref() == Some("exposed"));

        assert!(!hidden.unwrap().exported);
        assert!(exposed.unwrap().exported);
        assert!(hidden.unwrap().references.iter().any(|reference| {
            !reference.is_definition && reference.role == super::StaticReferenceRole::Export
        }));
        let export = hidden
            .unwrap()
            .references
            .iter()
            .find_map(|reference| reference.export.as_ref())
            .unwrap();
        assert_eq!(export.alias, "alias");
        assert!(export.qualified_name.ends_with("::alias"));
        assert!(export.exporter.starts_with("rust-hir-v1|"));
        assert!(export.alias_id.starts_with("rust-export-v1|"));
        assert!(hidden.unwrap().references.iter().any(|reference| {
            reference.role == super::StaticReferenceRole::Import
                && reference
                    .interface_spelling
                    .as_deref()
                    .is_some_and(|spelling| spelling.contains("as local_alias"))
        }));
    }

    #[test]
    fn graph_does_not_export_public_inherent_items_of_private_types() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
struct Private;
impl Private { pub fn hidden() {} }
mod private { pub struct Nested; }
impl private::Nested { pub fn nested_hidden() {} }
pub struct Public;
impl Public { pub fn exposed() {} }
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let tokens = index.tokens.iter().map(|(_, token)| token).collect::<Vec<_>>();

        assert!(
            !tokens
                .iter()
                .find(|token| token.display_name.as_deref() == Some("hidden"))
                .unwrap()
                .exported
        );
        assert!(
            !tokens
                .iter()
                .find(|token| token.display_name.as_deref() == Some("nested_hidden"))
                .unwrap()
                .exported
        );
        assert!(
            tokens
                .iter()
                .find(|token| token.display_name.as_deref() == Some("exposed"))
                .unwrap()
                .exported
        );
    }

    #[test]
    fn graph_marks_const_functions_for_interface_fingerprinting() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
pub const fn answer() -> u8 { 42 }
pub fn runtime() -> u8 { 42 }
"#,
        );
        let tokens = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded)
            .tokens
            .iter()
            .map(|(_, token)| token)
            .collect::<Vec<_>>();

        assert!(
            tokens
                .iter()
                .find(|token| token.display_name.as_deref() == Some("answer"))
                .unwrap()
                .const_function
        );
        assert!(
            !tokens
                .iter()
                .find(|token| token.display_name.as_deref() == Some("runtime"))
                .unwrap()
                .const_function
        );
    }

    #[test]
    fn graph_marks_default_trait_methods_as_trait_members() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
pub trait Render { fn render(&self) {} }
pub fn call(value: &dyn Render) { value.render(); }
"#,
        );
        let tokens = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded)
            .tokens
            .iter()
            .map(|(_, token)| token)
            .collect::<Vec<_>>();
        let render = tokens.iter().find(|token| {
            token.definition.is_some() && token.display_name.as_deref() == Some("render")
        });

        assert!(render.unwrap().trait_member);
    }

    #[test]
    fn graph_identity_separates_inherent_and_trait_methods_with_the_same_name() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
pub trait First { fn execute(&self); }
pub trait Second { fn execute(&self); }
pub struct Service;
impl Service { pub fn execute(&self) {} }
impl First for Service { fn execute(&self) {} }
impl Second for Service { fn execute(&self) {} }
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let ids = index
            .tokens
            .iter()
            .filter(|(_, token)| {
                token.definition.is_some() && token.display_name.as_deref() == Some("execute")
            })
            .map(|(_, token)| token.stable_id.clone())
            .collect::<BTreeSet<_>>();

        assert_eq!(ids.len(), 5, "{ids:#?}");
        assert!(ids.iter().any(|id| id.contains("implemented-trait")));
    }

    #[test]
    fn stock_static_index_avoids_graph_only_enrichment() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
pub trait Render { fn render(&self); }
pub struct Service;
impl Render for Service { fn render(&self) {} }
pub fn run() { Service.render(); }
"#,
        );
        let index = StaticIndex::compute(&analysis, VendoredLibrariesConfig::Excluded);

        assert!(index.relations.is_empty());
        assert!(index.tokens.iter_ref().all(|(_, token)| token.stable_id.is_empty()));
        assert!(
            index
                .tokens
                .iter_ref()
                .flat_map(|(_, token)| &token.references)
                .all(|reference| { reference.role == super::StaticReferenceRole::Reference })
        );
    }

    #[test]
    fn graph_reference_roles_do_not_promote_path_qualifiers() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
pub mod api {
    pub struct Service;
    impl Service { pub fn create() -> Self { Self } }
    pub fn execute() {}
}
pub fn run() {
    api::execute();
    let _: api::Service = api::Service::create();
}
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let roles = |name: &str| {
            index
                .tokens
                .iter_ref()
                .filter(|(_, token)| token.display_name.as_deref() == Some(name))
                .flat_map(|(_, token)| token.references.iter().map(|reference| reference.role))
                .collect::<Vec<_>>()
        };

        let api_roles = roles("api");
        assert!(api_roles.iter().all(|role| *role == super::StaticReferenceRole::Reference));
        assert!(roles("execute").contains(&super::StaticReferenceRole::Call));
        assert!(roles("Service").contains(&super::StaticReferenceRole::Type));
        assert!(roles("create").contains(&super::StaticReferenceRole::Call));
    }

    #[test]
    fn graph_marks_files_shared_by_multiple_crate_contexts() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/one.rs crate:one
#[path = "shared.rs"] mod shared;
//- /workspace/two.rs crate:two
#[path = "shared.rs"] mod shared;
//- /workspace/shared.rs
pub fn shared() {}
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);

        assert!(index.files.iter().any(|file| file.conditional_build));
    }

    #[test]
    fn graph_reference_roles_keep_macro_expansion_context() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- /workspace/lib.rs crate:main
macro_rules! invoke {
    ($value:expr) => { $value };
}
pub trait Render { fn render(&self); }
pub struct Service;
impl Render for Service { fn render(&self) {} }
pub fn run() { invoke!(Service.render()); }
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let mut service_is_constructed = false;
        let mut render_is_called = false;
        for (_, token) in index.tokens.iter() {
            match token.display_name.as_deref() {
                Some("Service") => {
                    service_is_constructed |= token
                        .references
                        .iter()
                        .any(|reference| reference.role == super::StaticReferenceRole::Instantiate);
                }
                Some("render") => {
                    render_is_called |= token
                        .references
                        .iter()
                        .any(|reference| reference.role == super::StaticReferenceRole::Call);
                }
                _ => {}
            }
        }
        assert!(service_is_constructed);
        assert!(render_is_called);
    }

    #[test]
    fn graph_covers_trait_generic_async_and_macro_semantics() {
        let (analysis, _) = fixture::annotations_without_marker(
            r#"
//- minicore: async_fn
//- /workspace/lib.rs crate:main
pub trait Parent { type Item; const VALUE: u8; fn same(&self) -> u8; }
pub trait Child: Parent { fn child(&self) -> u8; }
pub trait Other { fn same(&self) -> u8; }
pub struct Service<T>(pub T);
impl Service<u8> { pub fn inherent(&self) -> u8 { self.0 } }
impl Parent for Service<u8> {
    type Item = u8;
    const VALUE: u8 = 1;
    fn same(&self) -> u8 { self.0 }
}
impl Child for Service<u8> { fn child(&self) -> u8 { Parent::same(self) } }
impl Other for Service<u8> { fn same(&self) -> u8 { self.inherent() } }
macro_rules! invoke { ($value:expr) => { $value }; }
pub fn generic<T: Parent>(value: &T) -> u8 { value.same() }
pub async fn asynchronous(value: Service<u8>) -> u8 {
    let constructed = Service(1);
    let closure = || Other::same(&value);
    invoke!(closure() + Child::child(&value) + constructed.inherent())
}
"#,
        );
        let index = StaticIndex::compute_graph(&analysis, VendoredLibrariesConfig::Excluded);
        let relation_kinds =
            index.relations.iter().map(|relation| relation.kind).collect::<BTreeSet<_>>();
        let tokens = index.tokens.iter().map(|(_, token)| token).collect::<Vec<_>>();
        let same_ids = tokens
            .iter()
            .filter(|token| token.display_name.as_deref() == Some("same"))
            .map(|token| token.stable_id.as_str())
            .collect::<BTreeSet<_>>();
        let names = tokens
            .iter()
            .filter_map(|token| token.display_name.as_deref())
            .collect::<BTreeSet<_>>();
        let roles = tokens
            .iter()
            .flat_map(|token| token.references.iter().map(|reference| reference.role))
            .collect::<Vec<_>>();

        assert!(relation_kinds.contains(&StaticRelationKind::Extends));
        assert!(relation_kinds.contains(&StaticRelationKind::Implements));
        assert!(relation_kinds.contains(&StaticRelationKind::Overrides));
        assert!(same_ids.len() >= 4, "same-named trait and impl methods were conflated");
        assert!(names.is_superset(&BTreeSet::from([
            "Item",
            "VALUE",
            "asynchronous",
            "child",
            "closure",
            "generic",
            "inherent",
        ])));
        assert!(roles.contains(&StaticReferenceRole::Access));
        assert!(roles.contains(&StaticReferenceRole::Call));
        assert!(roles.contains(&StaticReferenceRole::Instantiate));
        assert!(roles.contains(&StaticReferenceRole::Type));
    }
}
