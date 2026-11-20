use std::collections::{HashMap, HashSet};
use std::path::Path;

use yaml_rust2::yaml::Hash;
use yaml_rust2::Yaml;

use crate::error::ExtractorError;
use crate::indexer::{AssetIndex, UnityRef, UnityYamlFile};
use crate::yaml_parser::helpers::{
    hash_get, set_yaml_by_tokens, tokenize_property_path, yaml_as_bool, yaml_as_i64, yaml_as_quat,
    yaml_as_string, yaml_as_vec3, yaml_as_virtual_id, yaml_get,
};

const MAX_PREFAB_DEPTH: usize = 32;

/// Unique ID of an instantiated object in the resolved virtual scene graph.
pub type VirtualId = usize;

/// Instantiated Unity object (after recursive PrefabInstance and Prefab Variant resolution).
#[derive(Debug, Clone)]
pub struct VirtualObject {
    pub id: VirtualId,
    pub class_id: i64,
    pub root_key: String,
    pub data: Yaml,
    /// Chain of `(asset_guid, file_id)` pairs this object originated from (root prefab -> variant -> outer).
    pub origins: Vec<(Option<String>, i64)>,
    pub removed: bool,
}

/// Resolved virtual scene holding all instantiated objects from a `.unity` scene or `.prefab`.
#[derive(Debug, Clone, Default)]
pub struct VirtualScene {
    pub objects: Vec<VirtualObject>,
}

impl VirtualScene {
    /// Loads and recursively resolves a `.unity` scene or `.prefab` file.
    pub fn load_and_resolve(
        file_path: &Path,
        index: &AssetIndex,
    ) -> Result<(Self, Vec<VirtualId>), ExtractorError> {
        let mut scene = Self::default();
        let own_guid = file_path
            .strip_prefix(index.project_dir())
            .ok()
            .and_then(|rel| index.guid_for_relative_path(rel))
            .map(str::to_owned);
        let mut stack = HashSet::new();
        let root_ids = scene.resolve_file_recursive(file_path, own_guid, index, 0, &mut stack)?;
        Ok((scene, root_ids))
    }

    pub fn get(&self, id: VirtualId) -> Option<&VirtualObject> {
        self.objects.get(id).filter(|o| !o.removed)
    }

    pub fn get_mut(&mut self, id: VirtualId) -> Option<&mut VirtualObject> {
        self.objects.get_mut(id).filter(|o| !o.removed)
    }

    fn alloc_object(
        &mut self,
        class_id: i64,
        root_key: String,
        data: Yaml,
        origins: Vec<(Option<String>, i64)>,
    ) -> VirtualId {
        let id = self.objects.len();
        self.objects.push(VirtualObject {
            id,
            class_id,
            root_key,
            data,
            origins,
            removed: false,
        });
        id
    }

    fn resolve_file_recursive(
        &mut self,
        file_path: &Path,
        file_guid: Option<String>,
        index: &AssetIndex,
        depth: usize,
        active_stack: &mut HashSet<String>,
    ) -> Result<Vec<VirtualId>, ExtractorError> {
        if depth > MAX_PREFAB_DEPTH {
            return Err(ExtractorError::PrefabResolution {
                message: format!(
                    "prefab recursion depth exceeded {} at {}",
                    MAX_PREFAB_DEPTH,
                    file_path.display()
                ),
            });
        }

        let canonical_key = file_path.to_string_lossy().to_ascii_lowercase();
        if !active_stack.insert(canonical_key.clone()) {
            return Err(ExtractorError::PrefabResolution {
                message: format!(
                    "cyclic prefab reference detected at {}",
                    file_path.display()
                ),
            });
        }

        let yaml_file = UnityYamlFile::load(file_path)?;
        let mut local_map: HashMap<i64, VirtualId> = HashMap::new();
        let mut own_non_stripped_ids: Vec<VirtualId> = Vec::new();
        let mut prefab_instances: Vec<(i64, Yaml)> = Vec::new();
        let mut stripped_docs: Vec<(i64, crate::indexer::StrippedRef)> = Vec::new();

        for doc in &yaml_file.documents {
            if doc.stripped {
                if let Some(sref) = &doc.stripped_ref {
                    stripped_docs.push((doc.file_id, sref.clone()));
                }
                continue;
            }
            if doc.class_id == 1001 || doc.root_key == "PrefabInstance" {
                prefab_instances.push((doc.file_id, doc.data.clone()));
                continue;
            }

            let vid = self.alloc_object(
                doc.class_id,
                doc.root_key.clone(),
                doc.data.clone(),
                vec![(file_guid.clone(), doc.file_id)],
            );
            local_map.insert(doc.file_id, vid);
            own_non_stripped_ids.push(vid);
        }

        // Instantiate each PrefabInstance in this file
        let mut pi_cloned_vids: HashMap<i64, Vec<VirtualId>> = HashMap::new();
        let mut pi_root_transforms: HashMap<i64, Option<VirtualId>> = HashMap::new();

        for (pi_file_id, pi_data) in &prefab_instances {
            let Some(source_ref) = UnityRef::from_yaml(&pi_data["m_SourcePrefab"]) else {
                continue;
            };
            let Some(source_guid) = source_ref.guid.clone() else {
                continue;
            };
            let Some(source_path) = index.resolve_guid(&source_guid).cloned() else {
                return Err(ExtractorError::AssetNotFound {
                    guid: source_guid,
                    context: format!(
                        "m_SourcePrefab in {} (PrefabInstance &{})",
                        file_path.display(),
                        pi_file_id
                    ),
                });
            };

            let src_vids = self.resolve_file_recursive(
                &source_path,
                Some(source_guid.clone()),
                index,
                depth + 1,
                active_stack,
            )?;

            // Clone resolved prefab objects into this instance scope so multiple instances stay independent
            let mut remap: HashMap<VirtualId, VirtualId> = HashMap::with_capacity(src_vids.len());
            let mut cloned_ids = Vec::with_capacity(src_vids.len());
            for &src_vid in &src_vids {
                let src_obj = self.objects[src_vid].clone();
                let mut origins = src_obj.origins;
                // Mark source_guid origin if not already tagged
                if let Some((_, first_fid)) = origins.first() {
                    let first_fid = *first_fid;
                    if !origins
                        .iter()
                        .any(|(g, f)| g.as_deref() == Some(source_guid.as_str()) && *f == first_fid)
                    {
                        origins.push((Some(source_guid.clone()), first_fid));
                    }
                }
                let new_vid =
                    self.alloc_object(src_obj.class_id, src_obj.root_key, src_obj.data, origins);
                self.objects[new_vid].removed = src_obj.removed;
                remap.insert(src_vid, new_vid);
                cloned_ids.push(new_vid);
            }

            // Remap internal __virtual_id references inside the cloned objects
            for &cid in &cloned_ids {
                rewrite_virtual_ids(&mut self.objects[cid].data, &remap);
            }

            // Remove original template objects so the scene doesn't hold uninstantiated duplicates
            for &src_vid in &src_vids {
                self.objects[src_vid].removed = true;
            }

            // Find root transform of the cloned prefab (Transform/RectTransform with null m_Father)
            let root_tf = cloned_ids.iter().copied().find(|&cid| {
                let obj = &self.objects[cid];
                !obj.removed
                    && is_transform_class(obj.class_id, &obj.root_key)
                    && is_null_virtual_or_unity_ref(&obj.data["m_Father"])
            });

            pi_cloned_vids.insert(*pi_file_id, cloned_ids);
            pi_root_transforms.insert(*pi_file_id, root_tf);
        }

        // Map stripped placeholder documents in this file to their cloned VirtualIds
        for (stripped_fid, sref) in &stripped_docs {
            if let Some(cloned) = pi_cloned_vids.get(&sref.prefab_instance_file_id) {
                if let Some(found) = find_cloned_by_origin(
                    &self.objects,
                    cloned,
                    sref.corresponding_source.guid.as_deref(),
                    sref.corresponding_source.file_id,
                ) {
                    local_map.insert(*stripped_fid, found);
                    if let Some(own_guid) = &file_guid {
                        self.objects[found]
                            .origins
                            .push((Some(own_guid.clone()), *stripped_fid));
                    } else {
                        self.objects[found].origins.push((None, *stripped_fid));
                    }
                }
            }
        }

        // Rewrite local fileIDs in this file's non-stripped objects to __virtual_id
        for &vid in &own_non_stripped_ids {
            rewrite_local_refs(&mut self.objects[vid].data, &local_map);
        }

        // Apply PrefabInstance modifications, removed/added components, and transform parenting
        for (pi_file_id, pi_data) in &prefab_instances {
            let Some(cloned) = pi_cloned_vids.get(pi_file_id).cloned() else {
                continue;
            };
            let modification = &pi_data["m_Modification"];

            // 1. m_Modifications
            if let Some(mods) = modification["m_Modifications"].as_vec() {
                for m in mods {
                    let Some(target_ref) = UnityRef::from_yaml(&m["target"]) else {
                        continue;
                    };
                    let Some(target_vid) = find_cloned_by_origin(
                        &self.objects,
                        &cloned,
                        target_ref.guid.as_deref(),
                        target_ref.file_id,
                    ) else {
                        continue;
                    };

                    let prop_path = m["propertyPath"].as_str().unwrap_or("");
                    if prop_path.is_empty() {
                        continue;
                    }

                    let obj_ref_yaml = &m["objectReference"];
                    let has_obj_ref = UnityRef::from_yaml(obj_ref_yaml)
                        .map(|r| !r.is_null())
                        .unwrap_or(false)
                        || yaml_as_virtual_id(obj_ref_yaml).is_some();

                    let new_val = if has_obj_ref {
                        let mut cloned_ref = obj_ref_yaml.clone();
                        rewrite_local_refs(&mut cloned_ref, &local_map);
                        cloned_ref
                    } else {
                        m["value"].clone()
                    };

                    let tokens = tokenize_property_path(prop_path);
                    set_yaml_by_tokens(&mut self.objects[target_vid].data, &tokens, new_val);
                }
            }

            // 2. m_RemovedComponents
            if let Some(removed_list) = modification["m_RemovedComponents"].as_vec() {
                for r in removed_list {
                    let Some(target_ref) = UnityRef::from_yaml(r) else {
                        continue;
                    };
                    if let Some(comp_vid) = find_cloned_by_origin(
                        &self.objects,
                        &cloned,
                        target_ref.guid.as_deref(),
                        target_ref.file_id,
                    ) {
                        self.objects[comp_vid].removed = true;
                        let go_vid =
                            yaml_as_virtual_id(&self.objects[comp_vid].data["m_GameObject"]);
                        if let Some(go_vid) = go_vid {
                            remove_component_from_game_object(
                                &mut self.objects[go_vid].data,
                                comp_vid,
                            );
                        }
                    }
                }
            }

            // 3. m_AddedComponents
            if let Some(added_list) = modification["m_AddedComponents"].as_vec() {
                for entry in added_list {
                    let added_vid = UnityRef::from_yaml(&entry["addedObject"])
                        .and_then(|r| local_map.get(&r.file_id).copied())
                        .or_else(|| yaml_as_virtual_id(&entry["addedObject"]));
                    let target_ref = UnityRef::from_yaml(&entry["targetCorrespondingSourceObject"]);
                    let Some(added_vid) = added_vid else {
                        continue;
                    };

                    let target_vid = target_ref.and_then(|tr| {
                        find_cloned_by_origin(
                            &self.objects,
                            &cloned,
                            tr.guid.as_deref(),
                            tr.file_id,
                        )
                    });

                    let go_vid = target_vid.and_then(|tv| {
                        if self.objects[tv].class_id == 1
                            || self.objects[tv].root_key == "GameObject"
                        {
                            Some(tv)
                        } else {
                            yaml_as_virtual_id(&self.objects[tv].data["m_GameObject"])
                        }
                    });

                    if let Some(go_vid) = go_vid {
                        append_component_to_game_object(&mut self.objects[go_vid].data, added_vid);
                        set_virtual_ref(&mut self.objects[added_vid].data, "m_GameObject", go_vid);
                    }
                }
            }

            // 4. m_TransformParent
            if let Some(Some(root_tf_vid)) = pi_root_transforms.get(pi_file_id) {
                let parent_vid = UnityRef::from_yaml(&modification["m_TransformParent"])
                    .and_then(|r| {
                        if r.file_id == 0 {
                            None
                        } else {
                            local_map.get(&r.file_id).copied()
                        }
                    })
                    .or_else(|| yaml_as_virtual_id(&modification["m_TransformParent"]));

                if let Some(parent_vid) = parent_vid {
                    set_virtual_ref(&mut self.objects[*root_tf_vid].data, "m_Father", parent_vid);
                    append_child_to_transform(&mut self.objects[parent_vid].data, *root_tf_vid);
                }
            }
        }

        active_stack.remove(&canonical_key);

        let mut all_ids = own_non_stripped_ids;
        for vids in pi_cloned_vids.into_values() {
            all_ids.extend(vids);
        }
        Ok(all_ids)
    }
}

fn find_cloned_by_origin(
    objects: &[VirtualObject],
    cloned_ids: &[VirtualId],
    guid: Option<&str>,
    file_id: i64,
) -> Option<VirtualId> {
    if file_id == 0 {
        return None;
    }
    if let Some(g) = guid {
        if let Some(found) = cloned_ids.iter().copied().find(|&vid| {
            objects[vid]
                .origins
                .iter()
                .any(|(og, of)| og.as_deref() == Some(g) && *of == file_id)
        }) {
            return Some(found);
        }
    }
    cloned_ids
        .iter()
        .copied()
        .find(|&vid| objects[vid].origins.iter().any(|(_, of)| *of == file_id))
}

fn is_transform_class(class_id: i64, root_key: &str) -> bool {
    class_id == 4 || class_id == 224 || root_key == "Transform" || root_key == "RectTransform"
}

fn is_null_virtual_or_unity_ref(yaml: &Yaml) -> bool {
    if yaml_as_virtual_id(yaml).is_some() {
        return false;
    }
    match UnityRef::from_yaml(yaml) {
        Some(r) => r.file_id == 0 && r.guid.is_none(),
        None => true,
    }
}

fn make_virtual_ref_yaml(vid: VirtualId) -> Yaml {
    let mut h = Hash::new();
    h.insert(
        Yaml::String("__virtual_id".to_owned()),
        Yaml::Integer(vid as i64),
    );
    Yaml::Hash(h)
}

fn set_virtual_ref(obj_data: &mut Yaml, field: &str, vid: VirtualId) {
    if let Yaml::Hash(h) = obj_data {
        h.insert(Yaml::String(field.to_owned()), make_virtual_ref_yaml(vid));
    }
}

fn append_component_to_game_object(go_data: &mut Yaml, comp_vid: VirtualId) {
    let mut wrapper = Hash::new();
    wrapper.insert(
        Yaml::String("component".to_owned()),
        make_virtual_ref_yaml(comp_vid),
    );
    if let Yaml::Hash(h) = go_data {
        let key = Yaml::String("m_Component".to_owned());
        let entry = h.entry(key).or_insert_with(|| Yaml::Array(Vec::new()));
        if let Yaml::Array(arr) = entry {
            arr.push(Yaml::Hash(wrapper));
        }
    }
}

fn remove_component_from_game_object(go_data: &mut Yaml, comp_vid: VirtualId) {
    if let Yaml::Hash(h) = go_data {
        if let Some(Yaml::Array(arr)) = h.get_mut(&Yaml::String("m_Component".to_owned())) {
            arr.retain(|item| {
                yaml_as_virtual_id(&item["component"]) != Some(comp_vid)
                    && yaml_as_virtual_id(item) != Some(comp_vid)
            });
        }
    }
}

fn append_child_to_transform(parent_tf_data: &mut Yaml, child_tf_vid: VirtualId) {
    if let Yaml::Hash(h) = parent_tf_data {
        let key = Yaml::String("m_Children".to_owned());
        let entry = h.entry(key).or_insert_with(|| Yaml::Array(Vec::new()));
        if let Yaml::Array(arr) = entry {
            let already = arr
                .iter()
                .any(|item| yaml_as_virtual_id(item) == Some(child_tf_vid));
            if !already {
                arr.push(make_virtual_ref_yaml(child_tf_vid));
            }
        }
    }
}

fn rewrite_local_refs(yaml: &mut Yaml, local_map: &HashMap<i64, VirtualId>) {
    match yaml {
        Yaml::Hash(map) => {
            // Check if this hash is a pure local UnityRef: `{fileID: N}` without `guid` and without `__virtual_id`
            if !map.contains_key(&Yaml::String("__virtual_id".to_owned()))
                && !map.contains_key(&Yaml::String("guid".to_owned()))
            {
                if let Some(fid) = hash_get(map, "fileID").and_then(yaml_as_i64) {
                    if fid != 0 {
                        if let Some(&vid) = local_map.get(&fid) {
                            *yaml = make_virtual_ref_yaml(vid);
                            return;
                        }
                    }
                }
            }
            for (_, v) in map.iter_mut() {
                rewrite_local_refs(v, local_map);
            }
        }
        Yaml::Array(arr) => {
            for item in arr.iter_mut() {
                rewrite_local_refs(item, local_map);
            }
        }
        _ => {}
    }
}

fn rewrite_virtual_ids(yaml: &mut Yaml, remap: &HashMap<VirtualId, VirtualId>) {
    match yaml {
        Yaml::Hash(map) => {
            if let Some(old_vid) = hash_get(map, "__virtual_id")
                .and_then(yaml_as_i64)
                .and_then(|v| usize::try_from(v).ok())
            {
                if let Some(&new_vid) = remap.get(&old_vid) {
                    map.insert(
                        Yaml::String("__virtual_id".to_owned()),
                        Yaml::Integer(new_vid as i64),
                    );
                }
                return;
            }
            for (_, v) in map.iter_mut() {
                rewrite_virtual_ids(v, remap);
            }
        }
        Yaml::Array(arr) => {
            for item in arr.iter_mut() {
                rewrite_virtual_ids(item, remap);
            }
        }
        _ => {}
    }
}

/// High-level component attached to a `SceneNode` in `SceneHierarchy`.
#[derive(Debug, Clone)]
pub struct VirtualComponent {
    pub object_id: VirtualId,
    pub class_id: i64,
    pub root_key: String,
    pub script_guid: Option<String>,
    pub script_file_id: Option<i64>,
    pub data: Yaml,
}

/// Resolved GameObject + Transform node in the scene hierarchy.
#[derive(Debug, Clone)]
pub struct SceneNode {
    pub id: VirtualId,
    pub transform_id: VirtualId,
    pub name: String,
    pub active: bool,
    pub parent: Option<VirtualId>,
    pub children: Vec<VirtualId>,
    pub local_position: [f32; 3],
    pub local_rotation: [f32; 4],
    pub local_scale: [f32; 3],
    pub components: Vec<VirtualComponent>,
}

/// Flat-indexed GameObject hierarchy built from a resolved `VirtualScene`.
#[derive(Debug, Clone, Default)]
pub struct SceneHierarchy {
    pub nodes: HashMap<VirtualId, SceneNode>,
    pub transform_to_go: HashMap<VirtualId, VirtualId>,
    pub roots: Vec<VirtualId>,
}

impl SceneHierarchy {
    /// Builds a `SceneHierarchy` from a `VirtualScene`.
    pub fn from_virtual_scene(vscene: &VirtualScene) -> Self {
        let mut nodes: HashMap<VirtualId, SceneNode> = HashMap::new();
        let mut transform_to_go: HashMap<VirtualId, VirtualId> = HashMap::new();

        // Pass 1: Collect all non-removed GameObjects and their Transform components
        for obj in &vscene.objects {
            if obj.removed {
                continue;
            }
            if obj.class_id != 1 && obj.root_key != "GameObject" {
                continue;
            }
            let name = yaml_get(&obj.data, "m_Name")
                .and_then(yaml_as_string)
                .unwrap_or_default();
            let active = yaml_get(&obj.data, "m_IsActive")
                .and_then(yaml_as_bool)
                .unwrap_or(true);

            let mut components = Vec::new();
            let mut transform_id = None;
            let mut local_position = [0.0, 0.0, 0.0];
            let mut local_rotation = [0.0, 0.0, 0.0, 1.0];
            let mut local_scale = [1.0, 1.0, 1.0];

            if let Some(comp_arr) = yaml_get(&obj.data, "m_Component").and_then(|v| v.as_vec()) {
                for comp_entry in comp_arr {
                    let comp_vid = yaml_as_virtual_id(&comp_entry["component"])
                        .or_else(|| yaml_as_virtual_id(comp_entry));
                    let Some(comp_vid) = comp_vid else {
                        continue;
                    };
                    let Some(comp_obj) = vscene.get(comp_vid) else {
                        continue;
                    };

                    if is_transform_class(comp_obj.class_id, &comp_obj.root_key) {
                        transform_id = Some(comp_vid);
                        if let Some(p) =
                            yaml_get(&comp_obj.data, "m_LocalPosition").and_then(yaml_as_vec3)
                        {
                            local_position = p;
                        }
                        if let Some(r) =
                            yaml_get(&comp_obj.data, "m_LocalRotation").and_then(yaml_as_quat)
                        {
                            local_rotation = r;
                        }
                        if let Some(s) =
                            yaml_get(&comp_obj.data, "m_LocalScale").and_then(yaml_as_vec3)
                        {
                            local_scale = s;
                        }
                    } else {
                        let script_ref =
                            yaml_get(&comp_obj.data, "m_Script").and_then(UnityRef::from_yaml);
                        components.push(VirtualComponent {
                            object_id: comp_vid,
                            class_id: comp_obj.class_id,
                            root_key: comp_obj.root_key.clone(),
                            script_guid: script_ref.as_ref().and_then(|r| r.guid.clone()),
                            script_file_id: script_ref.map(|r| r.file_id),
                            data: comp_obj.data.clone(),
                        });
                    }
                }
            }

            let tf_id = transform_id.unwrap_or(obj.id);
            transform_to_go.insert(tf_id, obj.id);

            nodes.insert(
                obj.id,
                SceneNode {
                    id: obj.id,
                    transform_id: tf_id,
                    name,
                    active,
                    parent: None,
                    children: Vec::new(),
                    local_position,
                    local_rotation,
                    local_scale,
                    components,
                },
            );
        }

        // Pass 2: Wire parent/child links from Transform m_Father and m_Children
        let go_ids: Vec<VirtualId> = nodes.keys().copied().collect();
        for go_id in &go_ids {
            let tf_id = nodes[go_id].transform_id;
            if let Some(tf_obj) = vscene.get(tf_id) {
                if let Some(father_tf_vid) = yaml_as_virtual_id(&tf_obj.data["m_Father"]) {
                    if let Some(&parent_go_vid) = transform_to_go.get(&father_tf_vid) {
                        if let Some(node) = nodes.get_mut(go_id) {
                            node.parent = Some(parent_go_vid);
                        }
                    }
                }
            }
        }

        // Preserve m_Children order where available, then add any parent-linked nodes missing from m_Children
        for go_id in &go_ids {
            let tf_id = nodes[go_id].transform_id;
            if let Some(tf_obj) = vscene.get(tf_id) {
                if let Some(children_arr) =
                    yaml_get(&tf_obj.data, "m_Children").and_then(|v| v.as_vec())
                {
                    let mut child_go_ids = Vec::new();
                    for child_ref in children_arr {
                        if let Some(child_tf_vid) = yaml_as_virtual_id(child_ref) {
                            if let Some(&child_go_vid) = transform_to_go.get(&child_tf_vid) {
                                if nodes.contains_key(&child_go_vid)
                                    && !child_go_ids.contains(&child_go_vid)
                                {
                                    child_go_ids.push(child_go_vid);
                                }
                            }
                        }
                    }
                    for &cid in &child_go_ids {
                        if let Some(child) = nodes.get_mut(&cid) {
                            child.parent = Some(*go_id);
                        }
                    }
                    if let Some(parent) = nodes.get_mut(go_id) {
                        parent.children = child_go_ids;
                    }
                }
            }
        }

        // Ensure all parent links are reflected in parent.children
        for go_id in &go_ids {
            if let Some(parent_id) = nodes[go_id].parent {
                if let Some(parent) = nodes.get_mut(&parent_id) {
                    if !parent.children.contains(go_id) {
                        parent.children.push(*go_id);
                    }
                }
            }
        }

        let mut roots: Vec<VirtualId> = nodes
            .values()
            .filter(|n| n.parent.is_none())
            .map(|n| n.id)
            .collect();
        roots.sort_unstable();

        Self {
            nodes,
            transform_to_go,
            roots,
        }
    }

    /// Finds a GameObject by exact name (preferring root or first match).
    pub fn find_by_name(&self, name: &str) -> Option<VirtualId> {
        // First check roots
        for &r in &self.roots {
            if self.nodes.get(&r).is_some_and(|n| n.name == name) {
                return Some(r);
            }
        }
        // Fallback BFS over all nodes
        let mut keys: Vec<VirtualId> = self.nodes.keys().copied().collect();
        keys.sort_unstable();
        keys.into_iter()
            .find(|&id| self.nodes.get(&id).is_some_and(|n| n.name == name))
    }

    /// Computes the relative path from `ancestor` (exclusive) to `node` (inclusive), e.g., `"Armature/Hips"`.
    pub fn relative_path(&self, ancestor: VirtualId, node: VirtualId) -> Option<String> {
        if ancestor == node {
            return Some(String::new());
        }
        let mut parts = Vec::new();
        let mut cur = Some(node);
        while let Some(cid) = cur {
            if cid == ancestor {
                parts.reverse();
                return Some(parts.join("/"));
            }
            let n = self.nodes.get(&cid)?;
            parts.push(n.name.clone());
            cur = n.parent;
        }
        None
    }

    /// Resolves a Transform or GameObject `__virtual_id` reference to its GameObject `VirtualId`.
    pub fn resolve_ref_to_go(&self, yaml: &Yaml) -> Option<VirtualId> {
        let vid = yaml_as_virtual_id(yaml)?;
        if self.nodes.contains_key(&vid) {
            Some(vid)
        } else {
            self.transform_to_go.get(&vid).copied()
        }
    }

    /// Resolves a component `__virtual_id` reference to `(go_id, &VirtualComponent)`.
    pub fn resolve_component_ref(&self, yaml: &Yaml) -> Option<(VirtualId, &VirtualComponent)> {
        let comp_vid = yaml_as_virtual_id(yaml)?;
        for (&go_id, node) in &self.nodes {
            if let Some(comp) = node.components.iter().find(|c| c.object_id == comp_vid) {
                return Some((go_id, comp));
            }
        }
        None
    }

    /// Returns all GameObject IDs in the subtree rooted at `root_id` (inclusive, pre-order).
    pub fn collect_subtree(&self, root_id: VirtualId) -> Vec<VirtualId> {
        let mut out = Vec::new();
        self.collect_subtree_rec(root_id, &mut out);
        out
    }

    fn collect_subtree_rec(&self, id: VirtualId, out: &mut Vec<VirtualId>) {
        out.push(id);
        if let Some(node) = self.nodes.get(&id) {
            let children = node.children.clone();
            for cid in children {
                self.collect_subtree_rec(cid, out);
            }
        }
    }

    /// Reparents `child_go` under `new_parent_go`.
    pub fn reparent(&mut self, child_go: VirtualId, new_parent_go: VirtualId) {
        if child_go == new_parent_go {
            return;
        }
        if let Some(old_parent) = self.nodes.get(&child_go).and_then(|n| n.parent) {
            if let Some(op) = self.nodes.get_mut(&old_parent) {
                op.children.retain(|&c| c != child_go);
            }
        } else {
            self.roots.retain(|&r| r != child_go);
        }

        if let Some(child) = self.nodes.get_mut(&child_go) {
            child.parent = Some(new_parent_go);
        }
        if let Some(parent) = self.nodes.get_mut(&new_parent_go) {
            if !parent.children.contains(&child_go) {
                parent.children.push(child_go);
            }
        }
    }
}
