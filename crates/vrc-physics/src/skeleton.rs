//! Hierarchical skeleton representation and forward kinematics pose evaluation.

use glam::{Mat4, Quat, Vec3};

/// Single node in an avatar's transform hierarchy (loaded from glTF / manifest).
#[derive(Debug, Clone, PartialEq)]
pub struct SkeletonNode {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub rest_translation: Vec3,
    pub rest_rotation: Quat,
    pub rest_scale: Vec3,
}

impl Default for SkeletonNode {
    fn default() -> Self {
        Self {
            name: String::new(),
            parent: None,
            children: Vec::new(),
            rest_translation: Vec3::ZERO,
            rest_rotation: Quat::IDENTITY,
            rest_scale: Vec3::ONE,
        }
    }
}

impl SkeletonNode {
    /// Creates a `SkeletonNode` with explicit rest transform components.
    pub fn new(
        name: impl Into<String>,
        parent: Option<usize>,
        children: Vec<usize>,
        rest_translation: Vec3,
        rest_rotation: Quat,
        rest_scale: Vec3,
    ) -> Self {
        Self {
            name: name.into(),
            parent,
            children,
            rest_translation,
            rest_rotation: rest_rotation.normalize(),
            rest_scale,
        }
    }
}

/// Evaluates local-to-world forward kinematics for a skeleton hierarchy.
#[derive(Debug, Clone)]
pub struct SkeletonPose {
    nodes: Vec<SkeletonNode>,
    full_paths: Vec<String>,
    topo_order: Vec<usize>,
    local_transforms: Vec<(Vec3, Quat, Vec3)>,
    world_matrices: Vec<Mat4>,
    world_positions: Vec<Vec3>,
    world_rotations: Vec<Quat>,
}

impl SkeletonPose {
    /// Constructs a `SkeletonPose` initialized at the rest pose of `nodes`.
    pub fn from_nodes(mut nodes: Vec<SkeletonNode>) -> Self {
        // Ensure parent <-> children links are consistent even if callers only populate one side.
        let count = nodes.len();
        for idx in 0..count {
            let children = nodes[idx].children.clone();
            for child_idx in children {
                if child_idx < count && nodes[child_idx].parent.is_none() {
                    nodes[child_idx].parent = Some(idx);
                }
            }
        }
        for idx in 0..count {
            if let Some(parent_idx) = nodes[idx].parent {
                if parent_idx < count && !nodes[parent_idx].children.contains(&idx) {
                    nodes[parent_idx].children.push(idx);
                }
            }
        }

        let topo_order = compute_topo_order(&nodes);
        let full_paths = compute_full_paths(&nodes);
        let local_transforms: Vec<(Vec3, Quat, Vec3)> = nodes
            .iter()
            .map(|n| {
                (
                    n.rest_translation,
                    n.rest_rotation.normalize(),
                    n.rest_scale,
                )
            })
            .collect();

        let mut pose = Self {
            nodes,
            full_paths,
            topo_order,
            local_transforms,
            world_matrices: vec![Mat4::IDENTITY; count],
            world_positions: vec![Vec3::ZERO; count],
            world_rotations: vec![Quat::IDENTITY; count],
        };
        pose.update_world_transforms();
        pose
    }

    /// Number of nodes in the skeleton.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns `true` if the skeleton contains no nodes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Returns the immutable node definitions.
    pub fn nodes(&self) -> &[SkeletonNode] {
        &self.nodes
    }

    /// Returns the hierarchical slash-delimited paths for each node.
    pub fn full_paths(&self) -> &[String] {
        &self.full_paths
    }

    /// Returns the current local `(translation, rotation, scale)` tuples.
    pub fn local_transforms(&self) -> &[(Vec3, Quat, Vec3)] {
        &self.local_transforms
    }

    /// Returns mutable access to local `(translation, rotation, scale)` tuples.
    pub fn local_transforms_mut(&mut self) -> &mut [(Vec3, Quat, Vec3)] {
        &mut self.local_transforms
    }

    /// Returns the evaluated world-space transformation matrices.
    pub fn world_matrices(&self) -> &[Mat4] {
        &self.world_matrices
    }

    /// Returns the evaluated world-space origin positions of each node.
    pub fn world_positions(&self) -> &[Vec3] {
        &self.world_positions
    }

    /// Returns the evaluated world-space rotations of each node.
    pub fn world_rotations(&self) -> &[Quat] {
        &self.world_rotations
    }

    /// Resets all local transforms to their rest values and recomputes world transforms.
    pub fn reset_to_rest(&mut self) {
        for (local, node) in self.local_transforms.iter_mut().zip(&self.nodes) {
            *local = (
                node.rest_translation,
                node.rest_rotation.normalize(),
                node.rest_scale,
            );
        }
        self.update_world_transforms();
    }

    /// Sets the local rotation of node `idx` (normalized).
    pub fn set_local_rotation(&mut self, idx: usize, rotation: Quat) {
        if let Some((_, rot, _)) = self.local_transforms.get_mut(idx) {
            *rot = rotation.normalize();
        }
    }

    /// Sets the local translation of node `idx`.
    pub fn set_local_translation(&mut self, idx: usize, translation: Vec3) {
        if let Some((trans, _, _)) = self.local_transforms.get_mut(idx) {
            *trans = translation;
        }
    }

    /// Multiplies the current local rotation of node `idx` by `offset` (`current * offset`).
    pub fn apply_local_rotation_offset(&mut self, idx: usize, offset: Quat) {
        if let Some((_, rot, _)) = self.local_transforms.get_mut(idx) {
            *rot = (*rot * offset).normalize();
        }
    }

    /// Recomputes world matrices, positions, and rotations in topological parent-first order.
    pub fn update_world_transforms(&mut self) {
        for i in 0..self.topo_order.len() {
            let idx = self.topo_order[i];
            self.update_single_node_world(idx);
        }
    }

    /// Updates world transform for a single node assuming its parent's world transform is current.
    pub fn update_single_node_world(&mut self, idx: usize) {
        if idx >= self.nodes.len() {
            return;
        }
        let (t, r, s) = self.local_transforms[idx];
        let r_norm = r.normalize();
        let local_mat = Mat4::from_scale_rotation_translation(s, r_norm, t);

        if let Some(parent_idx) = self.nodes[idx].parent {
            if parent_idx < self.nodes.len() && parent_idx != idx {
                let world_mat = self.world_matrices[parent_idx] * local_mat;
                let world_rot = (self.world_rotations[parent_idx] * r_norm).normalize();
                self.world_matrices[idx] = world_mat;
                self.world_positions[idx] = world_mat.w_axis.truncate();
                self.world_rotations[idx] = world_rot;
                return;
            }
        }

        self.world_matrices[idx] = local_mat;
        self.world_positions[idx] = local_mat.w_axis.truncate();
        self.world_rotations[idx] = r_norm;
    }

    /// Finds a node index by exact node name, full hierarchy path, or path suffix.
    pub fn find_node(&self, name_or_path: &str) -> Option<usize> {
        let query = name_or_path.trim();
        if query.is_empty() {
            return None;
        }

        if let Some(idx) = self.nodes.iter().position(|n| n.name == query) {
            return Some(idx);
        }
        if let Some(idx) = self.full_paths.iter().position(|p| p == query) {
            return Some(idx);
        }

        let leaf = query.rsplit('/').next().unwrap_or(query);
        if let Some(idx) = self.nodes.iter().position(|n| n.name == leaf) {
            return Some(idx);
        }

        let suffix = format!("/{query}");
        self.full_paths.iter().position(|p| p.ends_with(&suffix))
    }

    /// Returns the world matrix for node `idx` if valid.
    pub fn world_matrix(&self, idx: usize) -> Option<Mat4> {
        self.world_matrices.get(idx).copied()
    }

    /// Returns the world position for node `idx` if valid.
    pub fn world_position(&self, idx: usize) -> Option<Vec3> {
        self.world_positions.get(idx).copied()
    }

    /// Returns the world rotation for node `idx` if valid.
    pub fn world_rotation(&self, idx: usize) -> Option<Quat> {
        self.world_rotations.get(idx).copied()
    }
}

fn compute_topo_order(nodes: &[SkeletonNode]) -> Vec<usize> {
    let count = nodes.len();
    let mut visited = vec![false; count];
    let mut order = Vec::with_capacity(count);

    // Breadth-first from roots so parents are always visited before children.
    let mut queue = std::collections::VecDeque::new();
    for (idx, node) in nodes.iter().enumerate() {
        let is_root = match node.parent {
            None => true,
            Some(p) => p >= count || p == idx,
        };
        if is_root {
            visited[idx] = true;
            queue.push_back(idx);
        }
    }

    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        for &child in &nodes[idx].children {
            if child < count && !visited[child] {
                visited[child] = true;
                queue.push_back(child);
            }
        }
    }

    // Fallback for any disconnected/cyclic nodes so every index appears once.
    for (idx, &seen) in visited.iter().enumerate() {
        if !seen {
            order.push(idx);
        }
    }

    order
}

fn compute_full_paths(nodes: &[SkeletonNode]) -> Vec<String> {
    let count = nodes.len();
    let mut paths = Vec::with_capacity(count);

    for idx in 0..count {
        let mut segments = Vec::new();
        let mut cursor = Some(idx);
        let mut steps = 0;
        while let Some(curr) = cursor {
            if curr >= count || steps >= count {
                break;
            }
            segments.push(nodes[curr].name.as_str());
            cursor = nodes[curr].parent;
            steps += 1;
        }
        segments.reverse();
        paths.push(segments.join("/"));
    }

    paths
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_kinematics_propagates_parent_rotations() {
        let nodes = vec![
            SkeletonNode::new(
                "Root",
                None,
                vec![1],
                Vec3::new(0.0, 1.0, 0.0),
                Quat::from_rotation_z(std::f32::consts::FRAC_PI_2),
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "Child",
                Some(0),
                vec![],
                Vec3::new(0.0, 2.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
        ];

        let pose = SkeletonPose::from_nodes(nodes);
        let root_pos = pose.world_position(0).expect("root pos");
        let child_pos = pose.world_position(1).expect("child pos");

        assert!((root_pos - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-5);
        // Local (0, 2, 0) rotated +90 deg around Z becomes (-2, 0, 0), plus root (0, 1, 0) -> (-2, 1, 0)
        assert!(
            (child_pos - Vec3::new(-2.0, 1.0, 0.0)).length() < 1e-5,
            "unexpected child_pos: {child_pos:?}"
        );
        assert_eq!(pose.find_node("Root/Child"), Some(1));
        assert_eq!(pose.find_node("Child"), Some(1));
    }

    #[test]
    fn topo_order_handles_parent_after_child() {
        let nodes = vec![
            SkeletonNode::new(
                "Child",
                Some(1),
                vec![],
                Vec3::new(0.0, 1.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
            SkeletonNode::new(
                "Parent",
                None,
                vec![0],
                Vec3::new(1.0, 0.0, 0.0),
                Quat::IDENTITY,
                Vec3::ONE,
            ),
        ];

        let pose = SkeletonPose::from_nodes(nodes);
        let child_pos = pose.world_position(0).expect("child pos");
        assert!((child_pos - Vec3::new(1.0, 1.0, 0.0)).length() < 1e-5);
    }
}
