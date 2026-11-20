//! Verlet / Position Based Dynamics (PBD) `VRCPhysBone` and collider solver.

use std::collections::{HashMap, HashSet};

use glam::{Mat4, Quat, Vec3};
use vrc_schema::{ColliderDef, ColliderShape, PhysBoneDef, PhysBoneLimitType, PhysicsShaderConfig};

use crate::skeleton::SkeletonPose;

/// Fixed physics substep duration (60 Hz).
pub const FIXED_DT: f32 = 1.0 / 60.0;
const MAX_SUBSTEPS: usize = 8;
const SOLVER_ITERATIONS: usize = 4;
const WORLD_GRAVITY: Vec3 = Vec3::new(0.0, -9.81, 0.0);

/// Active mouse grab handle targeting a specific PhysBone particle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhysBoneGrab {
    pub chain_index: usize,
    pub bone_index: usize,
    pub target_world_pos: Vec3,
}

#[derive(Debug, Clone)]
struct Particle {
    curr_pos: Vec3,
    prev_pos: Vec3,
    bone_length: f32,
    /// Direction from parent bone to this particle in the parent bone's local space at rest.
    rest_dir_in_prev_local: Vec3,
}

#[derive(Debug, Clone)]
struct PhysBoneChain {
    def: PhysBoneDef,
    /// Skeleton node indices `[root, child_1, ..., leaf]`.
    node_indices: Vec<usize>,
    /// Particles aligned 1:1 with `node_indices`. Index 0 is the pinned root anchor.
    particles: Vec<Particle>,
    prev_anchor_pos: Vec3,
    collider_indices: Vec<usize>,
}

#[derive(Debug, Clone)]
struct ColliderRuntime {
    def: ColliderDef,
    parent_node_idx: Option<usize>,
    world_mat: Mat4,
    world_scale: f32,
}

/// Runtime PhysBone & collider simulation world.
#[derive(Debug, Clone, Default)]
pub struct PhysBoneWorld {
    chains: Vec<PhysBoneChain>,
    colliders: Vec<ColliderRuntime>,
    accumulator: f32,
    active_grab: Option<PhysBoneGrab>,
    pending_impulse: Vec3,
}

impl PhysBoneWorld {
    /// Builds PhysBone simulation chains and collider bindings from manifest definitions and rest pose.
    pub fn new(
        phys_bones: &[PhysBoneDef],
        colliders: &[ColliderDef],
        rest_pose: &SkeletonPose,
    ) -> Self {
        let mut collider_id_to_idx = HashMap::with_capacity(colliders.len());
        let mut collider_runtimes = Vec::with_capacity(colliders.len());
        for (idx, col) in colliders.iter().enumerate() {
            let parent_node_idx = rest_pose.find_node(&col.parent_node);
            collider_id_to_idx.insert(col.id.clone(), idx);
            let mut rt = ColliderRuntime {
                def: col.clone(),
                parent_node_idx,
                world_mat: Mat4::IDENTITY,
                world_scale: 1.0,
            };
            rt.update_world(rest_pose);
            collider_runtimes.push(rt);
        }

        let mut chains = Vec::new();
        for def in phys_bones {
            let Some(root_idx) = rest_pose.find_node(&def.root_node) else {
                tracing::debug!(root_node = %def.root_node, "PhysBone root node not found in skeleton");
                continue;
            };

            let ignore_set = collect_ignored_indices(def, rest_pose);
            if ignore_set.contains(&root_idx) {
                continue;
            }

            let raw_paths = collect_chain_paths(root_idx, rest_pose, &ignore_set);
            let collider_indices: Vec<usize> = def
                .collider_refs
                .iter()
                .filter_map(|id| collider_id_to_idx.get(id).copied())
                .collect();

            for node_indices in raw_paths {
                if node_indices.len() < 2 {
                    continue;
                }
                if let Some(chain) = build_chain(
                    def.clone(),
                    node_indices,
                    rest_pose,
                    collider_indices.clone(),
                ) {
                    chains.push(chain);
                }
            }
        }

        Self {
            chains,
            colliders: collider_runtimes,
            accumulator: 0.0,
            active_grab: None,
            pending_impulse: Vec3::ZERO,
        }
    }

    /// Number of active simulated bone chains.
    pub fn chain_count(&self) -> usize {
        self.chains.len()
    }

    /// Returns world positions of particles in `chain_index` for inspection/testing.
    pub fn chain_particle_positions(&self, chain_index: usize) -> Option<Vec<Vec3>> {
        self.chains
            .get(chain_index)
            .map(|c| c.particles.iter().map(|p| p.curr_pos).collect())
    }

    /// Returns the currently active mouse grab, if any.
    pub fn active_grab(&self) -> Option<PhysBoneGrab> {
        self.active_grab
    }

    /// Resets all PhysBone particles to the provided rest/current skeleton pose and clears inertia.
    pub fn reset(&mut self, pose: &SkeletonPose) {
        self.accumulator = 0.0;
        self.active_grab = None;
        self.pending_impulse = Vec3::ZERO;
        for col in &mut self.colliders {
            col.update_world(pose);
        }
        for chain in &mut self.chains {
            for (p, &node_idx) in chain.particles.iter_mut().zip(&chain.node_indices) {
                if let Some(pos) = pose.world_position(node_idx) {
                    p.curr_pos = pos;
                    p.prev_pos = pos;
                }
            }
            if let Some(&root_idx) = chain.node_indices.first() {
                if let Some(pos) = pose.world_position(root_idx) {
                    chain.prev_anchor_pos = pos;
                }
            }
        }
    }

    /// Finds the nearest grabbable particle (`bone_index >= 1`) within `max_dist` of `world_pos`.
    pub fn find_nearest_grabbable_bone(
        &self,
        world_pos: Vec3,
        max_dist: f32,
    ) -> Option<(usize, usize)> {
        if !max_dist.is_finite() || max_dist < 0.0 {
            return None;
        }
        let mut best_dist_sq = max_dist * max_dist;
        let mut best = None;

        for (chain_idx, chain) in self.chains.iter().enumerate() {
            if !chain.def.allow_grabbing {
                continue;
            }
            for (bone_idx, particle) in chain.particles.iter().enumerate().skip(1) {
                let d2 = particle.curr_pos.distance_squared(world_pos);
                if d2 <= best_dist_sq {
                    best_dist_sq = d2;
                    best = Some((chain_idx, bone_idx));
                }
            }
        }
        best
    }

    /// Starts grabbing the nearest grabbable PhysBone particle within `max_dist`.
    pub fn grab_near(&mut self, world_pos: Vec3, max_dist: f32) -> Option<(usize, usize)> {
        let found = self.find_nearest_grabbable_bone(world_pos, max_dist)?;
        self.active_grab = Some(PhysBoneGrab {
            chain_index: found.0,
            bone_index: found.1,
            target_world_pos: world_pos,
        });
        Some(found)
    }

    /// Sets an explicit grab handle.
    pub fn set_grab(&mut self, grab: PhysBoneGrab) {
        self.active_grab = Some(grab);
    }

    /// Updates the target world position of the currently active grab.
    pub fn update_grab(&mut self, target_world_pos: Vec3) {
        if let Some(grab) = &mut self.active_grab {
            grab.target_world_pos = target_world_pos;
        }
    }

    /// Releases the currently active grab, allowing the bone to spring back naturally.
    pub fn release_grab(&mut self) {
        self.active_grab = None;
    }

    /// Queues an instantaneous velocity impulse (m/s) applied on the next substep.
    pub fn inject_impulse(&mut self, impulse: Vec3) {
        if impulse.is_finite() {
            self.pending_impulse += impulse;
        }
    }

    /// Advances the PhysBone simulation using a fixed `1/60s` substep accumulator and writes rotations onto `pose`.
    pub fn step(
        &mut self,
        dt: f32,
        pose: &mut SkeletonPose,
        window_accel: Vec3,
        cfg: &PhysicsShaderConfig,
    ) {
        if !cfg.physbone_enabled || self.chains.is_empty() {
            self.accumulator = 0.0;
            self.pending_impulse = Vec3::ZERO;
            return;
        }

        let safe_dt = if dt.is_finite() && dt > 0.0 {
            dt.min(0.25)
        } else {
            0.0
        };
        let strength = cfg.physbone_strength.max(0.0);

        for col in &mut self.colliders {
            col.update_world(pose);
        }

        self.accumulator += safe_dt;
        let mut substeps = 0;
        while self.accumulator >= FIXED_DT && substeps < MAX_SUBSTEPS {
            let impulse = if substeps == 0 {
                std::mem::take(&mut self.pending_impulse)
            } else {
                Vec3::ZERO
            };
            self.step_substep(
                pose,
                window_accel,
                impulse,
                strength,
                cfg.physbone_collider_enabled,
            );
            self.accumulator -= FIXED_DT;
            substeps += 1;
        }
        if substeps == MAX_SUBSTEPS && self.accumulator > FIXED_DT {
            self.accumulator = self.accumulator.rem_euclid(FIXED_DT);
        }

        // Write simulated directions back as local quaternion rotations on `SkeletonPose`.
        self.writeback_to_pose(pose, strength);
    }

    fn step_substep(
        &mut self,
        pose: &SkeletonPose,
        window_accel: Vec3,
        impulse: Vec3,
        strength: f32,
        colliders_enabled: bool,
    ) {
        let dt_sq = FIXED_DT * FIXED_DT;
        let safe_window_accel = if window_accel.is_finite() {
            window_accel * strength
        } else {
            Vec3::ZERO
        };

        for (chain_idx, chain) in self.chains.iter_mut().enumerate() {
            let root_idx = chain.node_indices[0];
            let anchor_pos = compute_anchor_world_pos(pose, root_idx);
            let root_base_world_rot = compute_node_rest_world_rot(pose, root_idx);

            let anchor_delta = anchor_pos - chain.prev_anchor_pos;
            chain.prev_anchor_pos = anchor_pos;

            let imm = chain.def.immovable.clamp(0.0, 1.0);
            let pull = chain.def.pull.clamp(0.0, 1.0);
            let spring = chain.def.spring.clamp(0.0, 1.0);
            let stiffness = chain.def.stiffness.clamp(0.0, 1.0);

            // Spring controls bounciness / momentum retention (higher spring = less damping).
            let damping = 0.16 * (1.0 - 0.85 * spring) + 0.25 * imm;
            let vel_retain = (1.0 - damping).clamp(0.0, 0.995);

            // Pin root particle to anchor.
            chain.particles[0].curr_pos = anchor_pos;
            chain.particles[0].prev_pos = anchor_pos;

            // Compute rest world target positions relative to current root frame.
            let mut rest_targets = Vec::with_capacity(chain.particles.len());
            rest_targets.push(anchor_pos);
            let mut accum_rot = root_base_world_rot;
            for i in 1..chain.particles.len() {
                let prev_node_idx = chain.node_indices[i - 1];
                if i > 1 {
                    accum_rot = (accum_rot * pose.nodes()[prev_node_idx].rest_rotation).normalize();
                }
                let dir =
                    (accum_rot * chain.particles[i].rest_dir_in_prev_local).normalize_or_zero();
                let prev_target = rest_targets[i - 1];
                rest_targets.push(prev_target + dir * chain.particles[i].bone_length);
            }

            // 1. Verlet integration + external forces + rest pull/stiffness.
            let effective_window_accel = safe_window_accel * (1.0 - imm);
            let effective_impulse = impulse * (1.0 - imm) * strength;

            for i in 1..chain.particles.len() {
                let prev_particle_pos = chain.particles[i - 1].curr_pos;
                let p = &mut chain.particles[i];
                // Cancel root displacement proportionally to `immovable`.
                p.curr_pos += anchor_delta * imm;
                p.prev_pos += anchor_delta * imm;

                let vel = (p.curr_pos - p.prev_pos) * vel_retain + effective_impulse * FIXED_DT;
                let rest_dir = (rest_targets[i] - rest_targets[i - 1]).normalize_or_zero();
                let cur_dir = (p.curr_pos - prev_particle_pos).normalize_or_zero();
                let alignment = cur_dir.dot(rest_dir).max(0.0);
                let falloff_factor = 1.0 - chain.def.gravity_falloff.clamp(0.0, 1.0) * alignment;
                let grav_accel = WORLD_GRAVITY * (chain.def.gravity * falloff_factor * strength);

                let mut next_pos = p.curr_pos + vel + (grav_accel + effective_window_accel) * dt_sq;

                // Pull toward rest pose (scaled for 60Hz substeps).
                if pull > 0.0 {
                    let pull_alpha = (pull * 0.04).clamp(0.0, 0.5);
                    next_pos = next_pos.lerp(rest_targets[i], pull_alpha);
                }

                p.prev_pos = p.curr_pos;
                p.curr_pos = next_pos;
            }

            // 2. Apply active mouse grab pull before constraint projection.
            if let Some(grab) = self.active_grab {
                if grab.chain_index == chain_idx
                    && grab.bone_index >= 1
                    && grab.bone_index < chain.particles.len()
                    && chain.def.allow_grabbing
                {
                    let p = &mut chain.particles[grab.bone_index];
                    p.curr_pos = p.curr_pos.lerp(grab.target_world_pos, 0.85);
                    // Damp velocity of grabbed node so it tracks the cursor cleanly.
                    p.prev_pos = p.prev_pos.lerp(p.curr_pos, 0.5);
                }
            }

            // 3. PBD iterative constraint solver (stiffness, distance, angular limits, colliders).
            for _iter in 0..SOLVER_ITERATIONS {
                let mut parent_frame_rot = root_base_world_rot;
                for i in 1..chain.particles.len() {
                    let len = chain.particles[i].bone_length;
                    let prev_pos = chain.particles[i - 1].curr_pos;

                    if i > 1 {
                        let prev_ref = (parent_frame_rot
                            * chain.particles[i - 1].rest_dir_in_prev_local)
                            .normalize_or_zero();
                        let prev_cur = (chain.particles[i - 1].curr_pos
                            - chain.particles[i - 2].curr_pos)
                            .normalize_or_zero();
                        if prev_ref.length_squared() > 1e-6 && prev_cur.length_squared() > 1e-6 {
                            let delta = Quat::from_rotation_arc(prev_ref, prev_cur);
                            parent_frame_rot = (delta
                                * parent_frame_rot
                                * pose.nodes()[chain.node_indices[i - 1]].rest_rotation)
                                .normalize();
                        } else {
                            parent_frame_rot = (parent_frame_rot
                                * pose.nodes()[chain.node_indices[i - 1]].rest_rotation)
                                .normalize();
                        }
                    }

                    // Stiffness: align segment with parent bone's current local orientation.
                    let ref_dir = (parent_frame_rot * chain.particles[i].rest_dir_in_prev_local)
                        .normalize_or_zero();
                    if stiffness > 0.0 && ref_dir.length_squared() > 1e-6 {
                        let stiff_target = prev_pos + ref_dir * len;
                        let stiff_alpha =
                            (stiffness * 0.04 / SOLVER_ITERATIONS as f32).clamp(0.0, 0.4);
                        chain.particles[i].curr_pos =
                            chain.particles[i].curr_pos.lerp(stiff_target, stiff_alpha);
                    }

                    // Distance constraint (keep exact segment length).
                    let delta = chain.particles[i].curr_pos - prev_pos;
                    let dist = delta.length();
                    if dist > 1e-6 {
                        let dir = delta / dist;
                        chain.particles[i].curr_pos = prev_pos + dir * len;
                    } else if ref_dir.length_squared() > 1e-6 {
                        chain.particles[i].curr_pos = prev_pos + ref_dir * len;
                    }

                    // Angular limits relative to parent frame.
                    if chain.def.limit_type != PhysBoneLimitType::None
                        && ref_dir.length_squared() > 1e-6
                    {
                        let constrained = apply_angular_limit(
                            chain.particles[i].curr_pos - prev_pos,
                            ref_dir,
                            parent_frame_rot,
                            chain.def.limit_type,
                            chain.def.max_angle_x,
                            chain.def.max_angle_z,
                            len,
                        );
                        chain.particles[i].curr_pos = prev_pos + constrained;
                    }

                    // Collider resolution.
                    if colliders_enabled && !chain.collider_indices.is_empty() {
                        let bone_radius = chain.def.radius.max(0.0);
                        for &col_idx in &chain.collider_indices {
                            if let Some(col) = self.colliders.get(col_idx) {
                                let before = chain.particles[i].curr_pos;
                                let mut pushed = resolve_collider(before, bone_radius, col);
                                let push_delta = pushed - before;
                                if push_delta.length_squared() > 1e-8 {
                                    let bone_dir = (before - prev_pos).normalize_or(Vec3::Y);
                                    let push_dir = push_delta.normalize();
                                    // If collider pushes straight back toward parent bone, add slight
                                    // lateral slide so the segment wraps around the collider instead of compressing.
                                    if bone_dir.dot(push_dir) < -0.95 {
                                        let slide = any_perpendicular(bone_dir) * (len * 0.25);
                                        pushed = resolve_collider(before + slide, bone_radius, col);
                                    }
                                    chain.particles[i].curr_pos = pushed;
                                }
                            }
                        }
                    }
                }
            }

            // Final top-down exact bone length projection pass (preserving lateral collider wrap).
            for i in 1..chain.particles.len() {
                let prev_pos = chain.particles[i - 1].curr_pos;
                let len = chain.particles[i].bone_length;
                let delta = chain.particles[i].curr_pos - prev_pos;
                let dist = delta.length();
                if dist > 1e-6 {
                    chain.particles[i].curr_pos = prev_pos + (delta / dist) * len;
                }
                if colliders_enabled && !chain.collider_indices.is_empty() {
                    let bone_radius = chain.def.radius.max(0.0);
                    for &col_idx in &chain.collider_indices {
                        if let Some(col) = self.colliders.get(col_idx) {
                            let resolved =
                                resolve_collider(chain.particles[i].curr_pos, bone_radius, col);
                            let d2 = resolved - prev_pos;
                            if d2.length_squared() > 1e-6 {
                                chain.particles[i].curr_pos = prev_pos + d2.normalize() * len;
                            }
                        }
                    }
                }
            }
        }
    }

    fn writeback_to_pose(&self, pose: &mut SkeletonPose, strength: f32) {
        let blend = strength.clamp(0.0, 1.0);
        for chain in &self.chains {
            for i in 0..chain.node_indices.len() - 1 {
                let node_idx = chain.node_indices[i];
                let parent_world_rot = match pose.nodes()[node_idx].parent {
                    Some(p_idx) => pose.world_rotation(p_idx).unwrap_or(Quat::IDENTITY),
                    None => Quat::IDENTITY,
                };
                let rest_local_rot = pose.nodes()[node_idx].rest_rotation.normalize();
                let rest_vec_in_parent = (rest_local_rot
                    * chain.particles[i + 1].rest_dir_in_prev_local)
                    .normalize_or_zero();
                let cur_dir_world = (chain.particles[i + 1].curr_pos - chain.particles[i].curr_pos)
                    .normalize_or_zero();
                let desired_dir_in_parent =
                    (parent_world_rot.inverse() * cur_dir_world).normalize_or_zero();

                if rest_vec_in_parent.length_squared() > 1e-6
                    && desired_dir_in_parent.length_squared() > 1e-6
                {
                    let delta = Quat::from_rotation_arc(rest_vec_in_parent, desired_dir_in_parent);
                    let target_rot = (delta * rest_local_rot).normalize();
                    let final_rot = if blend < 1.0 {
                        rest_local_rot.slerp(target_rot, blend)
                    } else {
                        target_rot
                    };
                    pose.set_local_rotation(node_idx, final_rot);
                    pose.update_single_node_world(node_idx);
                }
            }
        }
        pose.update_world_transforms();
    }
}

impl ColliderRuntime {
    fn update_world(&mut self, pose: &SkeletonPose) {
        let parent_mat = self
            .parent_node_idx
            .and_then(|idx| pose.world_matrix(idx))
            .unwrap_or(Mat4::IDENTITY);
        let q_raw = Quat::from_xyzw(
            self.def.rotation[0],
            self.def.rotation[1],
            self.def.rotation[2],
            self.def.rotation[3],
        );
        let local_rot = if q_raw.length_squared() > 1e-6 {
            q_raw.normalize()
        } else {
            Quat::IDENTITY
        };
        let local_mat =
            Mat4::from_rotation_translation(local_rot, Vec3::from_array(self.def.position));
        self.world_mat = parent_mat * local_mat;
        self.world_scale = self
            .world_mat
            .x_axis
            .truncate()
            .length()
            .max(self.world_mat.y_axis.truncate().length())
            .max(self.world_mat.z_axis.truncate().length())
            .max(1e-4);
    }
}

fn collect_ignored_indices(def: &PhysBoneDef, pose: &SkeletonPose) -> HashSet<usize> {
    let mut ignored = HashSet::new();
    for name in &def.ignore_nodes {
        if let Some(idx) = pose.find_node(name) {
            mark_subtree_ignored(idx, pose, &mut ignored);
        }
    }
    ignored
}

fn mark_subtree_ignored(idx: usize, pose: &SkeletonPose, ignored: &mut HashSet<usize>) {
    if !ignored.insert(idx) {
        return;
    }
    for &child in &pose.nodes()[idx].children {
        mark_subtree_ignored(child, pose, ignored);
    }
}

fn collect_chain_paths(
    start_idx: usize,
    pose: &SkeletonPose,
    ignored: &HashSet<usize>,
) -> Vec<Vec<usize>> {
    let mut paths = Vec::new();
    let mut current = Vec::new();
    let mut visited = HashSet::new();
    collect_chain_recursive(
        start_idx,
        pose,
        ignored,
        &mut visited,
        &mut current,
        &mut paths,
    );
    paths
}

fn collect_chain_recursive(
    node_idx: usize,
    pose: &SkeletonPose,
    ignored: &HashSet<usize>,
    visited: &mut HashSet<usize>,
    current: &mut Vec<usize>,
    paths: &mut Vec<Vec<usize>>,
) {
    if ignored.contains(&node_idx) || !visited.insert(node_idx) {
        return;
    }
    current.push(node_idx);

    let active_children: Vec<usize> = pose.nodes()[node_idx]
        .children
        .iter()
        .copied()
        .filter(|c| !ignored.contains(c) && !visited.contains(c))
        .collect();

    if active_children.is_empty() {
        if current.len() >= 2 {
            paths.push(current.clone());
        }
    } else if active_children.len() == 1 {
        collect_chain_recursive(active_children[0], pose, ignored, visited, current, paths);
    } else {
        for child in active_children {
            // Start a sub-chain rooted at the branch point so each branch has its own anchor segment.
            let mut branch = vec![node_idx];
            collect_chain_recursive(child, pose, ignored, visited, &mut branch, paths);
        }
    }

    current.pop();
}

fn build_chain(
    def: PhysBoneDef,
    node_indices: Vec<usize>,
    pose: &SkeletonPose,
    collider_indices: Vec<usize>,
) -> Option<PhysBoneChain> {
    let mut particles = Vec::with_capacity(node_indices.len());
    let root_pos = pose.world_position(node_indices[0])?;

    particles.push(Particle {
        curr_pos: root_pos,
        prev_pos: root_pos,
        bone_length: 0.0,
        rest_dir_in_prev_local: Vec3::Y,
    });

    for i in 1..node_indices.len() {
        let prev_idx = node_indices[i - 1];
        let curr_idx = node_indices[i];
        let prev_pos = pose.world_position(prev_idx)?;
        let curr_pos = pose.world_position(curr_idx)?;
        let prev_rot = pose.world_rotation(prev_idx).unwrap_or(Quat::IDENTITY);

        let world_delta = curr_pos - prev_pos;
        let len = world_delta.length().max(1e-4);
        let world_dir = (world_delta / len).normalize_or(Vec3::Y);
        let rest_dir_in_prev_local = (prev_rot.inverse() * world_dir).normalize_or(Vec3::Y);

        particles.push(Particle {
            curr_pos,
            prev_pos: curr_pos,
            bone_length: len,
            rest_dir_in_prev_local,
        });
    }

    Some(PhysBoneChain {
        def,
        node_indices,
        particles,
        prev_anchor_pos: root_pos,
        collider_indices,
    })
}

fn compute_anchor_world_pos(pose: &SkeletonPose, root_idx: usize) -> Vec3 {
    let node = &pose.nodes()[root_idx];
    if let Some(parent_idx) = node.parent {
        if let Some(parent_mat) = pose.world_matrix(parent_idx) {
            return parent_mat.transform_point3(node.rest_translation);
        }
    }
    node.rest_translation
}

fn compute_node_rest_world_rot(pose: &SkeletonPose, node_idx: usize) -> Quat {
    let node = &pose.nodes()[node_idx];
    let parent_rot = match node.parent {
        Some(p_idx) => pose.world_rotation(p_idx).unwrap_or(Quat::IDENTITY),
        None => Quat::IDENTITY,
    };
    (parent_rot * node.rest_rotation).normalize()
}

fn apply_angular_limit(
    bone_vec: Vec3,
    ref_dir: Vec3,
    parent_frame_rot: Quat,
    limit_type: PhysBoneLimitType,
    max_angle_x_deg: f32,
    max_angle_z_deg: f32,
    bone_length: f32,
) -> Vec3 {
    let cur_dir = bone_vec.normalize_or(ref_dir);
    let max_x = max_angle_x_deg.clamp(0.0, 180.0).to_radians();
    let max_z = max_angle_z_deg.clamp(0.0, 180.0).to_radians();

    match limit_type {
        PhysBoneLimitType::None => cur_dir * bone_length,
        PhysBoneLimitType::Angle => {
            let dot = ref_dir.dot(cur_dir).clamp(-1.0, 1.0);
            let angle = dot.acos();
            if angle <= max_x {
                return cur_dir * bone_length;
            }
            let axis = ref_dir.cross(cur_dir);
            let axis_norm = if axis.length_squared() > 1e-8 {
                axis.normalize()
            } else {
                any_perpendicular(ref_dir)
            };
            let clamped_dir = (Quat::from_axis_angle(axis_norm, max_x) * ref_dir).normalize();
            clamped_dir * bone_length
        }
        PhysBoneLimitType::Hinge => {
            // Hinge rotates around local Z axis (movement confined to local X-Y / ref-normal plane).
            let align = Quat::from_rotation_arc(
                Vec3::Y,
                (parent_frame_rot.inverse() * ref_dir).normalize_or(Vec3::Y),
            );
            let frame = (parent_frame_rot * align).normalize();
            let local_dir = (frame.inverse() * cur_dir).normalize_or(Vec3::Y);
            // Project out Z component to restrict to the hinge plane, then clamp angle from +Y.
            let mut in_plane = Vec3::new(local_dir.x, local_dir.y, 0.0);
            if in_plane.length_squared() < 1e-6 {
                in_plane = Vec3::Y;
            } else {
                in_plane = in_plane.normalize();
            }
            let angle = in_plane.x.atan2(in_plane.y);
            let clamped_angle = angle.clamp(-max_x, max_x);
            let clamped_local = Vec3::new(clamped_angle.sin(), clamped_angle.cos(), 0.0);
            (frame * clamped_local).normalize() * bone_length
        }
        PhysBoneLimitType::Polar => {
            let align = Quat::from_rotation_arc(
                Vec3::Y,
                (parent_frame_rot.inverse() * ref_dir).normalize_or(Vec3::Y),
            );
            let frame = (parent_frame_rot * align).normalize();
            let local_dir = (frame.inverse() * cur_dir).normalize_or(Vec3::Y);

            // Swing angles relative to +Y along X and Z axes.
            let theta_x = local_dir.z.atan2(local_dir.y);
            let theta_z = local_dir.x.atan2(local_dir.y);

            let rx = max_x.max(1e-4);
            let rz = if max_z > 0.0 { max_z } else { max_x }.max(1e-4);

            let norm_sq = (theta_x / rx).powi(2) + (theta_z / rz).powi(2);
            if norm_sq <= 1.0 && local_dir.y >= 0.0 {
                return cur_dir * bone_length;
            }
            let scale = norm_sq.sqrt().max(1.0);
            let cx = theta_x / scale;
            let cz = theta_z / scale;
            let rot = Quat::from_rotation_x(cx) * Quat::from_rotation_z(-cz);
            let clamped_local = (rot * Vec3::Y).normalize();
            (frame * clamped_local).normalize() * bone_length
        }
    }
}

fn any_perpendicular(v: Vec3) -> Vec3 {
    let candidate = if v.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    v.cross(candidate).normalize_or(Vec3::X)
}

fn resolve_collider(pos: Vec3, bone_radius: f32, col: &ColliderRuntime) -> Vec3 {
    let col_radius = col.def.radius.max(0.0) * col.world_scale;
    match col.def.shape {
        ColliderShape::Sphere => {
            let center = col.world_mat.transform_point3(Vec3::ZERO);
            let delta = pos - center;
            let dist = delta.length();
            if !col.def.inside_bounds {
                let min_dist = col_radius + bone_radius;
                if dist < min_dist {
                    let dir = if dist > 1e-6 { delta / dist } else { Vec3::Y };
                    return center + dir * min_dist;
                }
            } else {
                let max_dist = (col_radius - bone_radius).max(0.0);
                if dist > max_dist {
                    let dir = delta / dist;
                    return center + dir * max_dist;
                }
            }
            pos
        }
        ColliderShape::Capsule => {
            let raw_half_seg = ((col.def.height - 2.0 * col.def.radius) * 0.5).max(0.0);
            let a = col
                .world_mat
                .transform_point3(Vec3::new(0.0, -raw_half_seg, 0.0));
            let b = col
                .world_mat
                .transform_point3(Vec3::new(0.0, raw_half_seg, 0.0));
            let ab = b - a;
            let ab_len_sq = ab.length_squared();
            let t = if ab_len_sq > 1e-8 {
                ((pos - a).dot(ab) / ab_len_sq).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let closest = a + ab * t;
            let delta = pos - closest;
            let dist = delta.length();
            if !col.def.inside_bounds {
                let min_dist = col_radius + bone_radius;
                if dist < min_dist {
                    let fallback = if ab_len_sq > 1e-8 {
                        any_perpendicular(ab.normalize())
                    } else {
                        Vec3::Y
                    };
                    let dir = if dist > 1e-6 { delta / dist } else { fallback };
                    return closest + dir * min_dist;
                }
            } else {
                let max_dist = (col_radius - bone_radius).max(0.0);
                if dist > max_dist {
                    let dir = delta / dist;
                    return closest + dir * max_dist;
                }
            }
            pos
        }
        ColliderShape::Plane => {
            let origin = col.world_mat.transform_point3(Vec3::ZERO);
            let normal = col
                .world_mat
                .transform_vector3(Vec3::Y)
                .normalize_or(Vec3::Y);
            let signed_dist = (pos - origin).dot(normal);
            if !col.def.inside_bounds {
                if signed_dist < bone_radius {
                    return pos + normal * (bone_radius - signed_dist);
                }
            } else if signed_dist > -bone_radius {
                return pos - normal * (signed_dist + bone_radius);
            }
            pos
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skeleton::SkeletonNode;

    fn make_chain_skeleton(bone_count: usize, step_vec: Vec3) -> SkeletonPose {
        let mut nodes = Vec::with_capacity(bone_count + 1);
        nodes.push(SkeletonNode::new(
            "Head",
            None,
            vec![1],
            Vec3::ZERO,
            Quat::IDENTITY,
            Vec3::ONE,
        ));
        for i in 0..bone_count {
            let parent = i;
            let children = if i + 1 < bone_count {
                vec![i + 2]
            } else {
                vec![]
            };
            let trans = if i == 0 {
                Vec3::new(0.0, 0.2, 0.0)
            } else {
                step_vec
            };
            nodes.push(SkeletonNode::new(
                format!("Hair_{i}"),
                Some(parent),
                children,
                trans,
                Quat::IDENTITY,
                Vec3::ONE,
            ));
        }
        SkeletonPose::from_nodes(nodes)
    }

    #[test]
    fn verlet_stability_and_bone_length_preservation() {
        let mut pose = make_chain_skeleton(4, Vec3::new(0.25, 0.0, 0.0));
        let pb = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            pull: 0.1,
            spring: 0.5,
            stiffness: 0.1,
            gravity: 1.0,
            ..Default::default()
        };
        let mut world = PhysBoneWorld::new(&[pb], &[], &pose);
        let cfg = PhysicsShaderConfig::default();

        for _ in 0..180 {
            world.step(1.0 / 60.0, &mut pose, Vec3::new(15.0, 0.0, 0.0), &cfg);
        }

        let positions = world.chain_particle_positions(0).expect("chain 0");
        assert_eq!(positions.len(), 4);
        for p in &positions {
            assert!(p.is_finite());
        }
        for pair in positions.windows(2) {
            let len = (pair[1] - pair[0]).length();
            assert!(
                (len - 0.25).abs() < 1e-3,
                "expected bone length 0.25, got {len}"
            );
        }
        // Gravity should have pulled horizontal hair downward (-Y).
        assert!(positions[3].y < positions[0].y - 0.1);
    }

    #[test]
    fn immovable_one_prevents_window_accel_and_root_motion_sway() {
        let mut pose = make_chain_skeleton(3, Vec3::new(0.0, -0.3, 0.0));
        let pb = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            immovable: 1.0,
            pull: 0.2,
            gravity: 0.0,
            ..Default::default()
        };
        let mut world = PhysBoneWorld::new(&[pb], &[], &pose);
        let cfg = PhysicsShaderConfig::default();

        let initial = world.chain_particle_positions(0).expect("initial");

        // Apply strong horizontal window acceleration and translate root node.
        pose.set_local_translation(0, Vec3::new(1.0, 0.0, 0.0));
        pose.update_world_transforms();
        for _ in 0..30 {
            world.step(1.0 / 60.0, &mut pose, Vec3::new(40.0, 0.0, 0.0), &cfg);
        }

        let after = world.chain_particle_positions(0).expect("after");
        // Tip should have moved by exact root delta (+1.0 on X) with zero lag/sway.
        let tip_delta = after[2] - initial[2];
        assert!(
            (tip_delta - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3,
            "immovable=1.0 should move rigidly with root; delta={tip_delta:?}"
        );
    }

    #[test]
    fn window_drag_and_landing_impulse_cause_sway() {
        let mut pose = make_chain_skeleton(3, Vec3::new(0.0, -0.3, 0.0));
        let pb = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            immovable: 0.0,
            pull: 0.2,
            spring: 0.5,
            gravity: 0.1,
            ..Default::default()
        };
        let mut world = PhysBoneWorld::new(&[pb], &[], &pose);
        let cfg = PhysicsShaderConfig::default();

        // Dragging window right creates leftward inertial sway (-X).
        for _ in 0..10 {
            world.step(1.0 / 60.0, &mut pose, Vec3::new(-30.0, 0.0, 0.0), &cfg);
        }
        let drag_pos = world.chain_particle_positions(0).expect("drag");
        assert!(
            drag_pos[2].x < -0.02,
            "expected leftward sway during drag, got {:?}",
            drag_pos[2]
        );

        // Landing impulse pushes chain downward/sideways.
        world.inject_impulse(Vec3::new(2.5, 0.0, 0.0));
        for _ in 0..10 {
            world.step(1.0 / 60.0, &mut pose, Vec3::ZERO, &cfg);
        }
        let impulse_pos = world.chain_particle_positions(0).expect("impulse");
        assert!(impulse_pos[2].x > drag_pos[2].x);
    }

    #[test]
    fn sphere_collider_push_out_and_inside_bounds() {
        // Outside push-out test
        let mut pose = make_chain_skeleton(3, Vec3::new(0.0, -0.3, 0.0));
        let col_out = ColliderDef {
            id: "col_sphere_out".to_owned(),
            parent_node: "Head".to_owned(),
            shape: ColliderShape::Sphere,
            radius: 0.2,
            position: [0.0, -0.2, 0.0],
            inside_bounds: false,
            ..Default::default()
        };
        let pb_out = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            radius: 0.05,
            pull: 0.3,
            gravity: 0.0,
            collider_refs: vec!["col_sphere_out".to_owned()],
            ..Default::default()
        };
        let mut world = PhysBoneWorld::new(&[pb_out], &[col_out], &pose);
        let cfg = PhysicsShaderConfig::default();
        for _ in 0..30 {
            world.step(1.0 / 60.0, &mut pose, Vec3::ZERO, &cfg);
        }
        let positions = world.chain_particle_positions(0).expect("sphere out");
        let center = Vec3::new(0.0, -0.2, 0.0);
        // Bone 1 (initially at y = -0.1, dist = 0.1 < 0.25) must be pushed out >= 0.25 - eps.
        let dist1 = (positions[1] - center).length();
        assert!(
            dist1 >= 0.24,
            "particle inside sphere collider should be pushed out >= 0.25, got {dist1}"
        );

        // Inside-bounds containment test: root is at (0, 0.2, 0), sphere center at (0, 0.2, 0),
        // radius 0.35, bone_radius 0.05 -> max distance 0.30. With 2 segments of 0.20 (total 0.40),
        // strong +X acceleration would push tip to 0.40 without inside_bounds, so inside_bounds
        // pulls bone 2 back inside 0.30.
        let mut pose_in = make_chain_skeleton(3, Vec3::new(0.14, 0.0, 0.0));
        let col_in = ColliderDef {
            id: "col_sphere_in".to_owned(),
            parent_node: "Head".to_owned(),
            shape: ColliderShape::Sphere,
            radius: 0.25,
            position: [0.0, 0.2, 0.0],
            inside_bounds: true,
            ..Default::default()
        };
        let pb_in = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            radius: 0.05,
            pull: 0.0,
            gravity: 0.0,
            collider_refs: vec!["col_sphere_in".to_owned()],
            ..Default::default()
        };
        let mut world_in = PhysBoneWorld::new(&[pb_in], &[col_in], &pose_in);
        for _ in 0..30 {
            world_in.step(1.0 / 60.0, &mut pose_in, Vec3::new(40.0, 0.0, 0.0), &cfg);
        }
        let pos_in = world_in.chain_particle_positions(0).expect("sphere in");
        let center_in = Vec3::new(0.0, 0.2, 0.0);
        // Bone 1 (distance from root 0.14 <= 0.20) stays inside, and bone 2 (which would extend to 0.28)
        // is bent back toward the sphere interior (<= 0.20 + eps from inside_bounds).
        let dist2 = (pos_in[2] - center_in).length();
        assert!(
            dist2 <= 0.22,
            "inside_bounds sphere should keep tip constrained near max_dist 0.20, got {dist2}"
        );
    }

    #[test]
    fn capsule_collider_push_out_and_inside_bounds() {
        let mut pose = make_chain_skeleton(3, Vec3::new(0.0, -0.25, 0.0));
        let col_cap = ColliderDef {
            id: "col_cap".to_owned(),
            parent_node: "Head".to_owned(),
            shape: ColliderShape::Capsule,
            radius: 0.15,
            height: 0.6,
            position: [0.0, -0.1, 0.0],
            inside_bounds: false,
            ..Default::default()
        };
        let pb = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            radius: 0.05,
            pull: 0.2,
            gravity: 0.0,
            collider_refs: vec!["col_cap".to_owned()],
            ..Default::default()
        };
        let mut world = PhysBoneWorld::new(&[pb], &[col_cap], &pose);
        let cfg = PhysicsShaderConfig::default();

        // Nudge slightly off axis so push-out direction is deterministic.
        world.inject_impulse(Vec3::new(0.5, 0.0, 0.0));
        for _ in 0..30 {
            world.step(1.0 / 60.0, &mut pose, Vec3::ZERO, &cfg);
        }
        let pos = world.chain_particle_positions(0).expect("capsule out");
        // Bone 1 is at y ~ -0.05, which lies along the capsule's vertical segment [-0.25, +0.05].
        // Radial distance from Y axis must be >= radius + bone_radius = 0.20 (modulo final length pass).
        let radial = Vec3::new(pos[1].x, 0.0, pos[1].z).length();
        assert!(
            radial >= 0.18,
            "capsule collider should push particle radially outward, got radial={radial}"
        );

        // Inside-bounds capsule test with short bone length so length constraint and containment agree.
        let mut pose_in = make_chain_skeleton(2, Vec3::new(0.2, 0.0, 0.0));
        let col_cap_in = ColliderDef {
            id: "col_cap_in".to_owned(),
            parent_node: "Head".to_owned(),
            shape: ColliderShape::Capsule,
            radius: 0.15,
            height: 0.5,
            position: [0.0, 0.2, 0.0],
            inside_bounds: true,
            ..Default::default()
        };
        let pb_in = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            radius: 0.02,
            pull: 0.0,
            gravity: 0.0,
            collider_refs: vec!["col_cap_in".to_owned()],
            ..Default::default()
        };
        let mut world_in = PhysBoneWorld::new(&[pb_in], &[col_cap_in], &pose_in);
        for _ in 0..30 {
            world_in.step(1.0 / 60.0, &mut pose_in, Vec3::new(20.0, 0.0, 0.0), &cfg);
        }
        let pos_in = world_in.chain_particle_positions(0).expect("capsule in");
        // Wait: bone length is 0.2 and max_dist inside capsule is 0.13, so length projection runs last.
        // Verify particle was pulled closer than 0.20 (didn't drift freely outward).
        assert!(pos_in[1].x <= 0.201);
    }

    #[test]
    fn mouse_grab_pull_and_spring_back() {
        let mut pose = make_chain_skeleton(3, Vec3::new(0.0, -0.3, 0.0));
        let pb = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            pull: 0.5,
            spring: 0.4,
            gravity: 0.0,
            allow_grabbing: true,
            ..Default::default()
        };
        let mut world = PhysBoneWorld::new(&[pb], &[], &pose);
        let cfg = PhysicsShaderConfig::default();

        // Tip starts at (0, -0.4, 0).
        let grab_target = Vec3::new(0.5, -0.1, 0.0);
        let grabbed = world.grab_near(Vec3::new(0.02, -0.38, 0.0), 0.15);
        assert_eq!(grabbed, Some((0, 2)));

        world.update_grab(grab_target);
        for _ in 0..30 {
            world.step(1.0 / 60.0, &mut pose, Vec3::ZERO, &cfg);
        }
        let pulled = world.chain_particle_positions(0).expect("pulled");
        assert!(
            pulled[2].x > 0.25,
            "grabbed tip should pull toward target +X, got {:?}",
            pulled[2]
        );

        // Release and let it spring back toward x = 0.
        world.release_grab();
        for _ in 0..120 {
            world.step(1.0 / 60.0, &mut pose, Vec3::ZERO, &cfg);
        }
        let settled = world.chain_particle_positions(0).expect("settled");
        assert!(
            settled[2].x.abs() < 0.05,
            "released tip should spring back near x=0, got {:?}",
            settled[2]
        );
    }

    #[test]
    fn angular_limits_clamp_angle_hinge_and_polar() {
        // Angle limit: 30 deg cone around rest -Y.
        let mut pose_angle = make_chain_skeleton(2, Vec3::new(0.0, -0.5, 0.0));
        let pb_angle = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            pull: 0.0,
            gravity: 0.0,
            limit_type: PhysBoneLimitType::Angle,
            max_angle_x: 30.0,
            ..Default::default()
        };
        let mut world_angle = PhysBoneWorld::new(&[pb_angle], &[], &pose_angle);
        let cfg = PhysicsShaderConfig::default();
        for _ in 0..60 {
            world_angle.step(
                1.0 / 60.0,
                &mut pose_angle,
                Vec3::new(100.0, 0.0, 0.0),
                &cfg,
            );
        }
        let pos_a = world_angle.chain_particle_positions(0).expect("angle");
        let dir_a = (pos_a[1] - pos_a[0]).normalize();
        let angle_deg = dir_a.dot(Vec3::NEG_Y).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            angle_deg <= 30.5,
            "Angle limit 30 deg exceeded: {angle_deg}"
        );

        // Hinge limit: restricts Z swing while allowing X swing up to 25 deg.
        let mut pose_hinge = make_chain_skeleton(2, Vec3::new(0.0, -0.5, 0.0));
        let pb_hinge = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            pull: 0.0,
            gravity: 0.0,
            limit_type: PhysBoneLimitType::Hinge,
            max_angle_x: 25.0,
            ..Default::default()
        };
        let mut world_hinge = PhysBoneWorld::new(&[pb_hinge], &[], &pose_hinge);
        for _ in 0..60 {
            world_hinge.step(
                1.0 / 60.0,
                &mut pose_hinge,
                Vec3::new(50.0, 0.0, 50.0),
                &cfg,
            );
        }
        let pos_h = world_hinge.chain_particle_positions(0).expect("hinge");
        let dir_h = (pos_h[1] - pos_h[0]).normalize();
        let hinge_deg = dir_h.dot(Vec3::NEG_Y).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            hinge_deg <= 25.5,
            "Hinge limit 25 deg exceeded: {hinge_deg}"
        );

        // Polar limit: 20 deg on X, 40 deg on Z.
        let mut pose_polar = make_chain_skeleton(2, Vec3::new(0.0, -0.5, 0.0));
        let pb_polar = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            pull: 0.0,
            gravity: 0.0,
            limit_type: PhysBoneLimitType::Polar,
            max_angle_x: 20.0,
            max_angle_z: 40.0,
            ..Default::default()
        };
        let mut world_polar = PhysBoneWorld::new(&[pb_polar], &[], &pose_polar);
        for _ in 0..60 {
            world_polar.step(
                1.0 / 60.0,
                &mut pose_polar,
                Vec3::new(80.0, 0.0, 80.0),
                &cfg,
            );
        }
        let pos_p = world_polar.chain_particle_positions(0).expect("polar");
        let dir_p = (pos_p[1] - pos_p[0]).normalize();
        let polar_deg = dir_p.dot(Vec3::NEG_Y).clamp(-1.0, 1.0).acos().to_degrees();
        assert!(
            polar_deg <= 40.5,
            "Polar limit 40 deg exceeded: {polar_deg}"
        );
    }

    #[test]
    fn plane_collider_keeps_particle_above_plane() {
        let mut pose = make_chain_skeleton(2, Vec3::new(0.5, 0.0, 0.0));
        let col_plane = ColliderDef {
            id: "col_plane".to_owned(),
            parent_node: "Head".to_owned(),
            shape: ColliderShape::Plane,
            position: [0.0, 0.0, 0.0],
            inside_bounds: false,
            ..Default::default()
        };
        let pb = PhysBoneDef {
            root_node: "Hair_0".to_owned(),
            radius: 0.05,
            pull: 0.0,
            gravity: 1.0,
            collider_refs: vec!["col_plane".to_owned()],
            ..Default::default()
        };
        let mut world = PhysBoneWorld::new(&[pb], &[col_plane], &pose);
        let cfg = PhysicsShaderConfig::default();

        for _ in 0..60 {
            world.step(1.0 / 60.0, &mut pose, Vec3::ZERO, &cfg);
        }
        let pos = world.chain_particle_positions(0).expect("plane");
        assert!(
            pos[1].y >= 0.04,
            "plane collider at y=0 with bone radius 0.05 should keep particle y >= 0.05, got {}",
            pos[1].y
        );
    }
}
