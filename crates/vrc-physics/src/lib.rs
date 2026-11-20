//! `VRCPhysBone` and `VRCPhysBoneCollider` Verlet/PBD simulation engine.

use glam::Vec3;
use vrc_schema::{ColliderDef, PhysBoneDef};

/// Minimal physics world skeleton holding PhysBone and collider definitions.
#[derive(Debug, Default)]
pub struct PhysicsWorld {
    pub phys_bones: Vec<PhysBoneDef>,
    pub colliders: Vec<ColliderDef>,
    pub gravity: Vec3,
}

impl PhysicsWorld {
    /// Creates an empty `PhysicsWorld` with standard downward gravity.
    pub fn new(phys_bones: Vec<PhysBoneDef>, colliders: Vec<ColliderDef>) -> Self {
        Self {
            phys_bones,
            colliders,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}
