pub mod animator;
pub mod avatar;
pub mod helpers;
pub mod material;
pub mod prefab;

pub use animator::{parse_animation_clip, parse_animator_controller, ParsedController};
pub use avatar::{
    extract_avatar_descriptor_data, extract_physbones_and_colliders, find_avatar_candidates,
    AvatarDescriptorData,
};
pub use material::{parse_material_asset, ExtractedMaterial};
pub use prefab::{SceneHierarchy, SceneNode, VirtualComponent, VirtualScene};
