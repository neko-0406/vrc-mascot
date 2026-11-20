use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;

use crate::pipeline::DEPTH_FORMAT;

const STATE_IDLE: u8 = 0;
const STATE_COPIED: u8 = 1;
const STATE_MAPPING: u8 = 2;
const STATE_MAPPED_OK: u8 = 3;
const STATE_MAPPED_ERR: u8 = 4;

/// Alpha threshold separating click-through (`<= 0.05`) from interactive avatar pixels (`> 0.05`).
pub const HIT_TEST_ALPHA_THRESHOLD: f32 = 0.05;

/// Decoded result of querying an offscreen hit-test pixel at `(x, y)`.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct HitTestResult {
    pub alpha: f32,
    pub is_opaque: bool,
    pub node_index: Option<usize>,
    pub is_head: bool,
}

impl HitTestResult {
    /// Decodes a `(alpha_byte, node_id_low, node_id_high, part_flags)` pixel from `Rgba8Unorm`.
    pub fn from_pixel(pixel: [u8; 4]) -> Self {
        let alpha = pixel[0] as f32 / 255.0;
        let is_opaque = alpha > HIT_TEST_ALPHA_THRESHOLD;
        let encoded_id = (pixel[1] as u16) | ((pixel[2] as u16) << 8);
        let node_index = if !is_opaque || encoded_id == 0 {
            None
        } else {
            Some((encoded_id - 1) as usize)
        };
        let is_head = is_opaque && (pixel[3] & 0x01) != 0;
        Self {
            alpha,
            is_opaque,
            node_index,
            is_head,
        }
    }
}

/// Manages offscreen hit-test render targets and non-blocking async staging buffer readback.
#[derive(Debug)]
pub struct HitTestPass {
    width: u32,
    height: u32,
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_texture: wgpu::Texture,
    depth_view: wgpu::TextureView,
    staging_buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
    readback_state: Arc<AtomicU8>,
    cached_pixels: Vec<u8>,
}

impl HitTestPass {
    /// Creates hit-test offscreen targets and staging buffer for `(width, height)`.
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let (
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            staging_buffer,
            padded_bytes_per_row,
        ) = create_resources(device, w, h);

        Self {
            width: w,
            height: h,
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            staging_buffer,
            padded_bytes_per_row,
            readback_state: Arc::new(AtomicU8::new(STATE_IDLE)),
            cached_pixels: vec![0u8; (w as usize) * (h as usize) * 4],
        }
    }

    /// Recreates offscreen textures and staging buffer on viewport resize.
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let w = width.max(1);
        let h = height.max(1);
        if w == self.width && h == self.height {
            return;
        }
        if self.readback_state.load(Ordering::Acquire) == STATE_MAPPED_OK {
            self.staging_buffer.unmap();
        }
        self.readback_state = Arc::new(AtomicU8::new(STATE_IDLE));
        let (
            color_texture,
            color_view,
            depth_texture,
            depth_view,
            staging_buffer,
            padded_bytes_per_row,
        ) = create_resources(device, w, h);
        self.width = w;
        self.height = h;
        self.color_texture = color_texture;
        self.color_view = color_view;
        self.depth_texture = depth_texture;
        self.depth_view = depth_view;
        self.staging_buffer = staging_buffer;
        self.padded_bytes_per_row = padded_bytes_per_row;
        self.cached_pixels.clear();
        self.cached_pixels
            .resize((w as usize) * (h as usize) * 4, 0);
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn color_view(&self) -> &wgpu::TextureView {
        &self.color_view
    }

    pub fn depth_view(&self) -> &wgpu::TextureView {
        &self.depth_view
    }

    /// Returns `true` when the staging buffer is idle and ready to receive a new GPU copy.
    pub fn can_copy_this_frame(&self) -> bool {
        self.readback_state.load(Ordering::Acquire) == STATE_IDLE
    }

    /// Encodes `copy_texture_to_buffer` into `encoder` if the staging buffer is idle.
    pub fn encode_copy_to_staging(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.readback_state.load(Ordering::Acquire) != STATE_IDLE {
            return;
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &self.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.staging_buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.readback_state.store(STATE_COPIED, Ordering::Release);
    }

    /// Advances async staging buffer mapping without stalling the render thread.
    ///
    /// Call after submitting the command encoder that recorded [`Self::encode_copy_to_staging`].
    pub fn poll_readback(&mut self, device: &wgpu::Device) {
        if self.readback_state.load(Ordering::Acquire) == STATE_COPIED {
            self.readback_state.store(STATE_MAPPING, Ordering::Release);
            let flag = Arc::clone(&self.readback_state);
            self.staging_buffer
                .slice(..)
                .map_async(wgpu::MapMode::Read, move |res| {
                    let next = if res.is_ok() {
                        STATE_MAPPED_OK
                    } else {
                        STATE_MAPPED_ERR
                    };
                    flag.store(next, Ordering::Release);
                });
        }

        let _ = device.poll(wgpu::PollType::Poll);

        match self.readback_state.load(Ordering::Acquire) {
            STATE_MAPPED_OK => {
                if let Ok(view) = self.staging_buffer.slice(..).get_mapped_range() {
                    let row_bytes = (self.width as usize) * 4;
                    let padded = self.padded_bytes_per_row as usize;
                    for y in 0..self.height as usize {
                        let src_start = y * padded;
                        let dst_start = y * row_bytes;
                        if src_start + row_bytes <= view.len()
                            && dst_start + row_bytes <= self.cached_pixels.len()
                        {
                            self.cached_pixels[dst_start..dst_start + row_bytes]
                                .copy_from_slice(&view[src_start..src_start + row_bytes]);
                        }
                    }
                }
                self.staging_buffer.unmap();
                self.readback_state.store(STATE_IDLE, Ordering::Release);
            }
            STATE_MAPPED_ERR => {
                self.readback_state.store(STATE_IDLE, Ordering::Release);
            }
            _ => {}
        }
    }

    /// Queries the cached CPU hit-test pixel at `(x, y)` in O(1) time.
    pub fn hit_test(&self, x: u32, y: u32) -> HitTestResult {
        if x >= self.width || y >= self.height {
            return HitTestResult::default();
        }
        let offset = ((y as usize) * (self.width as usize) + (x as usize)) * 4;
        if offset + 4 > self.cached_pixels.len() {
            return HitTestResult::default();
        }
        HitTestResult::from_pixel([
            self.cached_pixels[offset],
            self.cached_pixels[offset + 1],
            self.cached_pixels[offset + 2],
            self.cached_pixels[offset + 3],
        ])
    }

    /// Replaces the cached pixel buffer directly (useful for testing or external updates).
    pub fn set_cached_pixels_for_test(&mut self, pixels: Vec<u8>) {
        if pixels.len() == self.cached_pixels.len() {
            self.cached_pixels = pixels;
        }
    }
}

fn create_resources(
    device: &wgpu::Device,
    width: u32,
    height: u32,
) -> (
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Texture,
    wgpu::TextureView,
    wgpu::Buffer,
    u32,
) {
    let color_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrc_hit_test_color"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("vrc_hit_test_depth"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let unpadded = width * 4;
    let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = unpadded.div_ceil(align) * align;
    let buffer_size = (padded_bytes_per_row as u64) * (height as u64);

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("vrc_hit_test_staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    (
        color_texture,
        color_view,
        depth_texture,
        depth_view,
        staging_buffer,
        padded_bytes_per_row,
    )
}
