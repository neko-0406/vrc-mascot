use std::collections::VecDeque;
use std::f32::consts::PI;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use vrc_schema::AudioLinkData;
use wasapi::{Direction, SampleType, StreamMode};

use crate::error::RendererError;

/// Number of mono samples processed per AudioLink FFT window.
pub const FFT_SIZE: usize = 1024;
/// Hop size (50% overlap) when streaming samples into [`AudioLinkAnalyzer::push_samples`].
pub const FFT_HOP: usize = 512;

/// Pure CPU 1024-sample Hann-windowed FFT analyzer producing 4-band AudioLink data + 32-frame history.
pub struct AudioLinkAnalyzer {
    sample_rate: u32,
    fft: Arc<dyn Fft<f32>>,
    hann_window: Vec<f32>,
    fft_buffer: Vec<Complex<f32>>,
    scratch: Vec<Complex<f32>>,
    sample_ring: VecDeque<f32>,
    data: AudioLinkData,
    attack: f32,
    release: f32,
}

impl std::fmt::Debug for AudioLinkAnalyzer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioLinkAnalyzer")
            .field("sample_rate", &self.sample_rate)
            .field("buffered_samples", &self.sample_ring.len())
            .field("bands", &self.data.bands)
            .field("active", &self.data.active)
            .finish()
    }
}

impl AudioLinkAnalyzer {
    /// Creates a new analyzer configured for `sample_rate` Hz mono audio.
    pub fn new(sample_rate: u32) -> Self {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_SIZE);
        let scratch_len = fft.get_inplace_scratch_len();
        let hann_window = (0..FFT_SIZE)
            .map(|i| {
                let phase = (2.0 * PI * i as f32) / ((FFT_SIZE - 1) as f32);
                0.5 * (1.0 - phase.cos())
            })
            .collect();

        Self {
            sample_rate: sample_rate.max(8_000),
            fft,
            hann_window,
            fft_buffer: vec![Complex::new(0.0, 0.0); FFT_SIZE],
            scratch: vec![Complex::new(0.0, 0.0); scratch_len],
            sample_ring: VecDeque::with_capacity(FFT_SIZE * 2),
            data: AudioLinkData::default(),
            attack: 0.65,
            release: 0.22,
        }
    }

    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Returns the latest computed [`AudioLinkData`].
    pub fn data(&self) -> AudioLinkData {
        self.data
    }

    /// Resets smoothed bands and history to zero.
    pub fn reset(&mut self) {
        self.sample_ring.clear();
        self.data = AudioLinkData::default();
    }

    /// Pushes arbitrary-length mono samples and runs FFT whenever at least 1024 samples are queued.
    pub fn push_samples(&mut self, mono_samples: &[f32]) -> Option<AudioLinkData> {
        self.sample_ring.extend(mono_samples.iter().copied());
        let mut updated = None;
        let mut window_buf = [0.0_f32; FFT_SIZE];

        while self.sample_ring.len() >= FFT_SIZE {
            for (dst, src) in window_buf
                .iter_mut()
                .zip(self.sample_ring.iter().take(FFT_SIZE))
            {
                *dst = *src;
            }
            updated = Some(self.analyze_frame(&window_buf));
            let drain_count = FFT_HOP.min(self.sample_ring.len());
            self.sample_ring.drain(..drain_count);
        }

        updated
    }

    /// Analyzes a single 1024-sample window (zero-padded if shorter) and updates smoothed bands & 32-frame history.
    pub fn analyze_frame(&mut self, samples: &[f32]) -> AudioLinkData {
        for i in 0..FFT_SIZE {
            let sample = samples.get(i).copied().unwrap_or(0.0);
            self.fft_buffer[i] = Complex::new(sample * self.hann_window[i], 0.0);
        }

        self.fft
            .process_with_scratch(&mut self.fft_buffer, &mut self.scratch);

        // Coherent gain of Hann window is 0.5; single-sided real FFT doubles non-DC bins -> divide by N/4.
        let norm_scale = 1.0 / (FFT_SIZE as f32 * 0.25);
        let bin_hz = self.sample_rate as f32 / FFT_SIZE as f32;

        let mut band_energy_sq = [0.0_f32; 4];
        let half_bins = FFT_SIZE / 2;

        for k in 1..half_bins {
            let freq = k as f32 * bin_hz;
            let band_idx = if (20.0..250.0).contains(&freq) {
                Some(0)
            } else if (250.0..1000.0).contains(&freq) {
                Some(1)
            } else if (1000.0..4000.0).contains(&freq) {
                Some(2)
            } else if (4000.0..=16000.0).contains(&freq) {
                Some(3)
            } else {
                None
            };

            if let Some(idx) = band_idx {
                let mag = self.fft_buffer[k].norm() * norm_scale;
                band_energy_sq[idx] += mag * mag;
            }
        }

        for (i, &energy_sq) in band_energy_sq.iter().enumerate() {
            let target = energy_sq.sqrt().clamp(0.0, 1.0);
            let prev = self.data.bands[i];
            let coeff = if target > prev {
                self.attack
            } else {
                self.release
            };
            self.data.bands[i] = (prev + (target - prev) * coeff).clamp(0.0, 1.0);
        }

        for hist_idx in (1..self.data.history.len()).rev() {
            self.data.history[hist_idx] = self.data.history[hist_idx - 1];
        }
        self.data.history[0] = self.data.bands;
        self.data.active = true;

        self.data
    }
}

/// Optional background WASAPI loopback audio capture feeding [`AudioLinkAnalyzer`].
///
/// When disabled or when no audio render endpoint is available, [`AudioLinkCapture::latest_data`]
/// gracefully returns zeroed [`AudioLinkData`] without failing.
pub struct AudioLinkCapture {
    enabled: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    shared_data: Arc<Mutex<AudioLinkData>>,
    worker: Option<JoinHandle<()>>,
}

impl std::fmt::Debug for AudioLinkCapture {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioLinkCapture")
            .field("enabled", &self.enabled.load(Ordering::Relaxed))
            .finish()
    }
}

impl AudioLinkCapture {
    /// Starts the background capture controller. If `enabled` is `false`, idles with zero CPU until enabled.
    pub fn new(enabled: bool) -> Self {
        let enabled_flag = Arc::new(AtomicBool::new(enabled));
        let stop_flag = Arc::new(AtomicBool::new(false));
        let shared_data = Arc::new(Mutex::new(AudioLinkData::default()));

        let worker_enabled = Arc::clone(&enabled_flag);
        let worker_stop = Arc::clone(&stop_flag);
        let worker_data = Arc::clone(&shared_data);

        let worker = thread::Builder::new()
            .name("vrc-audiolink-wasapi".to_owned())
            .spawn(move || {
                capture_thread_main(worker_enabled, worker_stop, worker_data);
            })
            .ok();

        Self {
            enabled: enabled_flag,
            stop: stop_flag,
            shared_data,
            worker,
        }
    }

    /// Enables or disables live loopback capture. Disabling clears output to zeros.
    pub fn set_enabled(&self, enabled: bool) {
        let prev = self.enabled.swap(enabled, Ordering::AcqRel);
        if prev && !enabled {
            if let Ok(mut guard) = self.shared_data.lock() {
                *guard = AudioLinkData::default();
            }
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    /// Returns the most recent [`AudioLinkData`], or zeros when disabled / unavailable.
    pub fn latest_data(&self) -> AudioLinkData {
        if !self.is_enabled() {
            return AudioLinkData::default();
        }
        self.shared_data
            .lock()
            .map(|guard| *guard)
            .unwrap_or_default()
    }
}

impl Drop for AudioLinkCapture {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(handle) = self.worker.take() {
            let _ = handle.join();
        }
    }
}

fn capture_thread_main(
    enabled: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    shared_data: Arc<Mutex<AudioLinkData>>,
) {
    let com_hr = wasapi::initialize_mta();
    let should_uninit = com_hr.is_ok();

    while !stop.load(Ordering::Acquire) {
        if !enabled.load(Ordering::Acquire) {
            thread::sleep(Duration::from_millis(50));
            continue;
        }

        if let Err(err) = run_wasapi_loopback_session(&enabled, &stop, &shared_data) {
            tracing::debug!("AudioLink WASAPI loopback unavailable: {err}");
            if let Ok(mut guard) = shared_data.lock() {
                *guard = AudioLinkData::default();
            }
            // Wait before retrying so we don't spin if no audio device is attached.
            for _ in 0..20 {
                if stop.load(Ordering::Acquire) || !enabled.load(Ordering::Acquire) {
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    }

    if should_uninit {
        wasapi::deinitialize();
    }
}

fn run_wasapi_loopback_session(
    enabled: &AtomicBool,
    stop: &AtomicBool,
    shared_data: &Mutex<AudioLinkData>,
) -> Result<(), RendererError> {
    let enumerator =
        wasapi::DeviceEnumerator::new().map_err(|e| RendererError::Audio(e.to_string()))?;
    let device = enumerator
        .get_default_device(Direction::Render)
        .map_err(|e| RendererError::Audio(e.to_string()))?;
    let mut audio_client = device
        .get_iaudioclient()
        .map_err(|e| RendererError::Audio(e.to_string()))?;

    let mix_fmt = audio_client
        .get_mixformat()
        .map_err(|e| RendererError::Audio(e.to_string()))?;
    let sample_rate = mix_fmt.get_samplespersec();
    let channels = mix_fmt.get_nchannels() as usize;
    let bits_per_sample = mix_fmt.get_bitspersample();
    let sample_type = mix_fmt
        .get_subformat()
        .map_err(|e| RendererError::Audio(e.to_string()))?;
    let block_align = mix_fmt.get_blockalign() as usize;

    if channels == 0 || block_align == 0 {
        return Err(RendererError::Audio(
            "WASAPI mix format reported 0 channels".to_owned(),
        ));
    }

    let (_def_period, min_period) = audio_client
        .get_device_period()
        .map_err(|e| RendererError::Audio(e.to_string()))?;

    let mode = StreamMode::PollingShared {
        autoconvert: false,
        buffer_duration_hns: min_period.max(200_000),
    };
    audio_client
        .initialize_client(&mix_fmt, Direction::Capture, mode, None)
        .map_err(|e| RendererError::Audio(e.to_string()))?;

    let capture_client = audio_client
        .get_audiocaptureclient()
        .map_err(|e| RendererError::Audio(e.to_string()))?;
    audio_client
        .start_stream()
        .map_err(|e| RendererError::Audio(e.to_string()))?;

    let mut byte_queue: VecDeque<u8> = VecDeque::with_capacity(block_align * 4096);
    let mut analyzer = AudioLinkAnalyzer::new(sample_rate);
    let mut mono_scratch = Vec::with_capacity(2048);

    while !stop.load(Ordering::Acquire) && enabled.load(Ordering::Acquire) {
        let packet_frames = capture_client
            .get_next_packet_size()
            .map_err(|e| RendererError::Audio(e.to_string()))?
            .unwrap_or(0);

        if packet_frames > 0 {
            capture_client
                .read_from_device_to_deque(&mut byte_queue)
                .map_err(|e| RendererError::Audio(e.to_string()))?;

            let available_frames = byte_queue.len() / block_align;
            if available_frames > 0 {
                let bytes_to_consume = available_frames * block_align;
                let frame_bytes: Vec<u8> = byte_queue.drain(..bytes_to_consume).collect();
                decode_to_mono(
                    &frame_bytes,
                    channels,
                    bits_per_sample,
                    sample_type,
                    &mut mono_scratch,
                );

                if let Some(updated) = analyzer.push_samples(&mono_scratch) {
                    if let Ok(mut guard) = shared_data.lock() {
                        *guard = updated;
                    }
                }
            }
        } else {
            thread::sleep(Duration::from_millis(16));
        }
    }

    let _ = audio_client.stop_stream();
    Ok(())
}

fn decode_to_mono(
    bytes: &[u8],
    channels: usize,
    bits_per_sample: u16,
    sample_type: SampleType,
    out: &mut Vec<f32>,
) {
    out.clear();
    let bytes_per_sample = (bits_per_sample as usize) / 8;
    if bytes_per_sample == 0 || channels == 0 {
        return;
    }
    let frame_size = bytes_per_sample * channels;
    let inv_ch = 1.0 / channels as f32;

    for frame in bytes.chunks_exact(frame_size) {
        let mut sum = 0.0_f32;
        for ch in 0..channels {
            let s_bytes = &frame[ch * bytes_per_sample..(ch + 1) * bytes_per_sample];
            let sample = match (sample_type, bits_per_sample) {
                (SampleType::Float, 32) => f32::from_le_bytes(s_bytes.try_into().unwrap_or([0; 4])),
                (SampleType::Int, 16) => {
                    i16::from_le_bytes(s_bytes.try_into().unwrap_or([0; 2])) as f32 / 32768.0
                }
                (SampleType::Int, 32) => {
                    i32::from_le_bytes(s_bytes.try_into().unwrap_or([0; 4])) as f32
                        / 2_147_483_648.0
                }
                _ => 0.0,
            };
            sum += sample;
        }
        out.push((sum * inv_ch).clamp(-1.0, 1.0));
    }
}
