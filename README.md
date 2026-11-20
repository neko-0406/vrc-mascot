# VRC Desktop Mascot

Windows 向けの透過最前面 3D デスクトップマスコットアプリケーションです。
Unity エディタを起動することなく VRChat 向け Unity プロジェクトからアバター（メッシュ・ボーン・BlendShape・PhysBone・VRCExpressionsMenu/FX・シェーダーパラメータ）を抽出し、`wgpu` + Win32 API による軽量な常駐表示、AI 自律行動・対話、VOICEVOX 音声合成・リップシンク連携を行います。

## クレート構成

- `crates/vrc-schema`: 中間データ (`mascot_manifest.json`)・設定 (`config.toml`)・共通ランタイム型定義
- `crates/vrc-extractor`: Unity プロジェクト外部解析エンジン & CLI (`avatar.glb` + `mascot_manifest.json` 出力)
- `crates/vrc-physics`: `VRCPhysBone` / `Collider` 互換 Verlet/PBD 物理演算
- `crates/vrc-renderer`: `wgpu` + WGSL トゥーンシェーダーレンダラー (lilToon / Poiyomi / UTS / Fur / AudioLink)
- `crates/vrc-ai`: Gemini Decisions API / 対話 LLM / VOICEVOX Engine クライアント & リップシンク
- `crates/vrc-mascot-app`: Windows 常駐デスクトップアプリ本体 (`winit` 2ウィンドウ + `egui` UI + Win32 制御 + トレイ常駐)

## 前提環境

- **OS**: Windows 10 (21H2以降) / Windows 11 (64bit)
- **Rust**: Stable (`x86_64-pc-windows-msvc`, `rustfmt`, `clippy`)
- **C/C++ ビルドツール**: Visual Studio Build Tools 2022 (C++ によるデスクトップ開発) + Windows 10/11 SDK
- **GPU**: DirectX 12 または Vulkan 1.2 対応 GPU
- **VOICEVOX Engine**（任意・音声合成利用時）: `http://127.0.0.1:50021`

## ビルド・テスト・実行

```powershell
# ワークスペース全体のビルド
cargo build --workspace

# テスト実行
cargo test --workspace

# Clippy チェック
cargo clippy --workspace --all-targets

# Unity プロジェクトからアバター抽出
cargo run -p vrc-extractor -- `
  --project "C:\Path\To\UnityProject" `
  --scene "Assets/SampleScene.unity" `
  --avatar "SampleAvatar" `
  --output ".\out\SampleAvatar"

# デスクトップマスコット起動
cargo run -p vrc-mascot-app -- --manifest ".\out\SampleAvatar\mascot_manifest.json"
```

## 設定と環境変数

- 実行時設定は `%APPDATA%/vrc-mascot/config.toml` に保存されます。
- 開発時のフォールバックとして `.env.example` を `.env` にコピーして使用できます（`.env` は `.gitignore` 対象であり、リポジトリへコミットしないでください）。

## Git-Flow 運用

- `main`: リリース・安定版ブランチ（直接コミット禁止）
- `develop`: 開発統合ブランチ（直接コミット禁止）
- `feature/<name>`, `fix/<name>`, `chore/<name>`: `develop` から分岐し、作業完了後に `--no-ff` で `develop` へマージ
