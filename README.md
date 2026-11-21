# VRC Desktop Mascot

Windows 向けの透過最前面 3D デスクトップマスコットアプリケーションです。
Unity エディタを起動することなく VRChat 向け Unity プロジェクトからアバター（メッシュ・ボーン・BlendShape・PhysBone・VRCExpressionsMenu/FX・シェーダーパラメータ）を抽出し、`wgpu` + Win32 API による軽量な常駐表示、AI 自律行動・対話、VOICEVOX 音声合成・リップシンク連携を行います。

## クレート構成（6クレート・ワークスペース）

- `crates/vrc-schema`: 中間データ (`mascot_manifest.json`)・設定 (`config.toml`)・アバターレジストリ (`AvatarRegistry`)・共通ランタイム型定義
- `crates/vrc-extractor`: Unity プロジェクト外部解析エンジン & CLI（YAML/Prefab Variant 解決、Modular Avatar 仮想適用、`ufbx` 読込、`avatar.glb` + `mascot_manifest.json` 出力）
- `crates/vrc-physics`: スケルトン FK、`VRCPhysBone` / `Collider` 互換 Verlet/PBD 物理演算、自律モーション・撫で判定、FX ステートマシン
- `crates/vrc-renderer`: `wgpu` + WGSL 4パス・トゥーンレンダラー（lilToon / Poiyomi / UTS / Outline / Fur / WASAPI AudioLink / オフスクリーンヒットテスト）
- `crates/vrc-ai`: Gemini Decisions API / 対話 LLM / プライバシーフィルタ / VOICEVOX Engine クライアント / `rodio` 同期リップシンク再生
- `crates/vrc-mascot-app`: Windows 常駐デスクトップアプリ本体（単一 `winit` 2ウィンドウ構成、`egui` 4タブ UI、Win32 透過・クリック透過・ウィンドウ着席追従、システムトレイ常駐）

## 前提環境

- **OS**: Windows 10 (21H2以降) / Windows 11 (64bit)
- **Rust**: Stable (`x86_64-pc-windows-msvc`, `rustfmt`, `clippy`)
- **C/C++ ビルドツール**: Visual Studio Build Tools 2022 (C++ によるデスクトップ開発) + Windows 10/11 SDK
- **GPU**: DirectX 12 または Vulkan 1.2 対応 GPU
- **VOICEVOX Engine**（任意・音声合成利用時）: `http://127.0.0.1:50021`

## ビルド・テスト・実行

```powershell
# ワークスペース全体のビルド（開発用 / リリース）
cargo build --workspace
cargo build --workspace --release

# テスト・フォーマット・Clippy チェック
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace

# Unity プロジェクトからアバター抽出 CLI
# vrc-extractor --project <DIR> --scene <SCENE> --avatar <NAME> --output <DIR> [--no-modular-avatar] [--max-texture-size <PX>]
cargo run -p vrc-extractor -- `
  --project "C:\Path\To\UnityProject" `
  --scene "Assets/SampleScene.unity" `
  --avatar "SampleAvatar" `
  --output ".\out\SampleAvatar" `
  --max-texture-size 2048

# デスクトップマスコット起動 CLI
# vrc-mascot-app [--manifest <PATH>] [--show-window]
cargo run -p vrc-mascot-app -- --manifest ".\out\SampleAvatar\mascot_manifest.json"

# ステータス・設定ウィンドウを最初から開いて起動（登録アバターまたは内蔵フォールバックモデルを使用）
cargo run -p vrc-mascot-app -- --show-window
```

## 主な操作方法

- **移動・重力・着席**: アバター本体の不透明部分を左ドラッグで移動（離すとタスクバーまたは他ウィンドウ上辺へ落下・着席）
- **PhysBone 直接掴み**: `Shift` + 左ドラッグで髪や耳などの PhysBone を直接引っ張る
- **拡大・縮小**: アバター上でマウスホイール上下（`0.5x〜2.0x`）
- **メニュー・設定画面**: アバター右クリック、またはタスクトレイアイコンのメニュー／クリックから「ステータス・設定ウィンドウ」開閉・アバター切替・Expressions 操作・終了が可能（ステータスウィンドウは閉じてもトレイ常駐を継続します）

## 設定と環境変数

- 実行時設定は `%APPDATA%/vrc-mascot/config.toml` に保存されます（設定タブから全 6 セクション編集・保存可能）。
- 開発時のフォールバックとして `.env.example` を `.env` にコピーして使用できます（`.env` は `.gitignore` 対象であり、リポジトリへコミットしないでください）。

## Git-Flow 運用

- `main`: リリース・安定版ブランチ（直接コミット禁止）
- `develop`: 開発統合ブランチ（直接コミット禁止）
- `feature/<name>`, `fix/<name>`, `chore/<name>`: `develop` から分岐し、作業完了後に `--no-ff` で `develop` へマージ
