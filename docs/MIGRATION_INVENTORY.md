# Migration Inventory

作成日: 2026-05-11

Androidアプリから、PC上で動くGUI付きツールへ方針転換するための初期コピー記録です。元ファイルは移動せず、すべてコピーで配置しています。

## コピーしたもの

### Pythonツール

コピー元:

- `tools/*.py`
- `tools/requirements.txt`
- `tools/README.md`

コピー先:

- `shogi_gazo_desktop/tools/`

理由:

- PCツール化で直接使う認識、評価、可視化、offline probeがPython資産として残っているため。
- Android実機評価用スクリプトも一旦コピーしたが、PCツール本体では非推奨/整理対象。

### テンプレートassets

コピー元:

- `app/src/main/assets/app_piece_templates.json`
- `app/src/main/assets/hand_digit_templates.json`
- `app/src/main/assets/hand_layouts.json`
- `app/src/main/assets/hand_piece_templates.json`
- `app/src/main/assets/known_position_samples.json`
- `app/src/main/res/drawable-nodpi/shogi_board.png`
- `app/src/main/res/drawable-nodpi/shogi_pieces.png`

コピー先:

- `shogi_gazo_desktop/assets/android_templates/`
- `shogi_gazo_desktop/assets/legacy_drawables/`

理由:

- Android実装から切り離しても、既存テンプレート・既知局面データはPC認識ツールの初期資産として参照できるため。
- 盤/駒画像はPC GUIの描画・プレビュー用の初期素材として使える可能性があるため。

### 主サンプル/ラベル

コピー元:

- `tools/samples/screenshots_by_app_piece_style/`
- `tools/samples/labels/boards_by_app_piece_style/`
- `tools/samples/piece_style_manifest.csv`

コピー先:

- `shogi_gazo_desktop/data/samples/`

理由:

- 現在の認識改善・評価の主セットが `screenshots_by_app_piece_style` / `boards_by_app_piece_style` であるため。
- glyph系の古い/追加サンプルは初期コピーから除外した。必要になったら追加する。

### 評価/可視化結果

コピー元:

- `tools/out/piece_accuracy_review_current_full86_20260510_light/`
- `tools/out/shogivision_probe_all_86_20260510/`

コピー先:

- `shogi_gazo_desktop/reports/`

理由:

- GUIツール化前に現状精度を目視確認する基準として必要。
- ShogiVisionは置換ではなく比較/補助候補として参照する。

### AI/作業文脈

コピー元:

- `AI_CONTEXT/README.md`
- `AI_CONTEXT/PROJECT_MAP.md`
- `AI_CONTEXT/DEVELOPMENT_AND_QA.md`
- `AI_CONTEXT/handoffs/HANDOFF_2026-05-10_piece-accuracy-visual-full86.md`
- `AI_CONTEXT/handoffs/HANDOFF_2026-05-11_piece-accuracy-html-lightweight.md`
- `AI_CONTEXT/handoffs/HANDOFF_2026-05-10_shogivision-local-setup.md`
- `AI_CONTEXT/handoffs/HANDOFF_2026-05-10_shogivision-offline-probe.md`

コピー先:

- `shogi_gazo_desktop/docs/ai_context/`

理由:

- 新しいAI/開発者が「なぜPCツールに切り替えたか」「現在の精度がどこまで悪いか」を追えるようにするため。

### ShogiVision参照

コピー元:

- `ShogiVision-master/ShogiVision-master/`

コピー先:

- `shogi_gazo_desktop/third_party/ShogiVision/`

除外:

- `venv/`
- `__pycache__/`

理由:

- ShogiVisionの実験結果とモデルは今後のPCツール候補として重要。
- ただし第三者由来なので、GitHub公開時はライセンス/再配布可否の確認が必要。

## コピーしなかったもの

- `app/`
- `gradle/`
- `.gradle/`
- `.idea/`
- `.kotlin/`
- `build/`
- `gradlew`, `gradlew.bat`, `settings.gradle.kts`, `build.gradle.kts`, `gradle.properties`

理由:

- Androidアプリ開発を終了し、PCツールに移るため。

追加でコピーしなかったもの:

- `tools/out/android_device_eval/` 全体
- `tools/samples/screenshots_by_app_glyph/`
- `tools/samples/labels/boards_by_app_glyph/`
- `ShogiVision-master/ShogiVision-master/venv/`

理由:

- サイズが大きい、または現行PCツール化の主セットではないため。
- 必要になったら明示的に追加する。

## 現在のサイズ感

- 新フォルダ全体: 約174MB
- 主サンプル画像: 約118MB
- ShogiVision参照コピー: venv除外、modelsあり
- 軽量HTMLレビュー: 約5MB

## 次の整理候補

1. `tools/` 内のAndroid専用スクリプトを `legacy_android_tools/` へ分離する。
2. GUI本体を `src/shogi_gazo_desktop/` に作る。
3. 公開用サンプルを小さな `examples/` に切り出し、フルデータセットはGitHub外で配布する。
4. ShogiVisionのLICENSE/再配布可否を確認し、`third_party/ShogiVision/` を公開対象にするか決める。
