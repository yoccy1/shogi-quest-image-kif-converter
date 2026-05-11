# Publication Notes

GitHub公開前に確認すること。

## 公開に含めない方がよいもの

- `data/samples/screenshots_by_app_piece_style/`
  - 約118MBあり、第三者アプリのスクリーンショットを含む可能性がある。
- `reports/`
  - 評価結果としては有用だが、GitHub公開用には重く、再生成可能。
- `third_party/ShogiVision/`
  - 第三者由来。LICENSE/モデル再配布可否を確認するまで公開しない。

`.gitignore` では上記を除外済み。

## 公開に含めてよい候補

- `src/shogi_gazo_desktop/`
- `tools/` のうちPC認識/評価/可視化に必要なもの
- `assets/android_templates/` のうち自前生成/再配布可能なもの
- `docs/`
- 小さなサンプル画像を別途作成した `examples/`

## 次に必要な判断

GUIライブラリを決める。

候補:

- PySide6: GitHub公開ツールとして自然。画像ビュー、表、詳細パネルを作りやすい。
- Tkinter: 標準ライブラリで軽いが、画像比較UIは作り込みが必要。
- Web UI: FastAPI/Streamlit等。配布は簡単だが、ローカルGUI感は薄い。

現時点の推奨は PySide6。ただし、まずはCLIで「画像を読み込む、盤面を推定する、結果をHTML/CSVに出す」を安定させてからGUIを被せる。
