# Project Map

新しく作業するAI向けの構成メモです。細かい実装は各ファイルを読んで確認してください。

## ルート構成

- `app/`: Androidアプリ本体。
- `tools/`: ローカル画像処理、学習、評価、asset生成用のPython補助ツール。
- `tools/samples/`: スクリーンショット、ラベル、manifest。
- `app/src/main/assets/`: Androidアプリが読む認識用JSON assets。
- `docs/`: 既存の手順書、レビュー、過去の引き継ぎ。
- `AI_CONTEXT/`: 新規AI向けの入口と今後の引き継ぎ置き場。
- `AI_CONTEXT/reviews/`: 既存レビュー報告書のAI向け正本。
- `AI_CONTEXT/handoffs/`: 旧引き継ぎ資料と今後の新規引き継ぎの保存先。

## Androidアプリ

主な流れは、画像入力、解析設定、盤面/持ち駒認識、手動確認、KIF/SFEN出力です。

- `MainActivity`: Photo Picker、共有Intent受信、KIF保存/共有、SFEN共有、デバッグZIP共有。
- `ShogiViewModel`: UI状態、画像読み込み、解析パイプライン、OCR統合、手動補正、検証、KIF/SFEN生成。
- `ShogiGazoApp`: Jetpack Compose UI本体。
- `ShogiModels`: 駒、升、局面、手番、盤向きなどのモデル。
- `PositionValidator`: 未確定マス、玉数、持ち駒、二歩、行き所のない駒、低信頼マスを検証。
- `BitmapTools`: 盤検出、クロップ、81マス切り出し、空マス推定、持ち駒領域/候補推定。
- `ScriptBoardGridDetector`: スクリーンショットから9x9盤線を検出。
- `PieceTemplateRecognizer`: インク特徴、テンプレート、位置事前分布で駒種を推定。
- `OcrRecognizer`: ML Kit Japanese Text Recognition による補助候補生成。
- `HandCountAggregator`: 持ち駒候補を集約し、盤上枚数との整合を取る。
- `KifExporter` / `SfenExporter`: 局面をKIF/BOD、Shift_JISバイト列、SFENへ変換。

## Pythonツール

`tools/` はAndroidアプリに同梱しないローカル補助領域です。

`tools/README.md` には古い記述が一部あります。特に「only supported Python helper」という趣旨の記述は現状と合わず、実際には解析、評価、学習、asset生成、device evalの多数のrunnerがあります。AI作業ではこのファイルと `AI_CONTEXT/DEVELOPMENT_AND_QA.md` を合わせて確認してください。

- 盤面検出: `detect_board_grid.py`
- 盤面駒認識: `recognize_board_pieces.py`, `learned_piece_recognizer.py`
- 一括解析: `analyze_shogi_screenshot.py`, `run_analysis_by_app_piece_style.py`
- 持ち駒検出/認識: `detect_hand_areas.py`, `recognize_hand_pieces.py`
- 学習/asset生成: `train_piece_model.py`, `export_android_piece_templates.py`, `export_android_hand_assets.py`, `export_known_position_samples.py`
- 評価: `evaluate_piece_recognition.py`, `evaluate_analysis_by_app_piece_style.py`, `run_android_device_eval.py`
- レビュー補助: `make_visual_review.py`, `audit_position_label_inventory.py`, `compare_recognition_runs.py`

## データの流れ

1. スクリーンショットを `tools/samples/screenshots_by_app_piece_style` に整理する。
2. 対応ラベルを `tools/samples/labels/boards_by_app_piece_style` に置く。
3. `piece_style_manifest.csv` でサンプルと分類情報を管理する。
4. Pythonツールで解析、学習、評価を行う。
5. Android用JSON assetsを `app/src/main/assets` に書き出す。
6. Android側の認識パイプラインとデバイス評価で確認する。

## 主要assets

- `app_piece_templates.json`: アプリ/駒スタイル別の盤上駒テンプレートと位置事前分布。
- `known_position_samples.json`: 既知局面サンプル。
- `hand_piece_templates.json`: 持ち駒用テンプレート。
- `hand_digit_templates.json`: 持ち駒枚数数字テンプレート。
- `hand_layouts.json`: 持ち駒台レイアウト。

## ラベル形式

盤面ラベルは9x9です。左上から `9一`、右下が `1九` です。

- 空マス: `empty`
- 駒: `black:FU`, `white:RY` のような `color:piece`
- 持ち駒: `hands.black` / `hands.white` に `HI KA KI GI KE KY FU` の枚数

現在のドラフトラベルを無条件の正解として扱わないでください。特に新規追加分は認識結果由来の下書きを含むため、visual review と inventory audit を併用します。

## 駒コード

JSONラベルでは次の駒コードを使います。UIやユーザー向け表示では日本語を優先してください。

- `OU`: 玉
- `HI`: 飛
- `KA`: 角
- `KI`: 金
- `GI`: 銀
- `KE`: 桂
- `KY`: 香
- `FU`: 歩
- `RY`: 龍/竜
- `UM`: 馬
- `TO`: と
- `NY`: 成香
- `NK`: 成桂
- `NG`: 成銀

色は `black = 先手/下側`、`white = 後手/上側` です。座標は右上が `11`、左下が `99` です。

インベントリ監査では成駒を元駒に換算します。

- `RY -> HI`
- `UM -> KA`
- `NG -> GI`
- `NK -> KE`
- `NY -> KY`
- `TO -> FU`
