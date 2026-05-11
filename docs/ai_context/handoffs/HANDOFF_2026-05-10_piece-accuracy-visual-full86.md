# HANDOFF 2026-05-10 piece accuracy visual full86

## 概要

現在の駒特定精度を、全86サンプル画像ごとに目視確認できるHTML/CSV/JSONとして出力した。

今回はユーザー指示により複数サブエージェントを使用し、以下を並行調査した。

- Android実機評価出力の構造: 最新2026-05-10系は3件のみ、全86件の新規runが必要。
- 既存Python評価/ラベル構造: `evaluate_piece_recognition.py` と `position_label_utils.py` を再利用。既存 `make_visual_review.py` は3階層の全86件rootを拾えないため専用ツールが必要。
- ShogiVision出力: Android結果とjoin可能だが、主指標ではなく参考候補欄向き。今回のHTML主指標には入れていない。

## 変更/追加ファイル

- `tools/make_piece_accuracy_review.py`
  - `manifest.csv` または `app/style/sample/piece_report.json` から全サンプルを列挙。
  - `boards_by_app_piece_style` ラベルと照合し、サンプル画像、予測盤面、expected、board/hand/leak errorsをHTML化。
  - 併せて `piece_accuracy_samples.csv`, `piece_accuracy_cells.csv`, `piece_accuracy_summary.json` を出力。

## 生成物

- 現行全86評価run:
  - `tools/out/android_device_eval/android_eval_current_full86_noleak_visual_20260510`
- 可視化出力:
  - `tools/out/piece_accuracy_review_current_full86_20260510/index.html`
  - `tools/out/piece_accuracy_review_current_full86_20260510/piece_accuracy_samples.csv`
  - `tools/out/piece_accuracy_review_current_full86_20260510/piece_accuracy_cells.csv`
  - `tools/out/piece_accuracy_review_current_full86_20260510/piece_accuracy_summary.json`
- 検証用出力:
  - `tools/out/piece_accuracy_review_b5_29_smoke_20260510`
  - `tools/out/piece_accuracy_review_full86_legacy_20260509`

## 実行コマンド

```powershell
python -m py_compile tools\make_piece_accuracy_review.py

python tools\make_piece_accuracy_review.py `
  tools\out\android_device_eval\android_eval_b5_29_clean_component_diagnostics_20260510 `
  --out-dir tools\out\piece_accuracy_review_b5_29_smoke_20260510

python tools\make_piece_accuracy_review.py `
  tools\out\android_device_eval\android_eval_full86_20260509_final `
  --out-dir tools\out\piece_accuracy_review_full86_legacy_20260509 `
  --no-strict-leak-guard

python tools\run_android_device_eval.py `
  --run-id android_eval_current_full86_noleak_visual_20260510 `
  --no-known-samples `
  --strict-leak-guard `
  --collect-candidate-diagnostics

python tools\make_piece_accuracy_review.py `
  tools\out\android_device_eval\android_eval_current_full86_noleak_visual_20260510 `
  --out-dir tools\out\piece_accuracy_review_current_full86_20260510
```

## 端末状態

- `adb` はPATH上になかったため、`%LOCALAPPDATA%\Android\Sdk\platform-tools\adb.exe` を使用。
- 物理端末 `39181JEHN02798` は `unauthorized`。
- 評価は `emulator-5554` で実行した。
- そのため速度指標は実機性能ではなくエミュレータ参考値として扱うこと。

## 現行全86評価結果

`android_eval_current_full86_noleak_visual_20260510` は86件すべて評価済み。ただしstrict leak guardで失敗終了した。評価条件は緩めず、metricsをそのまま記録する。

- `evaluated_samples=86`
- `skipped_samples=0`
- `true_piece=2768`
- `true_empty=4195`
- `empty_accuracy=1.0`
- `piece_presence_accuracy=1.0`
- `confirmed_identity_accuracy=0.5087`
- `top1_identity_accuracy=0.5538`
- `top3_contains_identity_accuracy=0.673`
- `errors=1235`
- `high_confidence_errors=518`
- `unknown_on_piece=963`
- `hand_errors=356`
- `leak_errors=273`
- `over_limit_count=86`
- `max_observed_seconds=31.542`

HTML/CSV側のセルstatus集計:

- `ok-empty=4195`
- `ok-piece=1408`
- `unknown=597`
- `unknown-top3=366`
- `wrong=302`
- `top3-only=88`
- `top1-only=7`
- `ignored=3`

## leak_errorsの内訳

leakは主に初期配置テンプレート由来。

- `ぴよ将棋 / 太文字駒 / ぴよ将棋_太文字駒_初期配置_01`: 172
- `将棋クエスト / 書籍風 / 将棋クエスト_書籍風_初期配置_01`: 78
- `ぴよ将棋 / 風波一文字 / ぴよ将棋_風波一文字_初期配置_01`: 20
- `ぴよ将棋 / 昇竜一文字 / ぴよ将棋_昇竜一文字_初期配置_01`: 2
- `ぴよ将棋 / 昇竜 / ぴよ将棋_昇竜_初期配置_01`: 1

代表例:

```text
$.cells[0].candidates[1].source leaks forbidden source ぴよ将棋_太文字駒_初期配置_01:
app_template:initial:ぴよ将棋_太文字駒_初期配置_01
```

## HTML検証

ブラウザでの直接確認は、全画像入りHTMLが大きく in-app browser の初回読み込みでタイムアウトした。代わりにファイル内容とCSVを静的確認した。

- `index.html`: 86 sample sections / 86 boards / filter toolbarあり
- `piece_accuracy_samples.csv`: 86 rows
- `piece_accuracy_cells.csv`: 6966 rows
- CSV上の画像パス missing: 0
- B5-29 3件smokeでは既存summaryと一致:
  - `errors=12`
  - `hand_errors=0`
  - `leak_errors=0`
  - `confirmed_identity_accuracy=0.8839`
  - `top1_identity_accuracy=0.8929`

## 次にやるべきこと

1. `tools/out/piece_accuracy_review_current_full86_20260510/index.html` を開き、まず `Board errors only` と `Hand errors only` で悪いサンプルを目視確認する。
2. 純粋なno-leak全86精度が必要なら、次は以下のrunを回す。

```powershell
python tools\run_android_device_eval.py `
  --run-id android_eval_current_full86_strict_no_initial_no_ocr_20260510 `
  --no-known-samples `
  --no-initial-position-pattern `
  --no-board-ocr `
  --strict-leak-guard `
  --collect-candidate-diagnostics
```

3. ShogiVisionを参考欄に出す場合は、`tools/out/shogivision_probe_all_86_20260510/shogivision_cell_predictions.csv` を `sample,row,col` でjoinする。主指標にはしない。
4. 駒認識改善の主作業は、HTMLでworst samplesを確認し、`wrong` / `unknown-top3` / `top3-only` を分けて改善仮説を作ること。

## 注意

- `android_eval_full86_20260509_final` は全86件でperfectに見えるが、既知サンプル使用のため現状把握には甘い。
- 今回の `android_eval_current_full86_noleak_visual_20260510` は `--no-known-samples` だが `useInitialPositionPattern=true` なので、初期配置系の一部でstrict leak guardに引っかかる。
- speedはエミュレータ参考値。Pixel 7a等の物理端末がauthorizedになったら再実行する。
