# HANDOFF 2026-05-10 ShogiVision Offline Probe

## Status

ShogiVisionを既存86枚ラベル付きスクリーンショットへ流すoffline probeを追加・実行した。

結論:

- ShogiVisionは起動/モデル読込/推論可能。
- ただし、現行Android認識を置き換える精度ではない。
- 特に `ぴよ将棋/ひよこ駒` では top1 identity 0.6718、top3 0.8252。
- 全86枚では top1 identity 0.5658、top3 0.7565。
- 一方で piece presence は全体 0.9978、empty accuracy は 0.9802 と高く、補助診断や候補生成には使える可能性がある。

## Files Changed

- `tools/shogivision_offline_probe.py`
  - ShogiVisionの `board_segmenter.pt` + `mixed.onnx` を使い、既存 `tools/samples/screenshots_by_app_piece_style` / `tools/samples/labels/boards_by_app_piece_style` を評価するoffline probe。
  - 出力:
    - `summary.json`
    - `shogivision_cell_predictions.csv`
    - `shogivision_board_summary.csv`
    - `shogivision_group_summary.csv`
    - `shogivision_errors.csv`

## How It Works

入力:

- screenshots: `tools/samples/screenshots_by_app_piece_style`
- labels: `tools/samples/labels/boards_by_app_piece_style`
- ShogiVision root: `ShogiVision-master/ShogiVision-master`

処理:

1. ShogiVisionのYOLO segmentationで盤面四隅を検出。
2. ShogiVisionの `BoardSplitter` で透視補正して81セルへ分割。
3. `mixed.onnx` で各セルの `Figure` 15分類 + `Direction` 3分類を推論。
4. labelの `orientation` に従って `Direction.UP/DOWN` を `black/white` へ変換。
5. `black:FU` のような既存ラベル形式へ変換し、top1/top3を評価。

## Commands Run

```powershell
C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master\venv\Scripts\python.exe -m py_compile tools\shogivision_offline_probe.py
```

Result: PASS

```powershell
C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master\venv\Scripts\python.exe tools\shogivision_offline_probe.py --app "ぴよ将棋" --style "ひよこ駒" --limit 1 --out-dir tools\out\shogivision_probe_smoke_20260510
```

Result:

- total 81
- true_piece 40
- errors 12
- top1_identity_accuracy 0.7000
- top3_identity_accuracy 0.8500
- empty_accuracy 1.0000
- piece_presence_accuracy 1.0000

```powershell
C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master\venv\Scripts\python.exe tools\shogivision_offline_probe.py --app "ぴよ将棋" --style "ひよこ駒" --out-dir tools\out\shogivision_probe_piyo_chick_all_20260510
```

Result:

- samples 16
- total 1296
- true_piece 515
- errors 203
- false_piece_on_empty 34
- false_empty_on_piece 1
- top1_identity_accuracy 0.6718
- top3_identity_accuracy 0.8252
- empty_accuracy 0.9565
- piece_presence_accuracy 0.9981

```powershell
C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master\venv\Scripts\python.exe tools\shogivision_offline_probe.py --out-dir tools\out\shogivision_probe_all_86_20260510
```

Result:

- samples 86
- total 6963
- true_piece 2768
- true_empty 4195
- errors 1285
- false_piece_on_empty 83
- false_empty_on_piece 6
- top1_identity_accuracy 0.5658
- top3_identity_accuracy 0.7565
- empty_accuracy 0.9802
- piece_presence_accuracy 0.9978

## Main Output

- `tools/out/shogivision_probe_all_86_20260510/summary.json`
- `tools/out/shogivision_probe_all_86_20260510/shogivision_cell_predictions.csv`
- `tools/out/shogivision_probe_all_86_20260510/shogivision_board_summary.csv`
- `tools/out/shogivision_probe_all_86_20260510/shogivision_group_summary.csv`
- `tools/out/shogivision_probe_all_86_20260510/shogivision_errors.csv`

## Group Summary

| app | style | samples | errors | top1 | top3 | empty | presence |
|---|---|---:|---:|---:|---:|---:|---:|
| `ぴよ将棋` | `昇竜` | 1 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| `将棋クエスト` | `書籍風` | 1 | 3 | 0.9250 | 0.9750 | 1.0000 | 1.0000 |
| `将棋クエスト` | `クラシック二文字駒` | 11 | 42 | 0.8754 | 0.9674 | 1.0000 | 1.0000 |
| `ぴよ将棋` | `二文字駒` | 1 | 5 | 0.8750 | 0.9750 | 1.0000 | 1.0000 |
| `ぴよ将棋` | `太文字駒` | 1 | 5 | 0.8750 | 1.0000 | 1.0000 | 1.0000 |
| `ぴよ将棋` | `一文字駒` | 1 | 10 | 0.7500 | 0.9000 | 1.0000 | 1.0000 |
| `ぴよ将棋` | `ひよこ駒` | 16 | 203 | 0.6718 | 0.8252 | 0.9565 | 0.9981 |
| `将棋クエスト` | `一文字駒` | 30 | 451 | 0.5231 | 0.7440 | 0.9888 | 0.9945 |
| `ぴよ将棋` | `昇竜一文字` | 1 | 22 | 0.4500 | 0.7500 | 1.0000 | 1.0000 |
| `将棋ウォーズ` | `二文字` | 8 | 179 | 0.4093 | 0.7181 | 0.9332 | 1.0000 |
| `ぴよ将棋` | `風波一文字` | 1 | 24 | 0.4000 | 0.6500 | 1.0000 | 1.0000 |
| `将棋ウォーズ` | `一文字` | 14 | 341 | 0.2827 | 0.4925 | 0.9910 | 1.0000 |

## Interpretation

ShogiVision単体は、デジタルスクショ全体の汎用置換には向かない。

有望な使い方:

- 駒あり/空マス判定の補助。
- app/style限定の補助候補生成。
- top3候補を既存template診断と組み合わせる。
- 特に `将棋クエスト/クラシック二文字駒` では候補として強い。

現時点で避けるべきこと:

- 現行Android recognizerをShogiVisionへ置換しない。
- `ぴよ将棋/ひよこ駒` の精度改善として採用しない。
- ShogiVisionのoffline metricをAndroid実機改善として報告しない。
- Android assetsへモデルを入れない。

## Notes

- `tools/shogivision_offline_probe.py` はShogiVision venvのPythonで実行する前提。
- Windowsでは `PyQt5` import前に `torch` を読む必要があるため、script内でも `torch` を先にimportしている。
- `ultralytics` のライセンス表示は `AGPL-3.0`。アプリ同梱前にライセンス確認が必要。

## Next

次に進めるなら、ShogiVisionをproduction実装へ入れるのではなく、以下のoffline診断が安全。

1. `shogivision_cell_predictions.csv` とB5-29 `piece_style_board_error_candidates.csv` をjoinする。
2. 残12件について、ShogiVision top3が正解を含むか、既存template候補を補えるか確認する。
3. `piece_presence_accuracy` が高いことを利用し、空/駒あり判定だけ補助できるかを見る。
4. app/style別にShogiVisionを使う価値がある箇所と、使ってはいけない箇所を明確にする。

Production implementation: NOT APPROVED.
