# HANDOFF 2026-05-11 piece accuracy HTML lightweight

## 概要

前回作成した全86件HTML `tools/out/piece_accuracy_review_current_full86_20260510/index.html` が重かったため、軽量版を作成した。

## 変更ファイル

- `tools/make_piece_accuracy_review.py`
  - `--split-pages` オプションを追加。
  - 通常モードは従来通り、全サンプルを1つの `index.html` に出す。
  - `--split-pages` では、軽量な一覧 `index.html` と、サンプル別詳細ページ `samples/s001.html` から `samples/s086.html` を出す。

## 生成物

- 軽量版:
  - `tools/out/piece_accuracy_review_current_full86_20260510_light/index.html`
  - `tools/out/piece_accuracy_review_current_full86_20260510_light/samples/s001.html` から `s086.html`
  - `tools/out/piece_accuracy_review_current_full86_20260510_light/piece_accuracy_samples.csv`
  - `tools/out/piece_accuracy_review_current_full86_20260510_light/piece_accuracy_cells.csv`
  - `tools/out/piece_accuracy_review_current_full86_20260510_light/piece_accuracy_summary.json`

## 実行コマンド

```powershell
python -m py_compile tools\make_piece_accuracy_review.py

python tools\make_piece_accuracy_review.py `
  tools\out\android_device_eval\android_eval_current_full86_noleak_visual_20260510 `
  --out-dir tools\out\piece_accuracy_review_current_full86_20260510_light `
  --split-pages
```

## 検証結果

- 旧HTML:
  - `tools/out/piece_accuracy_review_current_full86_20260510/index.html`
  - 2,339,745 bytes / 約2,284.9 KB
- 軽量HTML:
  - `tools/out/piece_accuracy_review_current_full86_20260510_light/index.html`
  - 64,405 bytes / 約62.9 KB
- 削減率:
  - 約97.2%

軽量indexの静的確認:

- sample rows: 86
- full board DOM: 0
- image tags: 0
- filter toolbar: あり

詳細ページの静的確認:

- `samples/s001.html` は board 1 / image 1 / back linkあり。
- `samples/*.html`: 86件。
- `piece_accuracy_samples.csv`: 86行。
- `piece_accuracy_cells.csv`: 6966行。

## 注意

- in-app browserで軽量indexを開く確認も試したが、ブラウザ接続側がタイムアウトしたため、最終確認は静的検証で行った。
- 旧all-in-one HTMLは残している。ユーザーには軽量版 `tools/out/piece_accuracy_review_current_full86_20260510_light/index.html` を案内する。
- 詳細ページは `samples/s001.html` のような連番で、軽量indexの `Detail` 列から開ける。
