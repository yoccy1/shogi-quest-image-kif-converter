# shogi-gazo-desktop

将棋アプリのスクリーンショットから局面を認識し、JSON / SFEN / KIF形式で出力するデスクトップ向けPython CLIです。

v1ではCLIを中心に公開します。画像を選んで確認できるローカルHTML UIも同梱しています。ShogiVision連携は今後の任意機能として扱います。

## 対象

現時点の主な対象は次の3アプリのスクリーンショットです。

- 将棋ウォーズ
- 将棋クエスト
- ぴよ将棋

「100%」は、対応3アプリの評価セットに対するgateを通すことを目標にしています。未知のテーマ、低品質画像、トリミング不良、対象外アプリのスクリーンショットは自動確定せず、要確認として扱います。

### HTML UIの現在の対応範囲

別ユーザーが簡単に使うためのHTML画像解析UIは、現時点では次の範囲を主対象にしています。

- 対応したのは「将棋クエスト」の1枚駒、つまり一文字駒のスクリーンショットです。
- スマホでスクリーンショットを撮影した画像のみを対象にしています。
- 確認済みの機種は Pixel 7a のみです。

他端末、別アプリ、別テーマ、トリミング済み画像でも試すことはできますが、公開時点の保証対象ではありません。KIF/SFENを他アプリへ渡す前に、盤面と持ち駒をUI上で目視確認してください。

## インストール

```powershell
pip install .
```

ShogiVision連携を試す場合だけ、重い推論依存を追加します。

```powershell
pip install ".[shogivision]"
```

開発中にインストールせず実行する場合は、`PYTHONPATH`に`src`を通します。

```powershell
$env:PYTHONPATH = "src"
python -m shogi_gazo_desktop.cli --help
```

## HTML画像解析UIを使う

Windowsでは、リポジトリ直下の `start_kif_ui.cmd` をダブルクリックすると、ローカルサーバーを起動してブラウザでHTML UIを開きます。

```powershell
.\start_kif_ui.cmd
```

ブラウザが開いたら、将棋クエスト一文字駒のスマホスクリーンショットを選択するか、クリップボードにコピーされた画像を `Ctrl+V` で貼り付けます。認識スタイルは既定で `将棋クエスト 一文字` です。

UIでは次を確認できます。

- 元画像と認識結果の盤面を横に並べて確認
- 先手の持ち駒と後手の持ち駒を盤面の上下に表示
- マスや持ち駒をクリックして、元画像のどの位置と対応しているか確認
- KIF/SFENをコピーまたはダウンロード

初回だけ、Python環境とモデルが必要です。GitHub公開版には大きな学習済みモデルや第三者アプリのスクリーンショットを含めないため、`outputs\models\piece_model.pkl` にモデルを置いてから起動してください。開発用データを持っている場合は、下の「モデルを用意する」のコマンドで作成できます。

詳しい手順は [docs/HTML_UI_USAGE_JA.md](docs/HTML_UI_USAGE_JA.md) を参照してください。

## CLI

インストール後は`shogi-gazo`コマンドを使います。

```powershell
shogi-gazo --help
```

### モデルを用意する

GitHub公開版には、大きな学習済みモデルや第三者アプリのスクリーンショットは同梱しません。認識するには、別途入手したモデルを `--model` で指定するか、ラベル付きサンプルからモデルを作ります。

```powershell
shogi-gazo train-model --screenshots-dir path\to\screenshots --labels path\to\labels --out models\piece_model.pkl --include-hands
```

このリポジトリの開発用データを持っている場合は、次のように作れます。

```powershell
shogi-gazo train-model --screenshots-dir data\samples\screenshots_by_app_piece_style --labels data\samples\labels\boards_by_app_piece_style --out outputs\models\piece_model.pkl --include-hands
```

### 1枚を認識する

```powershell
shogi-gazo recognize path\to\screenshot.png --model models\piece_model.pkl --out outputs\sample_run --include-hands
```

結果JSONのパスと`needs_review`が出力されます。出力先は`outputs\sample_run\<画像名>\recognition.json`で、HTMLレビュー用に同じフォルダへ`piece_report.json`も保存します。未知セルや低信頼の候補が残る場合は、終了コード`3`で要確認を示します。

### ディレクトリを一括認識する

```powershell
shogi-gazo batch path\to\screenshots --model models\piece_model.pkl --out outputs\batch_run --include-hands
```

入力配下の`.png`、`.jpg`、`.jpeg`、`.webp`を再帰的に処理し、`manifest.json`を出力します。
改善中の代表サンプルだけを処理したい場合は、`--sample <画像名の拡張子なし>`を複数指定できます。重いno-leak検証では`--limit`も使えます。

```powershell
shogi-gazo batch data\samples\screenshots_by_app_piece_style --out outputs\noleak_probe --include-hands --no-leak --sample 将棋ウォーズ_二文字_通常_07 --sample ぴよ将棋_一文字駒_初期配置_01
```

### 認識結果をexportする

```powershell
shogi-gazo export outputs\sample_run\<画像名>\recognition.json --format kif --side-to-move black --out outputs\sample.kif
shogi-gazo export outputs\sample_run\<画像名>\recognition.json --format sfen --side-to-move black
shogi-gazo export outputs\sample_run\<画像名>\recognition.json --format json --out outputs\sample.normalized.json
```

KIF出力は局面KIF/BODです。1枚のスクリーンショットから指し手履歴を復元するものではありません。
未知セル、二歩、駒数超過、玉数不整合、未解決の盤面制約が残る場合は、誤った局面ファイルを避けるためexportを失敗させます。

### HTMLレビューを作る

```powershell
shogi-gazo review outputs\batch_run --labels data\samples\labels\boards_by_app_piece_style --html outputs\batch_run\visual_review.html --include-hands
```

認識結果、低信頼セル、ラベルとの比較を目視確認するためのHTMLを生成します。

### 評価セットで検証する

```powershell
shogi-gazo evaluate outputs\batch_run --labels data\samples\labels\boards_by_app_piece_style --include-hands --require-perfect
```

`--require-perfect` は、盤上駒、持ち駒、高信頼エラー、リーク検出がすべて0でない限り非ゼロ終了します。対応3アプリの100%目標はこのgateで管理します。
評価JSONには全体metricsだけでなく、サンプル別のエラー詳細も保存されます。調査中は`--sample`と`--limit`で対象を絞れます。

学習に使った同じ画像を評価する closed-set では、現在の開発セット86枚で100%を確認しています。未知画像への汎化確認には、`batch --no-leak` と `evaluate --strict-leak-guard` を使います。no-leak smokeではリーク0を確認していますが、holdout精度はまだ100%ではないため、公開時の保証は「対応3アプリの評価gateを通したモデル/データセット」に限定します。

現在のno-leak代表検証では、駒の有無は100%を維持していますが、ウォーズ系の成駒・香・金銀で識別エラーが残ります。未解決制約やunknownが残る局面はexportで止め、人間のレビュー対象にします。

### ラベルを検証する

```powershell
shogi-gazo validate-labels --labels data\samples\labels\boards_by_app_piece_style
```

ラベルJSONの形式と駒数インベントリを検証します。問題がある場合は終了コード`4`になります。

## 開発実行例

ローカル作業中は次のように`PYTHONPATH`を指定して、インストール前のソースを直接実行できます。

```powershell
$env:PYTHONPATH = "src"
python -m shogi_gazo_desktop.cli train-model --screenshots-dir data\samples\screenshots_by_app_piece_style --labels data\samples\labels\boards_by_app_piece_style --out outputs\models\piece_model.pkl --include-hands
python -m shogi_gazo_desktop.cli recognize data\samples\screenshots_by_app_piece_style\ぴよ将棋\一文字駒\ぴよ将棋_一文字駒_初期配置_01.png --out outputs\dev_sample --include-hands
python -m shogi_gazo_desktop.cli batch data\samples\screenshots_by_app_piece_style --out outputs\dev_batch --include-hands
python -m shogi_gazo_desktop.cli batch data\samples\screenshots_by_app_piece_style --out outputs\dev_noleak --include-hands --no-leak
python -m shogi_gazo_desktop.cli export outputs\dev_sample\<画像名>\recognition.json --format kif --out outputs\dev_sample.kif
python -m shogi_gazo_desktop.cli review outputs\dev_batch --labels data\samples\labels\boards_by_app_piece_style --include-hands
python -m shogi_gazo_desktop.cli evaluate outputs\dev_batch --labels data\samples\labels\boards_by_app_piece_style --include-hands --require-perfect
python -m shogi_gazo_desktop.cli evaluate outputs\dev_noleak --labels data\samples\labels\boards_by_app_piece_style --include-hands --strict-leak-guard
python -m shogi_gazo_desktop.cli validate-labels --labels data\samples\labels\boards_by_app_piece_style
python -m shogi_gazo_desktop.cli kif-ui --host 127.0.0.1 --port 8765 --out outputs\kif_ui
```

## 公開対象と除外物

公開リポジトリには、CLI本体、軽量なサンプル、ドキュメント、テストを中心に含めます。

次のディレクトリはローカル評価・調査用であり、公開配布から除外します。

- `data/samples/screenshots_by_app_piece_style/`
- `reports/`
- `outputs/`
- `third_party/ShogiVision/`

ShogiVisionは将来的な任意連携候補です。v1の必須依存ではなく、同梱もしません。

## 注意

- 認識結果に`needs_review`が立つ場合、その局面は人間の確認を前提にしてください。
- 未知セルが残る局面、駒数が不整合な局面、対象外UIのスクリーンショットはexportできない場合があります。
- KIFは局面の保存用です。通常の棋譜のような指し手履歴は生成しません。

## テスト

```powershell
pip install -e ".[dev]"
python -m py_compile src\shogi_gazo_desktop\cli.py src\shogi_gazo_desktop\recognition.py src\shogi_gazo_desktop\export.py
python -m pytest tests
```
