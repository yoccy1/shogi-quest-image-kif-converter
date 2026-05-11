# Development And QA

PowerShellで `C:\Users\sakas\AndroidProjects\shogi_gazo` から実行する前提です。

## Android環境

必要なもの:

- Android Studio
- JDK 17以上
- Android SDK Platform 36
- Android SDK Build Tools 36.x

この環境ではAndroid Studio同梱JBRを使うことがあります。

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
```

## 基本コマンド

```powershell
.\gradlew.bat test
.\gradlew.bat assembleDebug
.\gradlew.bat :app:testDebugUnitTest --no-daemon
.\gradlew.bat :app:lintDebug --no-daemon
.\gradlew.bat :app:assembleDebug --no-daemon
```

端末/エミュレータ確認:

```powershell
$adb="$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
& $adb devices -l
```

インストールや計測前に、接続端末が `device` 状態であることを確認してください。過去には `emulator-5554` が使われていましたが、毎回確認してください。

## Python環境

```powershell
python -m venv .venv
.venv\Scripts\pip install -r tools\requirements.txt
```

構文チェック:

```powershell
python -m py_compile tools\run_android_device_eval.py tools\evaluate_analysis_by_app_piece_style.py tools\evaluate_piece_recognition.py
```

## 代表的なローカル解析

```powershell
python tools\analyze_shogi_screenshot.py tools\samples\screenshots\sample_001.png --out tools\out\analysis
python tools\run_analysis_by_app_piece_style.py --include-hands --low-confidence 0.55
python tools\make_visual_review.py tools\out\analysis_by_app_piece_style --labels-dir tools\samples\labels\boards_by_app_piece_style --include-hands
```

ラベル確認:

```powershell
python tools\validate_position_labels.py tools\samples\labels\boards_by_app_piece_style --require-hands
python tools\audit_position_label_inventory.py tools\samples\labels\boards_by_app_piece_style --analysis-dir tools\out\analysis_by_app_piece_style
```

glyph系ラベル作業の詳細な現状は `AI_CONTEXT/handoffs/HANDOFF_2026-05-09_labeling.md` を確認してください。新規56枚にはドラフトラベルが含まれるため、ラベルを教師データとして使う前に visual review と inventory audit を行います。

asset生成:

```powershell
python tools\train_piece_model.py --labels-dir tools\samples\labels\boards_by_app_piece_style --screenshots-dir tools\samples\screenshots_by_app_piece_style --calibration-dir tools\samples\screenshots_by_app_piece_style --include-hands --out tools\out\models\piece_model.pkl
python tools\export_android_piece_templates.py tools\out\models\piece_model.pkl
python tools\export_android_hand_assets.py
python tools\export_known_position_samples.py
```

## Androidデバイス評価

非リーク smoke 評価の例です。サンプル数やserialは状況に応じて調整してください。

```powershell
python tools\run_android_device_eval.py `
  --run-id android_eval_noleak_smoke_20260509 `
  --sample-dir tools\samples\screenshots_by_app_piece_style `
  --labels-dir tools\samples\labels\boards_by_app_piece_style `
  --limit 3 `
  --no-known-samples `
  --no-board-ocr `
  --strict-leak-guard `
  --max-seconds 6 `
  --require-perfect `
  --require-speed
```

2026-05-09時点では、この非リーク評価は失敗する可能性が高いです。その失敗は認識改善のための有用なシグナルなので、隠さず記録してください。

通常QAでは `--allow-missing-excluded-source` を使わないでください。古いreportを再評価するために必要な場合だけ明示し、その理由をhandoffへ書きます。`--strict-leak-guard` はleak検出時に単独でも非ゼロ終了する条件として扱います。

既存の取得済みrunを再評価する例:

```powershell
python tools\run_android_device_eval.py `
  --run-id android_eval_qa_noleak_smoke_20260509 `
  --skip-build --skip-install --skip-push --skip-instrumentation --skip-pull `
  --no-known-samples `
  --no-board-ocr `
  --strict-leak-guard `
  --max-seconds 6 `
  --require-perfect `
  --require-speed
```

## 変更後の確認目安

## 必須QAルール

作業したAIは、終了前に必ずユニットテストと担当範囲に応じたQAを実行してください。これは任意ではありません。

- Android/Kotlinを触った場合は、最低でも `:app:testDebugUnitTest` を実行する。
- Android本体、UI、認識、export、reportを触った場合は、原則 `:app:lintDebug` と `:app:assembleDebug` も実行する。
- Python toolを触った場合は、対象ファイルの `python -m py_compile ...` と関連テスト/小実行を行う。
- 評価/認識の変更では、可能な範囲で no-leak smoke または既存run再評価を行い、失敗metricsも隠さず記録する。
- テストやQAが失敗した場合は、失敗ログの要点、再現コマンド、未解決理由をhandoffに残す。
- 時間や端末都合で未実行のQAがある場合も、「未実行」と理由をhandoffに明記する。

Kotlin/Gradle incremental cacheの一時不整合が疑われる場合は、次で再確認します。

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
.\gradlew.bat :app:testDebugUnitTest --rerun-tasks --no-build-cache --no-daemon
```

- Kotlin/Android認識変更: `:app:testDebugUnitTest`, `:app:lintDebug`, `:app:assembleDebug`
- Python評価ツール変更: `python -m py_compile ...` と対象ツールの小さな実行
- assets生成変更: 関連unit test、Androidデバイス評価、visual review
- サンプル/ラベル変更: validate、audit、visual review

結果は作業終了時の引き継ぎmdに必ず残してください。

## レビュー資料

詳細な修正候補は次に集約済みです。

- `AI_CONTEXT/reviews/APP_REVIEW_REPORT_2026-05-09.md`
- `AI_CONTEXT/reviews/TOOLS_REVIEW_REPORT_2026-05-09.md`

優先順位の目安:

- Android側は、解析Job世代管理、手動編集保護、認識fallback、template補完、domain validator、release/privacyを先に見る。
- tools側は、no-leak評価、app/style/sampleキーでのラベル解決、安全なclean、CI gate、再現性metadata、手駒datasetの人手承認を先に見る。
