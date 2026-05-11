# AI_CONTEXT

このフォルダーは、新しく作業するAIが最初に読むための情報置き場です。プロジェクト全体の入口、開発/QA手順、構成マップ、作業終了時の引き継ぎルールをここに集約します。

最終更新: 2026-05-10

## 最初に読む順番

1. `AI_CONTEXT/README.md`
2. `AI_CONTEXT/PROJECT_MAP.md`
3. `AI_CONTEXT/DEVELOPMENT_AND_QA.md`
4. 複数AIで分担する場合は `AI_CONTEXT/WORK_SPLIT.md`
5. `AI_CONTEXT/HANDOFF_TEMPLATE.md`
6. 最新の `AI_CONTEXT/handoffs/HANDOFF_*.md`
7. 必要に応じて `AI_CONTEXT/reviews/*.md`
8. `README.md`
9. `docs/SETUP_AND_USAGE_JA.md`

AI向けの正本はこの `AI_CONTEXT` 配下です。`docs/AI_HANDOFF_2026-05-09.md`、`docs/APP_REVIEW_REPORT_2026-05-09.md`、`docs/TOOLS_REVIEW_REPORT_2026-05-09.md`、`tools/AI_HANDOFF.md` の内容は、次のファイルへ集約済みです。

- `AI_CONTEXT/handoffs/HANDOFF_2026-05-09_recognition-qa.md`
- `AI_CONTEXT/handoffs/HANDOFF_2026-05-09_labeling.md`
- `AI_CONTEXT/reviews/APP_REVIEW_REPORT_2026-05-09.md`
- `AI_CONTEXT/reviews/TOOLS_REVIEW_REPORT_2026-05-09.md`

作業前に必ず次を確認してください。

```powershell
git status --short
```

このワークツリーは、認識系Kotlin、評価Python、assets、サンプル画像/ラベルに未整理の変更が多くあります。自分が作った変更ではないものを戻したり削除したりしないでください。

複数AIでA/Bを細分化して作業する場合は、`AI_CONTEXT/WORK_SPLIT.md` を使って担当範囲を決めてください。既に動いているA/Bには、まず `a-current` / `b-current` のhandoffで現在地を宣言させます。

## プロジェクトの要点

- Androidアプリ `Shogi Gazo` は、将棋アプリのスクリーンショットから局面を認識し、手動補正後にKIF/BOD形式の局面KIFとSFENを出力するMVPです。
- Kotlin + Jetpack Compose の単一 `app` モジュールです。
- 画像入力はPhoto PickerまたはAndroid共有Intentです。
- 対象は主に `将棋ウォーズ`、`将棋クエスト`、`ぴよ将棋` のスクリーンショットです。
- 単一スクリーンショットから指し手履歴は復元できません。出力KIFは棋譜ではなく局面KIF/BODです。
- 現在の重点は、既知サンプル一致ではなく、未知スクリーンショットに対する認識精度とQAです。

## 現在の重要課題

最新の評価/QA/データ状態は `AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-ai2-cell-crop-strict-split-qa.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai2-qa-cell-crop-classifier-strict-split-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-ai1-cell-crop-strict-split-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai1-cell-crop-classifier-strict-split-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-ai2-cell-crop-classifier-qa.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai2-qa-cell-crop-classifier-offline-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-ai1-cell-crop-classifier-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai1-cell-crop-classifier-offline-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-recognition-accuracy-research.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_recognition-accuracy-research-boardgames.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-recognition-accuracy-research-prompt.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-real-device-app-check-plan.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-ai1-mask-variant-cell-audit.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai1-mask-variant-cell-audit.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-two-ai-next-plan.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-ai2-mask-variant-qa.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai2-qa-mask-variant-offline-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-after-ai1-mask-variant-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai1-mask-variant-offline-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai2-qa-multitemplate-consensus-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai2-qa-chamfer-offline-probe.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_ai1-clean-component-diagnostics.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-10_b5-27-gated-shape-diagnostics.md` を確認してください。直近の採用済みbaselineは `tools/out/android_device_eval/android_eval_b5_27_gated_shape_dimension_diagnostics_retry2_20260510`、最新診断追加runは `tools/out/android_device_eval/android_eval_b5_29_clean_component_diagnostics_20260510` です。

2026-05-10時点の最新評価:

- 前回の全86件HTMLが重かったため、`tools/make_piece_accuracy_review.py` に `--split-pages` を追加し、軽量版 `tools/out/piece_accuracy_review_current_full86_20260510_light/index.html` を作成済み。軽量indexは画像/盤面DOMを持たず、サンプル別詳細は `samples/s001.html` から `samples/s086.html` に分割。旧index 2,339,745 bytesから軽量index 64,405 bytesへ約97.2%削減。詳細は `AI_CONTEXT/handoffs/HANDOFF_2026-05-11_piece-accuracy-html-lightweight.md`。
- 現在の駒特定精度を全86サンプル画像ごとに確認するHTMLを作成済み。ツールは `tools/make_piece_accuracy_review.py`、成果物は `tools/out/piece_accuracy_review_current_full86_20260510/index.html`。同時に `piece_accuracy_samples.csv` / `piece_accuracy_cells.csv` / `piece_accuracy_summary.json` を出力。今回の現行全86run `tools/out/android_device_eval/android_eval_current_full86_noleak_visual_20260510` は `--no-known-samples --strict-leak-guard --collect-candidate-diagnostics` で実行し、`evaluated_samples=86`, `skipped=0`, `errors=1235`, `hand_errors=356`, `leak_errors=273`, `confirmed_identity_accuracy=0.5087`, `top1_identity_accuracy=0.5538`, `top3_contains_identity_accuracy=0.673`。leakは主に初期配置テンプレート `app_template:initial:<sample>` 由来。物理端末はunauthorizedだったため `emulator-5554` で実行、速度値は参考扱い。詳細は `AI_CONTEXT/handoffs/HANDOFF_2026-05-10_piece-accuracy-visual-full86.md`。
- ShogiVision offline probeを追加・実行済み。`tools/shogivision_offline_probe.py` で既存86枚へ `board_segmenter.pt` + `mixed.onnx` を適用し、全体top1 identity 0.5658、top3 0.7565、empty 0.9802、piece presence 0.9978。`ぴよ将棋/ひよこ駒` はtop1 0.6718、top3 0.8252。現行Android認識の置換は不可。次はB5-29候補CSVとShogiVision top3をjoinし、補助候補やpresence限定で使えるかoffline監査する。
- ShogiVisionはローカルセットアップ済み。`C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master` に専用 `venv`、`models`、`run_shogivision.ps1` を作成。Google Drive Modelsから `board_segmenter.pt`, `board_segmenter.onnx`, `mixed.onnx`, `figure_classifier.pt`, `direction_classifier.pt` を取得し、`MAIN_IMPORT_OK` / `FACTORIES_OK` / `GUI_INSTANTIATE_OK` を確認済み。詳細は `AI_CONTEXT/handoffs/HANDOFF_2026-05-10_shogivision-local-setup.md`。
- ユーザーが `C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master` にShogiVisionを追加。実体は `ShogiVision-master\ShogiVision-master`。利用前に `AI_CONTEXT/handoffs/HANDOFF_2026-05-10_coordinator-shogivision-requirements-prompt.md` に従い、Python/venv/依存、モデル取得、GPU要否、モデルサイズ、Android TFLite/ONNX化、評価計画を調査する。zip内にモデル本体とLICENSEは見当たらないため、production実装やassets追加はまだ禁止。
- ユーザー判断により、現時点の最優先課題は駒認識精度の抜本改善。実機完成度確認よりも、画像認識/OCR/クラス分類/他ボードゲーム事例の調査から次の認識改善実験を設計することを優先する。
- 調査担当AIのboardgames/OCR/classifier調査では、次の最優先実験として no-leak cell-crop HOG/linear classifier offline baseline を選定。AI-1は既存86 screenshots/labelsからderived crop datasetを作り、occupied cellの `color:piece` 多クラス分類をGroup split/leave-one-source-outで検証する。production認識ロジックはまだ変更しない。
- AI-1 cell-crop classifier offline probeはAI-2 QAで再現済み。2768 occupied cell crops / 86 sources / 28 classes、linear SVM LOSO top1 0.9332、top3 0.9751、no-leak failures 0。B5-29 residual 12件はvirtual top1で全修正。ただしstable baseline-correct cell degradationが1件（`ぴよ将棋_ひよこ駒_通常_01 8九 black:KE -> black:HI`, expected rank 2）あり、production未承認。次はより厳しいsplitで汎化確認。
- AI-1 strict split probeはAI-2 QAで再現済み。linear_svmはpiyo_chick strict foldsでresidual 11/12・stable degradation 6-7。logisticは `leave_app_style_out` / `leave_piyo_chick_normal_out` / `leave_piyo_chick_out` でresidual 12/12、top1約0.91-0.92、top3約0.96、stable degradation 2。ただしconfidence-only gateではaccepted errorsが残り、production gate candidate 0。production未承認。次はAI-1がB5-29 baseline/template診断とlogistic出力を結合し、安全gate候補とstable degradation 2件のroot-causeをoffline監査する。
- B5-27採用状態で `ぴよ将棋/ひよこ駒` 3サンプル 診断ON strict no-leak smoke は `errors=12`, `unknown_on_piece=13`, `confirmed_identity_accuracy=0.8839`, `top1_identity_accuracy=0.8929`, `max_observed_seconds=5.311`, `over_limit_count=0`。B5-25は `piyo_chick` display 8八限定のhome rook tie-breakで `通常_02 8八 black:HI` をtop1へ戻した。B5-27は認識score/rank/confirmを変えず、低信頼/小marginセルの `diagnostic_candidates` 限定でsource/target bbox・center・ink ratio/count診断を追加した。確定条件、global threshold / global margin は緩めていない。
- B5-28ではKY吸着限定で unpositioned `white:KY` full-bbox penaltyをprobeしたが、KY topは消えても `errors=12` 維持、`unknown_on_piece=14` に悪化したため不採用。probe差分は撤回済みでB5-27 baselineを維持。現在は `WAIT_QA`。
- B5-29ではscore/rank/confirmを変えず、`piyo_chick` の `diagnostic_candidates` にclean component / component-pruned glyph bbox / ink count系フィールドを追加。strict診断ON runは `errors=12`, `unknown_on_piece=13`, `confirmed_identity_accuracy=0.8839`, `top1_identity_accuracy=0.8929`, `max_observed_seconds=5.051`, `over_limit_count=0`。KYを安全に下げる単純根拠はまだ不足し、source側の赤/edge/広域glyphが1 componentに結合していることを確認。現在は `WAIT_QA`。
- B5-30相当のoffline chamfer probeではproduction認識ロジックを変えず、B5-29/B5-27診断CSVの残12件/336候補へ `topN=30` symmetric chamfer / distance-transform virtual rerankを実施。AI-2 QAで再現済み。正解は全12件topN内だが `fixed_by_virtual_count=0`, `virtual_errors_if_top1=12`、期待rank平均は `18.0833 -> 19.6667` に悪化したため、現clean source maskのままproduction化しない。`REJECTED_PROBE` として固定。
- identity-level multi-template consensus / source diversity offline診断では、B5-29を固定しstrict no-leak excluded sourceを反映してper-template再採点を実施。AI-2 QAでAI-1出力とSHA256一致、`template_score_rows=1530`, `candidate_rows=336`, `fixed_by_top3_mean_count=0`, `fixed_by_support_bonus_count=0`, `fixed_by_source_diversity_count=0` を再現。期待rank平均も `18.0833 -> 18.9167/20.5/18.9167` と概ね悪化したため、`REJECTED_PROBE` として固定しproduction化しない。
- source mask variant offline診断では、B5-29を固定し残12件/topN=30に `current_clean`, `red_pruned`, `edge_band_pruned`, `red_edge_pruned`, `interior_only`, `skeleton_like` を適用。AI-2 QAでAI-1出力とSHA256一致、summary rows 6 / cell summary rows 72 / template score rows 8316 / shape-only template score rows 9180 / leak 0を再現。red/interior/skeletonでKY/HI/GI吸着は弱まるが、default `base+variant` 修正は最大1件、shape-only base+修正は0件。`PASS_QA` だが production は未承認。
- mask variant改善セルのcell auditでは、6セルに絞って `current_clean` / `interior_only` / `skeleton_like` のexpected vs competitorをoffline監査。出力は `tools/out/android_device_eval/android_eval_b5_29_clean_component_diagnostics_20260510/mask_variant_cell_audit/`。identity rows 180 / pair deltas 90 / cell summary 6 / no-leak audit 18、leak/empty countsはすべて0。`black:OU` 2件は `interior_only` variant-onlyでtop1化するが `base+variant` で `black:GI` へ戻る。`white:KA 6六` は `skeleton_like` でbase+もtop1化するがsource count 56まで落ちる破壊的mask。`white:KI` 3件はrank改善のみ。production化せず、AI-2 QA待ち。
- `hand_errors=0`, `leak_errors=0`, `skipped_samples=0`, `high_confidence_errors=0`, `over_limit_count=0` は維持。
- `--strict-leak-guard` は leak を単独でも失敗扱いにする。古いreportの `model.excluded_source` 欠落を許す場合は `--allow-missing-excluded-source` を明示する必要がある。
- 主セット `tools/samples/screenshots_by_app_piece_style` / `tools/samples/labels/boards_by_app_piece_style` は 86 images / 86 labels、validate/inventory OK。`piece_style_manifest.csv` の stale `label_status` 9件は残る。
- glyph/new56は missing label 0 / skipped 0 だが、new56 manifest stale 56件、初期配置明示labelなし12件、`将棋クエスト_二文字_通常_31` inventory不整合が残る。目視確認までlabel修正禁止。

過去の経緯は `AI_CONTEXT/handoffs/HANDOFF_2026-05-09_android-recognition.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-09_b-integration-review.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-09_integration-result.md` も確認してください。

現在の認識側の次タスク:

- ShogiVisionはproduction置換せず、`tools/out/shogivision_probe_all_86_20260510/shogivision_cell_predictions.csv` とB5-29診断CSVをjoinして、既存templateの補助候補として使えるかだけをofflineで監査する。
- ShogiVisionは、まず3担当（環境/モデル取得、モデルサイズ/GPU/runtime、統合/評価計画）で要件調査を行う。大容量モデル/datasetのダウンロード、Android assets追加、production実装は統括判断まで行わない。
- 次の最優先はAI-1による logistic cell-crop classifier combined gate audit。B5-29 baseline/template診断とAI-2再現済みlogistic strict split出力をcell単位で結合し、accepted errors/stable degradationを出さずにresidualを直せるgate条件があるかをofflineで検証する。production実装はまだしない。
- Piyo bottom hand area の右端FU component検出を改善する。
- `HandCountAggregator` 前に落ちた raw hand candidates をdebug reportへ出す。
- board recognitionはposition priorを広く弱めず、`ぴよ将棋/ひよこ駒` の低信頼/小marginセルに残る `debug_candidates` / 明示ONの `diagnostic_candidates` と raw/clean/red-location/bbox/ink breakdownを使って、正解identityがtop10外へ落ちる候補生成/shape特徴を改善する。`通常_02 8二 black:UM` はB5-4で改善済み。B5-8ではcurated `BLACK GI 通常_02 r6c6` / `BLACK GI 通常_03 r8c4` のrepackで `通常_02 6八 black:GI` のtop1を正解へ戻し、`errors=15 -> 14`。B5-16では `white:HI` vs `white:GI` の狭いshape/bbox tie-breakで `通常_01 7五 white:GI` をtop1正解へ戻し、`errors=14 -> 13`。B5-17では同cellだけをshape/red profile guardで確定し、`unknown_on_piece=15 -> 14`。B5-18では `通常_02 6八 black:GI` だけを極狭blackGI shape/red profile guardで確定し、`unknown_on_piece=14 -> 13`。B5-19ではscore変更なしでposition boost分解diagnosticsを追加。B5-20ではboard errorごとの全診断候補CSVを追加。B5-21ではtop vs expected gapとgap summary CSVを追加。B5-22ではtemplate supply CSVを追加し、残13件すべてで正解identityの非leak template供給は存在するが、既存assetの並べ替え/repackだけでは改善根拠がないことを確認。B5-23では位置加点前/weight前の推定score診断を追加。B5-24では exact `base_template_score` を追加し、B5-23の推定が残13件では実採点値と実質一致していたこと、`通常_02 6三 white:KI -> white:FU` はposition boost差0.345がほぼ全ギャップ、`通常_02 8八 black:HI -> black:GI` はbase差0.0154/weighted差0.0168まで縮む一方、KY/HI/FU吸着の多くはshape/template差が残ることを確認。B5-25ではdisplay 8八限定home rook tie-breakで `通常_02 8八 black:HI` をtop1へ戻し、`errors=13 -> 12`, `top1/top3=0.8929`。B5-27ではsource/target bbox・center・ink ratio/count診断を `diagnostic_candidates` 限定で追加し、低信頼/小marginセルにgateして診断ON speed gateを維持。`unknown_on_piece=13` は安全側のまま。B5-9では `WHITE KI 通常_10 r3c4` positive supplyは変化なし、`BLACK HI 通常_09 r8c4` は `errors=16`, `unknown_on_piece=16` に悪化したため不採用。B5-10からB5-14では `WHITE GI 通常_02 r6c5`, `WHITE GI 通常_03 r3c7`, `BLACK FU 通常_01 r4c8 + 通常_02 r5c5`, `BLACK OU 通常_14 r8c7`, `BLACK GI 通常_09 r8c2` allowlist除去をprobeし、悪化または変化なしのため全撤回。B5-26のFU supply probeも `errors=15`, `unknown_on_piece=18`, speed overで不採用。初期配置パターン確定セルのreport debug候補は速度gate維持のため抑制。残るboard error 12件のtop1は `white:KY=9`, `black:GI=1`, `white:HI=1`, `white:FU=1`。残るerrorはすべてconfirmed unknownで、危険な誤confirmは出していない。B5-5/B5-14/B5-26でedge-red cleanmask、source-diverse/FU-diverse export、band scalar scoring、extra HI allowlist、too-strict GI confirm guard、単独blackGI r8c4 allowlist、blackHI r8c4 allowlist、追加WHITE GI/BLACK FU/BLACK OU/FU supply allowlist、blackGI r8c2除去はいずれも不採用。
- mask variant cell auditの再現QAは残タスクだが、ユーザー判断により、次の主作業はcell-crop classifier offline baselineを優先する。
- mask variant / multi-template consensus / chamfer系はいずれもAI-2 QA済みだが、現時点ではproduction化しない。
- Android評価レポートの `model.excluded_source` metadataは最新runで確認済み。legacy report以外で `--allow-missing-excluded-source` を使わない。
- Pixel 7aが戻ったら同じno-leak smokeを再実行する。
- `tools/samples/screenshots_by_app_piece_style` を主なサンプルセットとして扱う。

詳しくは `AI_CONTEXT/handoffs/HANDOFF_2026-05-09_android-recognition.md`、`AI_CONTEXT/handoffs/HANDOFF_2026-05-09_recognition-qa.md`、`AI_CONTEXT/reviews/APP_REVIEW_REPORT_2026-05-09.md` を参照してください。

アプリ全体レビューで特に高優先のもの:

- 解析Jobのキャンセル/世代管理を入れ、古い解析結果が新しい状態を上書きしないようにする。
- OCR完了時に、ユーザーが手動編集した盤面/持ち駒を上書きしないようにする。
- 二歩、行き所のない駒、負の持ち駒などはCでexport blocking error化済み。残りはUI/SFENコピー共有経路が `validation.canExport` と整合しているかの確認。
- releaseの署名、minify/shrink、backup/privacy、診断ZIPの扱いを整理する。

toolsレビューで特に高優先のもの:

- no-leak評価を既定にし、評価対象と同じ画像由来のlabel/calibration/known sampleが混ざったら失敗させる。
- ラベル探索をstem依存から `app/style/sample` の相対キーへ移行する。
- `--clean` やdraft生成の削除/上書き操作を安全化する。
- skipped、leak、missing labels、high-confidence errors、slow samplesをCI gateとして非ゼロ終了にできるようにする。
- 依存バージョン、seed、input digest、model digest、asset coverageを成果物へ残す。

ラベル作業で特に重要なもの:

- 2026-05-09時点の新規56枚には、ユーザー修正済み、サブエージェント修正済み、ドラフト要確認が混在する。
- `label_source` の有無だけを正解判定に使わない。visual review と inventory audit を併用する。
- ユーザー修正済みラベルを安易に上書きしない。
- 駒表示やHTMLレビュー/編集UIは日本語優先にする。

## 作業ルール

- Android本体の作業では、担当範囲でない限り `tools/` を編集しないでください。
- 認識精度や評価が絡む作業では、`tools/`、`app/src/main/assets`、`app/src/androidTest` が関係します。
- サンプル画像/ラベルはユーザーの作業成果を含む可能性があります。勝手に削除、移動、復元しないでください。
- 低信頼・僅差の認識を無理に確定させない方針です。`unknown` は意図された安全側の出力です。
- ユーザーが明示した場合は、複数サブエージェントでアプリ本体、Pythonツール、既存ドキュメントなどを分担調査してください。
- 作業終了時は必ずMarkdownの引き継ぎ資料を作成してください。ルールと雛形は `AI_CONTEXT/HANDOFF_TEMPLATE.md` にあります。
- 今後の新しい引き継ぎは `docs/` や `tools/` ではなく、必ず `AI_CONTEXT/handoffs/` に作成してください。

## 主要ファイル

- Android入口: `app/src/main/java/com/example/shogigazo/MainActivity.kt`
- 画面状態/解析統括: `app/src/main/java/com/example/shogigazo/ui/ShogiViewModel.kt`
- Compose UI: `app/src/main/java/com/example/shogigazo/ui/ShogiGazoApp.kt`
- ドメインモデル: `app/src/main/java/com/example/shogigazo/domain/ShogiModels.kt`
- 盤面/持ち駒画像処理: `app/src/main/java/com/example/shogigazo/image/BitmapTools.kt`
- 駒テンプレート認識: `app/src/main/java/com/example/shogigazo/image/PieceTemplateRecognizer.kt`
- 持ち駒集約: `app/src/main/java/com/example/shogigazo/image/HandCountAggregator.kt`
- KIF/SFEN出力: `app/src/main/java/com/example/shogigazo/export`
- Python補助ツール: `tools/`
- 主要サンプル: `tools/samples/screenshots_by_app_piece_style`
- 主要ラベル: `tools/samples/labels/boards_by_app_piece_style`
- Android認識assets: `app/src/main/assets`
