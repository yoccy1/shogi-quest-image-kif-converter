# Shogi Gazo 手順書

この手順書は、将棋スクリーンショットから局面を作り、KIF/SFENを書き出すAndroidアプリを動かすためのものです。

## 1. まず必要なもの

以下をインストールしてください。

- Android Studio
- JDK 17以上
- Android SDK Platform 36
- Android SDK Build Tools 36.x

一番簡単なのは、Android Studioを入れて、その中のSDK Managerから必要なSDKを入れる方法です。

## 2. Android Studioで開く

1. Android Studioを起動します。
2. `Open` を選びます。
3. 次のフォルダを開きます。

```text
C:\Users\sakas\AndroidProjects\shogi_gazo
```

4. Gradle Syncが始まるので完了まで待ちます。

もし「JDKが見つからない」と出た場合は、Android Studioの設定でGradle JDKをJDK 17以上にしてください。

## 3. SDKを確認する

Android Studioで次を開きます。

```text
File > Settings > Languages & Frameworks > Android SDK
```

確認する項目:

- `Android API 36` がインストール済み
- `Android SDK Build-Tools 36.x` がインストール済み

入っていなければチェックを入れてインストールします。

## 4. コマンドで確認する場合

PowerShellでプロジェクトフォルダに移動します。

```powershell
cd C:\Users\sakas\AndroidProjects\shogi_gazo
```

テスト:

```powershell
.\gradlew.bat test
```

デバッグAPK作成:

```powershell
.\gradlew.bat assembleDebug
```

APKは通常、次に出ます。

```text
app\build\outputs\apk\debug\app-debug.apk
```

## 5. 実機またはエミュレータで動かす

### エミュレータの場合

1. Android Studio右上のDevice Managerを開きます。
2. 仮想端末を作ります。
3. Android 15/16系、またはAPI 36に近いシステムイメージを選びます。
4. Runボタンを押します。

### 実機の場合

1. Android端末で開発者向けオプションを有効にします。
2. USBデバッグを有効にします。
3. PCにUSB接続します。
4. Android Studioの実行先に端末が出たらRunします。

## 6. アプリの使い方

### 画像から作る

1. `スクリーンショットを選択` を押します。
2. 将棋盤面のスクリーンショットを選びます。
3. 対象アプリと駒デザインを選びます。
4. 盤の向きを選びます。
   - 先手が下
   - 後手が下
5. `解析する` を押します。
6. アプリがPython補助スクリプトと同じ方針で盤面グリッド検出、81マス切り出し、空マス推定、OCR補助候補生成を行います。
7. `駒盤と持ち駒を確認する` 画面で、検出線の重ね描きと81マス切り出しプレビューを確認します。
8. プレビューや9x9駒盤のマスをタップすると、切り出し画像を拡大しながら駒を手動設定できます。
9. 解析後の9x9駒盤で認識結果を確認します。
10. 間違っているマスをタップして修正します。
11. `未確定を空` は、残った未確定マスを一括で空にしたい時に使います。
12. 手番、手数、先手/後手の持ち駒を持ち駒台で確認・入力します。
13. `KIF確認へ` を押します。
14. エラーがなければKIF保存、KIF共有、SFENコピー、SFEN共有ができます。

### 画像なしで試す

1. 起動画面で `画像なしで盤面を手入力` を押します。
2. 初期局面が入ります。
3. 盤面、持ち駒、手番を編集します。
4. 出力画面でKIF/SFENを確認します。

## 7. 重要な注意

このアプリは、スクリーンショット1枚から指し手履歴を復元するものではありません。

出力するKIFは、通常の棋譜ではなく、認識した局面を開始局面として持つ局面KIF/BODです。

また、画像だけでは次の情報は確定できません。

- 手番
- 持ち駒
- 手数
- 直前の指し手
- 対局者名や棋戦情報

そのため、手番と持ち駒は必ずユーザーが確認します。

## 8. よくあるエラー

### `JAVA_HOME is not set`

JDKが見つかっていません。

対応:

- Android Studioでプロジェクトを開き、Gradle JDKをJDK 17以上にする
- または環境変数`JAVA_HOME`をJDK 17以上に設定する

### `SDK location not found`

Android SDKの場所が見つかっていません。

対応:

- Android Studioで開く
- SDK ManagerでSDKを入れる
- 必要ならプロジェクト直下に`local.properties`が自動生成されるのを確認する

例:

```properties
sdk.dir=C\:\\Users\\sakas\\AppData\\Local\\Android\\Sdk
```

### `compileSdk 36 is not installed`

Android API 36が入っていません。

対応:

- SDK Managerから`Android API 36`をインストールする

### 盤面グリッド検出がずれる

自動検出は候補です。盤線が薄い、盤が小さい、スクリーンショット内に文字や装飾が多い場合は外れることがあります。

対応:

- 対象アプリ、駒デザイン、盤の向きがスクリーンショットと合っているか確認する
- 検出線の重ね描きで、盤面の外枠と9x9の線がプレビュー内に入っているか確認する
- 検出が大きく外れる場合は、スクリーンショットを取り直すか、9x9駒盤で手動補正する

### 対象アプリや駒デザインを間違えた

画像選択後の対象アプリと駒デザインは、解析結果に影響します。スクリーンショットの見た目と違うものを選んだ場合は、画像選択からやり直して正しい組み合わせを選んでください。

### 駒種が自動で入らない

現時点の自動処理は、盤面グリッド検出、81マス切り出し、空マス推定、OCR補助候補が中心です。駒種の確定は切り出し画像と解析後の9x9駒盤を見ながら手動補正する前提です。

対応:

- 盤面グリッドを盤面ぴったりにする
- 対象アプリと駒デザインを正しく選び直す
- 盤の向きを切り替える
- 未確定のマスは切り出しプレビューから手動修正する

## 9. 開発補助ツール

Python補助ツールはAndroidアプリには同梱しません。画像分割の確認用で、Androidアプリ本体とは分けて扱います。Androidアプリの作業では、担当範囲に含まれていない限り `tools/` 配下を編集しないでください。

セットアップ:

```powershell
python -m venv .venv
.venv\Scripts\pip install -r tools\requirements.txt
```

盤面線を検出して、重ね描き画像と81マス切り出しを作成:

```powershell
python tools\detect_board_grid.py tools\samples\screenshots\sample_001.png --out tools\out\grid_debug
```

## 10. 最初の確認順

おすすめの進め方:

1. Android Studioでプロジェクトを開く
2. Gradle Syncを通す
3. `.\gradlew.bat test` を通す
4. `.\gradlew.bat assembleDebug` を通す
5. エミュレータか実機で起動する
6. `画像なしで盤面を手入力` からKIF/SFEN出力を確認する
7. 次にスクリーンショット取り込みを試す

ここまで通れば、アプリの基本動作は確認できています。
