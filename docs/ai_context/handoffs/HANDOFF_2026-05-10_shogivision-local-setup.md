# HANDOFF 2026-05-10 ShogiVision Local Setup

## Status

ShogiVisionをローカルで起動可能な状態までセットアップした。

対象:

- `C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master`

## What Changed

作成/追加:

- `ShogiVision-master\ShogiVision-master\venv\`
- `ShogiVision-master\ShogiVision-master\models\`
- `ShogiVision-master\ShogiVision-master\run_shogivision.ps1`

変更:

- `ShogiVision-master\ShogiVision-master\tools\ShogiVision\main.py`
  - Windowsで `PyQt5` を先にimportすると `torch` の `c10.dll` 初期化が失敗したため、`torch` を先にimportする小修正を入れた。

## Environment

Python:

- `Python 3.12.10`

venv:

```powershell
cd C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master
python -m venv venv
.\venv\Scripts\python -m pip install --upgrade pip setuptools wheel
.\venv\Scripts\python -m pip install -r requirements.txt
.\venv\Scripts\python -m pip install gdown
```

`pip install -r requirements.txt` は成功。`ultralytics` 経由で `torch 2.11.0+cpu` / `torchvision 0.26.0` も入った。

`pypdfium2==4.30.1` はyanked警告が出たが、インストール自体は成功している。

## Models

READMEのGoogle Drive Modelsフォルダから `gdown` で取得した。

```powershell
.\venv\Scripts\python -m gdown --folder "https://drive.google.com/drive/folders/1QTWss5RQerwVI-kkQVF-ml3MvJ0GjDcT?usp=sharing" -O models
```

取得済み:

| file | size |
|---|---:|
| `models\board_segmenter.onnx` | 11.07 MiB |
| `models\board_segmenter.pt` | 5.72 MiB |
| `models\direction_classifier.pt` | 3.04 MiB |
| `models\figure_classifier.pt` | 10.55 MiB |
| `models\mixed.onnx` | 2.50 MiB |

通常GUI起動に最低限必要なのは `board_segmenter.pt` と `mixed.onnx`。

## Launch

推奨起動:

```powershell
cd C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master
powershell -ExecutionPolicy Bypass -File .\run_shogivision.ps1
```

または:

```powershell
cd C:\Users\sakas\AndroidProjects\shogi_gazo\ShogiVision-master\ShogiVision-master
.\venv\Scripts\python.exe -m tools.ShogiVision.main
```

`python tools\ShogiVision\main.py` の直実行はimport pathが壊れやすいため非推奨。

## Verification

実行済み:

```powershell
.\venv\Scripts\python -m pip check
```

結果:

- `No broken requirements found.`

```powershell
.\venv\Scripts\python -B -c "import config; print(config.paths.ROOT_DIR); print(config.GLOBAL_CONFIG.Settings.predict_board)"
```

結果:

- config import OK

```powershell
.\venv\Scripts\python -c "import numpy, cv2, onnxruntime, PyQt5, ultralytics, shogi, imagehash; print('IMPORTS_OK')"
```

結果:

- `IMPORTS_OK`

```powershell
.\venv\Scripts\python -c "import tools.ShogiVision.main; print('MAIN_IMPORT_OK')"
```

結果:

- `MAIN_IMPORT_OK`

```powershell
.\venv\Scripts\python -c "from extra import factories; r=factories.default_recognizer(); cd=factories.default_corner_detector(); print(type(r).__name__); print(type(cd).__name__); print('FACTORIES_OK')"
```

結果:

- `RecognizerONNX`
- `YOLOSegmentationCornerDetector`
- `FACTORIES_OK`

```powershell
.\venv\Scripts\python -c "import os, torch; from PyQt5.QtCore import QLibraryInfo; os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=QLibraryInfo.location(QLibraryInfo.PluginsPath); from PyQt5.QtWidgets import QApplication; from GUI.views.ShogiVision import ShogiVision; app=QApplication([]); sv=ShogiVision(); print('GUI_INSTANTIATE_OK')"
```

結果:

- `GUI_INSTANTIATE_OK`

## Known Warnings

GUI import/instantiate時にOpenCVのcamera warningが出ることがある。

例:

- `Camera index out of range`
- `async ReadSample() call is failed`

これはカメラデバイス探索由来で、GUI生成やモデル読み込みを止めるエラーではなかった。

## Notes For Next AI

- ShogiVision自体のLICENSEファイルはzip内に見当たらない。利用/再配布/Android同梱前にGitHub側のライセンス確認が必要。
- `ultralytics` のパッケージlicenseは `AGPL-3.0` と表示される。Androidアプリに同梱/派生利用する場合は法務・ライセンス観点の確認が必要。
- 今回はShogiVision GUIを使える状態にしただけで、現行Androidアプリへの統合はしていない。
- Android化の次段階では `mixed.onnx` / `mixed.tflite` 相当を、現行アプリの81マス切り出し後にdebug/offline比較として接続するのが安全。

## Next

次にやるなら:

1. 実際に `run_shogivision.ps1` でGUIを開き、手元のスクリーンショットを読ませる。
2. ShogiVisionの出力と現行Android評価セットの正解ラベルを比較するoffline bridgeを作る。
3. Androidへ入れる前に、ShogiVisionの `mixed.onnx` 推論結果をCSV化し、B5-29 baselineと比較する。
