# Shogi Gazo

Android MVP for converting a digital shogi-board screenshot into a reusable position file.

The app is intentionally small:

- Kotlin + Jetpack Compose, single `app` module
- Photo Picker / Android share receive for image input
- App/source and piece-design selection after image input
- Python-helper-equivalent board grid detection
- Screenshot setup flow for source app, piece design, and board orientation before analysis
- 9x9 cell split for digital screenshots
- Separate board-grid crop and per-cell piece cutout margins
- 81-cell piece cutout preview before manual correction
- Empty-cell estimation from each cutout, with ML Kit Japanese OCR as a secondary candidate generator
- 9x9 board and hand-stand review before export
- KIF/BOD position export and SFEN text export
- Bundled board/piece artwork in `app/src/main/res/drawable-nodpi/`

## Bundled Artwork

- `shogi_board.png`: board wood texture
- `shogi_pieces.png`: 8 x 4 piece sprite sheet

The sprite sheet is interpreted as:

- row 1: black normal pieces, columns 王/飛/角/金/銀/桂/香/歩
- row 2: black promoted pieces, columns 王/龍/馬/blank/全/圭/杏/と
- row 3: white normal pieces, same columns
- row 4: white promoted pieces, same columns

## What This App Exports

A single screenshot cannot reconstruct the move history. The generated `.kif` is therefore a position KIF/BOD: it stores the recognized board as a starting position, not a normal game record with moves.

SFEN export requires a complete position: all cells resolved, side to move selected, and both players' hands confirmed.

## Basic Flow

1. Select or share a board screenshot into the app.
2. Choose the source app and piece design for the screenshot.
3. Run analysis to detect board lines, split 81 cells, estimate empty cells, and add OCR candidates.
4. Confirm the position on the 9x9 shogi board and correct any cells.
5. Confirm side to move, move number, and both players' hands on the hand stands.
6. Export or share KIF/SFEN once the position is complete.

## Build Requirements

- JDK 17 or newer
- Android SDK Platform 36
- Android Build Tools 36.x
- Android Studio or the included Gradle wrapper

Build from the repository root:

```powershell
.\gradlew.bat test
.\gradlew.bat assembleDebug
```

## Development Tools

Python helpers live in `tools/`. They are only for local image-processing experiments and are not embedded into the Android app. Treat `tools/` as a separate helper area and do not edit it when working on Android app changes unless that is explicitly the assigned task.

```powershell
python -m venv .venv
.venv\Scripts\pip install -r tools\requirements.txt
python tools\detect_board_grid.py tools\samples\screenshots\sample_001.png --out tools\out\grid_debug
```
