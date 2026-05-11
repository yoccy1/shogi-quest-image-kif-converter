# Development Tools

> AI向けの全体文脈と最新QA方針は `AI_CONTEXT/README.md` と `AI_CONTEXT/DEVELOPMENT_AND_QA.md` に集約済みです。このファイルは `tools/` 固有の使い方メモとして残しています。古い記述が一部あるため、作業前に `AI_CONTEXT` 側も確認してください。

These helpers are for local recognition experiments only. The Android app does not embed Python.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -r tools\requirements.txt
```

## Detect Board Grid Lines

Put real screenshots in:

```text
tools\samples\screenshots
```

Run the detector:

```powershell
python tools\detect_board_grid.py --out tools\out\grid_debug
```

You can also pass a single image:

```powershell
python tools\detect_board_grid.py tools\samples\screenshots\sample_001.png --out tools\out\grid_debug
```

For each input image, the script writes:

- `grid_overlay.png`: original screenshot with detected board area and 10 x 10 grid lines
- `report.json`: detected line positions, grid rectangle, and Android-style `cropSelection`
- `grid_area.png`: crop from the first to last detected grid line
- `cells\*.png`: 81 cell crops based on the detected grid lines

This is the only supported Python helper. It matches the Android pipeline: detect the board grid first, then cut cells from the detected 10 x 10 line positions.

The detector tries both dark-line projection and brown board-line projection, then uses the higher-confidence result. This matters for screenshots where the board grid is brown rather than black.

## Recognize Pieces From 81 Cells

After grid detection has written `cells\*.png`, run:

```powershell
python tools\recognize_board_pieces.py tools\out\grid_debug\sample_001\cells
```

The recognizer uses the bundled `app\src\main\res\drawable-nodpi\shogi_pieces.png` sprite sheet as templates. It writes:

- `piece_report.json`: per-cell state, confidence, and top candidates
- `recognized_board.json`: compact board rows
- `recognized_board_preview.png`: 9 x 9 visual preview with labels
- `piece_match_overlay.png`: 9 x 9 preview with the matched bounding box for each best candidate

The default implementation is `hog_svm`. It trains a small OpenCV SVM from synthetic samples generated from the bundled piece sprites, then merges that learned glyph-shape prior with the stable OpenCV template candidate for the same-app sprite case. It uses:

- HOG features from cleaned glyph masks and whole-cell edges
- letter ink overlap from black/red text pixels
- edge overlap for the piece body and glyph outline
- black/red text color compatibility
- optional calibration templates from `tools\samples\screenshots\初期配置`
- optional labeled gameplay templates from `tools\samples\labels\boards`

Low-confidence and low-margin matches are intentionally left as `unknown` with candidates rather than forced to a wrong piece. In that case, inspect the top candidates in `piece_report.json`, the `?` label in `recognized_board_preview.png`, or the bounding boxes in `piece_match_overlay.png`.

Initial-position calibration screenshots are enabled by default when `tools\samples\screenshots\初期配置` exists. The recognizer detects the board in each image, auto-labels the known 40 starting pieces, records the full sample counts in `piece_report.json` under `calibration`, and uses a capped set of matching-app templates for normal pieces.

Labeled gameplay boards in `tools\samples\labels\boards` are also enabled by default. These add real in-game templates for normal and promoted pieces, including decorated boards, selected squares, and app-specific promoted glyphs. The templates are keyed separately from initial-position templates so their cached glyph features do not collide.

For calibrated pieces, recognition uses precomputed normalized glyph masks with small source-specific shift/scale variants instead of full sliding template scans. This keeps per-cell recognition fast while still leaving close calls as `unknown`.

You can point to another calibration set:

```powershell
python tools\recognize_board_pieces.py tools\out\analysis\sample_001\recognition_cells --method opencv --calibration-dir tools\samples\screenshots\初期配置
```

You can point to another labeled board set:

```powershell
python tools\recognize_board_pieces.py tools\out\analysis\sample_001\recognition_cells --method opencv --board-labels-dir tools\samples\labels\boards
```

The recognizer also writes:

- `empty_mask_overlay.png`: exact-cell ink mask and empty-score debug
- `piece_like_overlay.png`: `empty`, `piece`, and `unknown` state color map
- `candidate_grid.png`: top candidates and scores per cell

You can still run the OpenCV template matcher or old Pillow-only matcher for comparison:

```powershell
python tools\recognize_board_pieces.py tools\out\analysis\sample_001 --method opencv
python tools\recognize_board_pieces.py tools\out\analysis\sample_001 --method legacy
```

## Render Recognized Position

Render the recognized board with the same board and piece images bundled in the Android app:

```powershell
python tools\render_recognized_position.py tools\out\analysis\sample_001\analysis_report.json
```

The renderer writes:

- `position_render_confirmed.png`: board render with only confirmed `piece` cells
- `position_render_candidates.png`: board render with confirmed pieces plus translucent best candidates for `unknown` cells
- `position_comparison.png`: left side is the detected input grid area, right side is the rendered recognition result

Use `position_comparison.png` to visually check whether the recognized position matches the screenshot. Candidate pieces are labeled with `?` so they are not mistaken for confirmed recognition.

## Lightweight Visual Review

When you want to check recognition by eye without opening many output folders, create a single HTML review page from a benchmark or analysis output directory:

```powershell
python tools\make_visual_review.py tools\out\benchmark_piece_recognition --include-hands
```

`benchmark_piece_recognition.py` writes this same `visual_review.html` automatically after each run. Use `--no-visual-review` only when you want the absolute smallest output.

For a smoke-test folder:

```powershell
python tools\make_visual_review.py tools\out\benchmark_smoke_includehands --include-hands
```

Open the generated `visual_review.html`. Review in this order:

1. Red cells: label comparison mismatches. These are the only cells that need careful checking first.
2. Yellow cells: `unknown` or low-confidence recognition. These are uncertain, not forced corrections.
3. Hands panel: compare actual and expected captured-piece counts.
4. Screenshot panel: use it only as the visual source of truth when the HTML board looks suspicious.

This review page is a confirmation aid only. It does not feed teacher labels back into inference and should not be used as a label-oracle shortcut.

## Detect Captured-Piece Areas

Detect the hand areas, also called captured-piece stands, around the detected board:

```powershell
python tools\detect_hand_areas.py tools\samples\screenshots\sample_001.png --out tools\out\hand_areas
```

For each input image, the script writes:

- `hand_area_report.json`: detected top/bottom/left/right hand areas, owners, confidence, and evidence
- `hand_area_overlay.png`: original screenshot with search bands, detected grid, hand areas, and component evidence
- `areas\*.png`: crops for each detected hand area

The detector uses the board grid as the anchor. It supports the current sample layouts for ぴよ将棋, 将棋ウォーズ, and 将棋クエスト. When no captured piece is visible, it emits a low-confidence layout fallback area so downstream recognition can return zero pieces instead of treating the area as missing.

## Recognize Captured Pieces

Recognize held pieces and count them by owner:

```powershell
python tools\recognize_hand_pieces.py tools\samples\screenshots\sample_001.png --out tools\out\hand_recognition
```

For each input image, the script writes:

- `hand_pieces_report.json`: `hands.black` and `hands.white` counts for `HI`, `KA`, `KI`, `GI`, `KE`, `KY`, and `FU`
- `hand_area_report.json`: the hand-area detection report used by recognition
- `hand_piece_overlay.png`: accepted pieces, unknown candidates, and recognized digit candidates
- `hand_candidate_grid.png`: cropped candidate previews and top labels
- `candidate_crops\*.png`: individual crops passed to the existing piece classifier

The recognizer reuses the HOG/OpenCV piece classifier from `recognize_board_pieces.py`. It ignores kings and promoted pieces for hand counts. Owner is inferred from the detected hand side: top/left is white, bottom/right is black. Digit counts are associated only when the digit is a high-confidence candidate near a strong accepted piece; otherwise counts fall back to visible piece icons.

This is an experimental v1 helper. Use the debug overlay to verify results before treating counts as ground truth, especially on stylized pieces or UI layouts outside the bundled samples.

## Full Screenshot Analysis

Run grid detection and piece recognition in one command:

```powershell
python tools\analyze_shogi_screenshot.py tools\samples\screenshots\sample_001.png --out tools\out\analysis
```

`analyze_shogi_screenshot.py` also accepts `--calibration-dir`; by default it uses `tools\samples\screenshots\初期配置` when that folder is present.

For each input image, this writes the grid debug images, recognition reports, rendered board images, comparison image, and `analysis_report.json`.

The full analysis writes two cell sets:

- `cells\*.png`: exact 9 x 9 grid cells, used for empty-square detection
- `recognition_cells\*.png`: wider crops with horizontal and vertical padding, used for piece matching so tall pieces are not clipped

## Evaluate Piece Recognition

Board labels for the non-initial sample screenshots live in:

```text
tools\samples\labels\boards
```

Each label file is a 9 x 9 board, ordered from `9一` to `1九`. Cells are `empty` or `color:piece`, for example `black:FU` or `white:RY`.

Evaluate the current analysis output:

```powershell
python tools\evaluate_piece_recognition.py tools\out\analysis_opencv --out tools\out\analysis_opencv\evaluation_report.json
```

The evaluator reports empty-square accuracy, piece-presence accuracy, confirmed identity accuracy, top-1 and top-3 candidate accuracy, high-confidence errors, and confusion matrices. Use this before changing recognition thresholds so that new recognizer changes can be compared against a stable baseline. When the same labeled screenshots are used as recognition templates, this is a supervised self-check; point `--board-labels-dir` at an empty or missing directory when you want to measure the initial-position-only baseline.
