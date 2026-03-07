# OMAHA-READER CODEX

## Project Overview

Omaha-Reader is a real-time PLO (Pot Limit Omaha) poker assistant that captures poker table screenshots, detects cards, positions, and actions via OCR/template matching, and displays results in a browser UI.

The system captures poker tables running with **Jurojin HUD overlay** at **784x584 pixels**, processes them through a detection pipeline, and serves results via HTTP polling to a web dashboard.

---

## Architecture

```
Screen Capture → Image Processing → Detection Pipeline → Game Snapshot → HTTP API → Browser UI
```

### Key Components

| Layer | Files | Role |
|-------|-------|------|
| Detection entry | `poker_game_processor.py` | Orchestrates all detection, builds GameSnapshot |
| Position detection | `detect_utils.py` | Jurojin badge OCR + native template fallback |
| Template matching | `template_matcher_service.py` | Generic template matching engine |
| Template loading | `template_registry.py` | Lazy-loads template images by category |
| Matching core | `opencv_utils.py` | Low-level cv2.matchTemplate wrapper |
| Matching utils | `template_matching_utils.py` | Parallel matching, overlap filtering |
| Position recovery | `position_service.py` | Converts detections to Position enums |
| RFI ranges | `rfi_range_service.py` | CSV-based preflop raise/fold lookup |
| Game state | `game_snapshot.py` | Data model with hero_position, rfi_action |
| Data formatter | `game_data_formatter.py` | Formats snapshot for web API |
| Browser UI | `poker-app.js` + `poker-app.css` | Renders cards, positions, badges |

---

## Detection Pipeline Flow

### Position Detection (`detect_utils.py → detect_positions()`)

Three-method cascade with early exit:

```
1. PRIMARY:   Hero OCR direct (seat 1)     → if found, return immediately
2. FALLBACK:  Seats 2-6 OCR + deduce hero  → if any seat found, deduce hero via cycle
3. LAST RESORT: Native template matching    → per-seat template matching fallback
```

### Method 1: Hero Direct OCR

```python
_detect_hero_badge_ocr(cv2_image)
  → _find_badge_in_region()    # HSV gold ring detection in JUROJIN_POSITION_REGIONS[1]
  → _ocr_badge_at()            # Crop 36x36, CLAHE, Otsu/threshold, pytesseract
  → returns Detection("BTN", center, None, 0.95)
```

### Method 2: Other Seats OCR + Deduce

```python
_detect_other_seats_ocr(cv2_image)
  → for seat 2-6: _find_badge_in_region() + _ocr_badge_at()
  → validate: no duplicate positions (means overlay absent)
  → _deduce_hero(other_positions)
    → SEAT_POSITION_CYCLE = ['BTN', 'SB', 'BB', 'EP', 'MP', 'CO']
    → hero_idx = (known_idx - (seat - 1)) % 6
```

### Method 3: Native Template Matching (last resort)

```python
_detect_native_positions(cv2_image)
  → for each seat in PLAYER_POSITIONS:
    → coords_to_search_region() → fractional search region
    → TemplateMatchService.find_positions(image, search_region)
    → pick detected[0] (highest score)
```

---

## Coordinate Maps

### Image Dimensions: 784 x 584 pixels

### JUROJIN_POSITION_REGIONS (for OCR badge detection)

Large regions (~100-120px) to find Jurojin circle badges:

| Seat | x | y | w | h | Location |
|------|-----|-----|-----|-----|----------|
| 1 (hero) | 255 | 340 | 110 | 110 | Bottom center |
| 2 | 0 | 300 | 100 | 110 | Left bottom |
| 3 | 0 | 120 | 120 | 100 | Left top |
| 4 | 250 | 70 | 120 | 120 | Top center |
| 5 | 530 | 90 | 120 | 120 | Right top |
| 6 | 530 | 300 | 120 | 120 | Right bottom |

### PLAYER_POSITIONS (for native template matching fallback)

Small 40x40 regions for template matching:

| Seat | x | y | w | h |
|------|-----|-----|-----|-----|
| 1 | 300 | 375 | 40 | 40 |
| 2 | 35 | 330 | 40 | 40 |
| 3 | 35 | 173 | 40 | 40 |
| 4 | 297 | 120 | 40 | 40 |
| 5 | 562 | 168 | 40 | 40 |
| 6 | 565 | 332 | 40 | 40 |

### ACTION_POSITIONS (for Jurojin action detection)

| Seat | x | y | w | h |
|------|-----|-----|-----|-----|
| 1 | 300 | 430 | 200 | 30 |
| 2 | 10 | 400 | 200 | 30 |
| 3 | 25 | 120 | 200 | 30 |
| 4 | 315 | 80 | 200 | 30 |
| 5 | 580 | 130 | 200 | 30 |
| 6 | 580 | 380 | 200 | 30 |

---

## Template System

### Directory Structure

```
apps/table_detector/resources/templates/canada/
├── player_cards/          # Card templates (AS, KH, etc.)
├── table_cards/           # Table card templates
├── positions/             # OLD flat text position labels (24 templates)
│   ├── BB.png, BTN.png, CO.png, EP.png, MP.png, SB.png
│   ├── BB_fold.png, BTN_fold.png, CO_fold.png, EP_fold.png, MP_fold.png, SB_fold.png
│   ├── BB_low.png, BTN_fold_red.png, EP_low.png, EP_now.png, NO.png
│   └── bets.png, c_bets.png, calls.png, calls_1.png, checks.png, folds.png, open_raises.png
├── jurojin_positions/     # Jurojin circle badge templates (54x54px each)
│   ├── BB.png, BTN.png, CO.png, EP.png, MP.png, SB.png
├── jurojin_inner/         # Inner text crops (30x30px each, v1 + v2)
│   ├── BB.png, BB_v2.png, BTN.png, BTN_v2.png
│   ├── CO.png, CO_v2.png, EP.png, EP_v2.png
│   ├── MP.png, MP_v2.png, SB.png, SB_v2.png
├── moves/                 # Jurojin action templates
│   ├── bet.png, call.png, cb.png, check.png, fold.png
│   ├── limps.png, or.png, or_2.png, or_35.png
└── actions/               # Generic action button templates
```

### Template Registry Properties

| Property | Directory | Used By |
|----------|-----------|---------|
| `player_templates` | `player_cards/` | `find_player_cards()` |
| `table_templates` | `table_cards/` | `find_table_cards()` |
| `position_templates` | `positions/` | **NOT USED** (old flat text) |
| `jurojin_position_templates` | `jurojin_positions/` | `find_positions()` |
| `jurojin_inner_templates` | `jurojin_inner/` | Not currently used |
| `jurojin_action_templates` | `moves/` | `find_jurojin_actions()` |
| `action_templates` | `actions/` | `find_actions()` |

---

## Template Matching Configuration

### MatchConfig Parameters

```python
@dataclass
class MatchConfig:
    search_region: Optional[Tuple[float, float, float, float]]  # (left, top, right, bottom) as ratios
    threshold: float = 0.955          # Minimum match score
    overlap_threshold: float = 0.3     # IoU threshold for dedup
    min_size: int = 20                 # Minimum template size
    scale_factors: List[float] = [1.0] # Scale variations
    sort_by: str = 'x'                # 'x', 'y', 'score'
    max_workers: int = 4               # Thread pool size
    match_method: int = cv2.TM_CCORR_NORMED  # OpenCV matching method
```

### Per-Function Configs

| Function | Templates | Threshold | Method | Notes |
|----------|-----------|-----------|--------|-------|
| `find_player_cards()` | player_cards | 0.955 | TM_CCORR_NORMED | Search region (0.2, 0.5, 0.8, 0.95) |
| `find_table_cards()` | table_cards | 0.955 | TM_CCORR_NORMED | Full image search |
| `find_positions()` | **jurojin_positions** | **0.55** | **TM_CCOEFF_NORMED** | Per-seat search region |
| `find_actions()` | actions | 0.95 | TM_CCORR_NORMED | Action button area |
| `find_jurojin_actions()` | moves | 0.98 | TM_CCORR_NORMED | Per-seat action regions |

---

## OCR Pipeline (`_ocr_badge_at`)

Optimized to max 7 pytesseract calls per badge:

```
Input: 36x36 crop centered on badge center
  → Resize to 180x180 (INTER_CUBIC)
  → Convert to grayscale
  → CLAHE (clipLimit=3.0, tileGrid=4x4)

Attempt order (early exit on VALID_POSITIONS match):
  1. Otsu BINARY_INV  (auto-threshold, inverted)
  2. Otsu BINARY      (auto-threshold, normal)
  3. BINARY t=220     (bright badges)
  4. BINARY t=180
  5. BINARY t=140
  6. BINARY_INV t=140 (dim badges)
  7. BINARY_INV t=100

Config: --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ
```

### Badge Finding (`_find_badge_in_region`)

HSV color ranges for Jurojin gold ring detection:

| Range | H | S | V | Min Area | Description |
|-------|-----|-----|-----|----------|-------------|
| Bright gold | 10-40 | 80-255 | 120-255 | 800 | Standard active badge |
| Wide gold | 8-45 | 40-255 | 80-255 | 400 | Faded/variant badge |
| Dimmed gray-blue | 100-140 | 50-120 | 100-180 | 300 | Inactive/folded badge |

Filters: min contour 25x25, max area 5000, aspect ratio 0.7-1.4 (circularity check).

---

## Key Constants

```python
VALID_POSITIONS = {'BB', 'SB', 'BTN', 'EP', 'MP', 'CO'}
SEAT_POSITION_CYCLE = ['BTN', 'SB', 'BB', 'EP', 'MP', 'CO']
IMAGE_WIDTH = 784
IMAGE_HEIGHT = 584
```

---

## Browser UI

### Hero Badges Display

`poker-app.js → createHeroBadges(detection)`:

- Blue badge: hero position (BTN, SB, BB, EP, MP, CO)
- Green badge: RFI RAISE action
- Red badge: RFI FOLD action
- Gray "--" badge: no position data

### Data Flow to UI

```
GameSnapshot
  → to_game_update_message()    # Includes hero_position, rfi_action
  → HTTP /api/detections
  → game_data_formatter.py      # Passes hero_position, rfi_action through
  → poker-app.js                # Renders badges in hero-badges-row
```

### CSS Classes

```css
.hero-badges-row    { display: flex; gap: 8px; margin-top: 8px; }
.hero-position-badge { background: #2196F3; color: white; border-radius: 14px; }
.rfi-action-badge.rfi-raise { background: #4CAF50; }
.rfi-action-badge.rfi-fold  { background: #f44336; }
```

---

## RFI (Raise First In) Logic

Located in `poker_game_processor.py`, lines 65-80:

```python
if hero_position is not None and hero_position != "NO":
    if len(player_cards_detections) == 4:  # Full Omaha hand
        combo_str = "".join(card_names)
        rfi_result = rfi_service.check_rfi(combo_str, hero_position)
        # Returns RAISE or FOLD
```

RFI is only checked for EP, MP, CO, BTN positions (preflop raising positions).

---

## Exception Handling

### Critical Fix: `poker_game_processor.py` line 62

```python
# BEFORE (broken): only caught OmahaEngineException
except OmahaEngineException as e:

# AFTER (fixed): catches ALL exceptions including PositionService's plain Exception
except Exception as e:
```

**Why**: `PositionService.get_positions()` raises plain `Exception` when < 6 positions detected. The old `except OmahaEngineException` didn't catch it, which killed the entire `create_game_snapshot()` including card detection.

---

## Debug System

### Debug Captures (`_save_debug`)

- Directory: `{project_root}/debug_captures/`
- Per-table cooldown: 5 seconds (using pixel-based table key)
- Saves: `table_{key}_{timestamp}.png` (full) + `hero_{key}_{timestamp}.png` (hero crop)
- Entire method wrapped in try/except to prevent crashes

### Table Key Computation

```python
tbl_key = int(cv2_image[10, 10, 0]) + int(cv2_image[50, 50, 0]) * 256
```

Simple hash from two pixel values — different tables have different backgrounds.

---

## Changes Log (Current Session)

### 1. OCR Performance Optimization

**File**: `detect_utils.py`
**Before**: 3 crop sizes x 12 thresholds = 36 pytesseract calls per badge
**After**: 1 crop size, Otsu first + 5 key thresholds = max 7 calls per badge

### 2. Exception Handling Fix

**File**: `poker_game_processor.py` line 62
**Before**: `except OmahaEngineException as e:`
**After**: `except Exception as e:`
**Impact**: Cards stopped detecting entirely when position service threw plain Exception

### 3. Template Matching: Wrong Templates + Wrong Method

**File**: `template_matcher_service.py → find_positions()`

**Before**:
```python
threshold=0.99
match_method=cv2.TM_CCORR_NORMED       # Cross-correlation (bad for similar templates)
templates=position_templates             # OLD flat text templates from positions/
```

**After**:
```python
threshold=0.55
match_method=cv2.TM_CCOEFF_NORMED       # Correlation coefficient (discriminates patterns)
templates=jurojin_position_templates     # Actual Jurojin circle badges from jurojin_positions/
```

**Root cause of "all seats = EP" bug**: TM_CCORR_NORMED gives high baseline scores when templates are 90% identical (same gold ring, different text). EP template was also a dark/dimmed badge (mean brightness 80 vs ~150 for others) that correlated well with dark table regions.

### 4. match_method Parameter Threading

Added `match_method` parameter through the entire call chain so position detection uses TM_CCOEFF_NORMED while card detection keeps TM_CCORR_NORMED:

- `opencv_utils.py → match_template_at_scale()`: added `match_method` param (default: TM_CCORR_NORMED)
- `template_matching_utils.py → find_single_template_matches()`: threaded `match_method` through
- `template_matcher_service.py → MatchConfig`: added `match_method` field
- `template_matcher_service.py → find_matches()`: passes `config.match_method` to workers

### 5. Hero Badge UI

**Files**: `poker-app.js`, `poker-app.css`
- Added `createHeroBadges()` function with position + RFI badges
- Added `hero-badges-row` div in player cards section
- Gray "--" placeholder when no position data
- Console.log debug for hero detection data

### 6. Debug Save Robustness

**File**: `detect_utils.py → _save_debug()`
- Changed `_DEBUG_LAST_SAVE` from int to dict for per-table cooldown
- Wrapped entire method in try/except to prevent detection crashes
- Added table key to filename for multi-table support

### 7. Badge Circularity Check

**File**: `detect_utils.py → _find_badge_in_region()`
- Added aspect ratio check (0.7-1.4) to reject non-circular shapes
- Prevents stats overlays (Folds, C-Bets, Checks) from being matched as position badges

---

## Known Issues / Current State

1. **Jurojin badges only appear during active hands** — between hands or when sitting out, no position badges are shown by the HUD
2. **EP template in jurojin_positions/ is dark** (mean brightness ~80 vs ~150 for others) — may need recapture from active game state
3. **Position detection requires Jurojin HUD loaded** — "One full hand must be played before HUD appears"
4. **PositionService requires all 6 positions** — throws Exception if < 6 detected (caught by poker_game_processor)

---

## File Locations Quick Reference

```
apps/
├── table_detector/
│   ├── utils/
│   │   ├── detect_utils.py              # Main position detection (OCR + fallback)
│   │   ├── opencv_utils.py              # Template matching core (match_template_at_scale)
│   │   ├── template_matching_utils.py   # Parallel matching, overlap filtering
│   │   └── drawing_utils.py             # Debug visualization
│   ├── services/
│   │   ├── poker_game_processor.py      # Detection orchestrator
│   │   ├── template_matcher_service.py  # High-level template matching API
│   │   ├── template_registry.py         # Template image loading
│   │   ├── position_service.py          # Position enum recovery
│   │   ├── rfi_range_service.py         # Preflop range CSV lookup
│   │   └── flophero_link_service.py     # Solver link generation
│   ├── resources/templates/canada/      # All template images
│   └── domain/
│       └── captured_window.py           # Screen capture wrapper
├── server/
│   ├── web/static/
│   │   ├── poker-app.js                 # Browser UI logic
│   │   └── poker-app.css                # Browser UI styles
│   └── utils/
│       └── game_data_formatter.py       # API data formatting
├── shared/
│   └── domain/
│       ├── game_snapshot.py             # Central data model
│       ├── detection.py                 # Detection dataclass
│       └── position.py                  # Position enum
└── debug_captures/                      # Debug screenshots (auto-generated)
```
