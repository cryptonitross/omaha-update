import time
from pathlib import Path
from typing import List, Dict, Optional
from collections import Counter

import cv2
import numpy as np
from loguru import logger

from shared.domain.detection import Detection
from table_detector.utils.opencv_utils import coords_to_search_region
from table_detector.services.template_matcher_service import TemplateMatchService

# Debug: periodic capture for live tracking (per-table cooldown)
_DEBUG_DIR = Path(__file__).resolve().parent.parent.parent.parent / "debug_captures"
_DEBUG_INTERVAL = 5
_DEBUG_LAST_SAVE = {}  # dict: table_hash -> last_save_time

ACTION_POSITIONS = {
    1: (300, 430, 200, 30),
    2: (10, 400, 200, 30),
    3: (25, 120, 200, 30),
    4: (315, 80, 200, 30),
    5: (580, 130, 200, 30),
    6: (580, 380, 200, 30),
}

PLAYER_POSITIONS = {
    1: {'x': 268, 'y': 343, 'w': 90, 'h': 90},  # was 40x40 — too small for 54x54 templates
    2: {'x': 3,   'y': 298, 'w': 90, 'h': 90},
    3: {'x': 3,   'y': 141, 'w': 90, 'h': 90},
    4: {'x': 265, 'y': 88,  'w': 90, 'h': 90},
    5: {'x': 530, 'y': 136, 'w': 90, 'h': 90},
    6: {'x': 533, 'y': 300, 'w': 90, 'h': 90}
}

# Jurojin badge search regions per seat
JUROJIN_POSITION_REGIONS = {
    1: {'x': 280, 'y': 355, 'w': 65, 'h': 65},  # hero bottom center (badge at ~308,405)
    2: {'x': 15,  'y': 150, 'w': 70, 'h': 70},  # left top  (step 3 clockwise from hero)
    3: {'x': 15,  'y': 310, 'w': 70, 'h': 70},  # left bottom (step 1)
    4: {'x': 275, 'y': 100, 'w': 65, 'h': 65},  # top center (step 2)
    5: {'x': 540, 'y': 145, 'w': 65, 'h': 65},  # top right (step 4)
    6: {'x': 555, 'y': 330, 'w': 70, 'h': 70},  # bottom right (step 5)
}

# Clockwise step distance from hero (seat 1) to each seat.
# seat4 EXCLUDED — always gives false positives with score ~1.000
# seat5 INCLUDED — essential for detecting MP position (votes correctly for MP)
SEAT_STEP_MAP = {
    1: 0,  # hero: direct badge (step=0) — used as tiebreaker
    2: 3,  # left-top
    3: 1,  # left-bottom
    5: 4,  # top-right — key for MP detection
    6: 5,  # bottom-right — most reliable for BTN/CO/BB
}

POSITION_MARGIN = 10
IMAGE_WIDTH = 784
IMAGE_HEIGHT = 584
VALID_POSITIONS = {'BB', 'SB', 'BTN', 'EP', 'MP', 'CO'}
SEAT_POSITION_CYCLE = ['BTN', 'SB', 'BB', 'EP', 'MP', 'CO']

# Hero region bounds (to exclude from voting)
HERO_X1, HERO_X2 = 275, 345
HERO_Y1, HERO_Y2 = 350, 425


class DetectUtils:

    # ── Main entry point ──────────────────────────────────────────────

    @staticmethod
    def detect_positions(cv2_image) -> Dict[int, Detection]:
        """
        Detect positions using voting system:
        Each non-hero badge found votes for a hero position.
        Winner is used as hero_position.
        Falls back to native template matching if no votes.
        """
        t0 = time.time()
        try:
            result = DetectUtils._detect_by_voting(cv2_image)
            if result and result.get(1) and result[1].name != "NO":
                ms = (time.time() - t0) * 1000
                hero = result[1].name
                logger.info(f"  ── Position [{ms:.0f}ms]: HERO={hero} (voting) ──")
                DetectUtils._save_debug(cv2_image, result)
                return result

            # Fallback: native template matching
            logger.info("  ── Voting failed, native fallback ──")
            DetectUtils._save_debug(cv2_image)
            return DetectUtils._detect_native_positions(cv2_image)

        except Exception as e:
            logger.error(f"❌ Position detection error: {e}")
            return {}

    # ── Voting-based detection ────────────────────────────────────────

    @staticmethod
    def _detect_by_voting(cv2_image) -> Optional[Dict[int, Detection]]:
        """
        Voting-based hero detection with dedup and seat1 tiebreak:

        1. Scan seats {1,2,3,5,6} (seat4 excluded — always false positives)
        2. DEDUP: if same badge at multiple seats, keep highest-score occurrence
        3. Each surviving badge votes for hero position via SEAT_STEP_MAP
        4. Winner = candidate with most votes
        5. TIE → seat1 direct detection wins (badge at seat1 = hero position)

        Known reliability:
          seat6: best for BTN/CO/BB  seat5: key for MP
          seat1: correct for SB/CO/MP; wrong for BTN(→MP) and BB(→SB)
          seat4: EXCLUDED (false 1.000 scores always)
        """
        try:
            h, w = cv2_image.shape[:2]
            NO_RECT = (0, 0, 0, 0)

            # Step 1: detect best badge per seat
            raw: Dict[int, Detection] = {}
            for seat in SEAT_STEP_MAP:
                coords = JUROJIN_POSITION_REGIONS[seat]
                search_region = coords_to_search_region(
                    coords['x'], coords['y'], coords['w'], coords['h'],
                    image_width=w, image_height=h
                )
                try:
                    matches = TemplateMatchService.find_positions(cv2_image, search_region)
                    if matches and matches[0].name in VALID_POSITIONS:
                        raw[seat] = matches[0]
                except Exception as e:
                    logger.error(f"❌ Seat {seat} error: {e}")

            if not raw:
                return None

            # Step 2: DEDUP — per badge name keep only highest-score seat
            best_for_badge: Dict[str, tuple] = {}
            for seat, det in raw.items():
                if det.name not in best_for_badge or det.match_score > best_for_badge[det.name][0]:
                    best_for_badge[det.name] = (det.match_score, seat, det)
            deduped: Dict[int, Detection] = {seat: det for _, seat, det in best_for_badge.values()}

            # Step 3: vote
            vote_count: Dict[str, int] = {}
            vote_score: Dict[str, float] = {}
            for seat, det in deduped.items():
                step = SEAT_STEP_MAP[seat]
                implied = SEAT_POSITION_CYCLE[(SEAT_POSITION_CYCLE.index(det.name) - step) % 6]
                vote_count[implied] = vote_count.get(implied, 0) + 1
                vote_score[implied] = vote_score.get(implied, 0.0) + det.match_score
                label = "DIRECT" if seat == 1 else f"seat{seat}"
                logger.info(f"        [{label}] {det.name}({det.match_score:.3f}) → vote {implied}")

            # Step 4: winner by most votes
            max_count = max(vote_count.values())
            top = [p for p, c in vote_count.items() if c == max_count]

            if len(top) == 1:
                hero_name = top[0]
                logger.info(f"    → {hero_name} ({max_count} votes, score={vote_score[hero_name]:.3f})")
            elif 1 in deduped:
                hero_name = deduped[1].name   # seat1 step=0: badge IS hero
                logger.info(f"    → TIE {top} → seat1 tiebreak: {hero_name}")
            else:
                hero_name = max(top, key=lambda p: vote_score[p])
                logger.info(f"    → TIE {top} → score tiebreak: {hero_name}")

            if not hero_name:
                return None

            # Build FULL positions dict (all 6 seats required by PositionService)
            hero_idx = SEAT_POSITION_CYCLE.index(hero_name)
            ALL_SEAT_STEPS = {1: 0, 2: 3, 3: 1, 4: 2, 5: 4, 6: 5}
            player_positions = {}
            for seat, step in ALL_SEAT_STEPS.items():
                if seat in detections:
                    player_positions[seat] = detections[seat]
                else:
                    expected_idx = (hero_idx + step) % 6
                    player_positions[seat] = Detection(SEAT_POSITION_CYCLE[expected_idx], None, NO_RECT, 0)

            return player_positions

        except Exception as e:
            logger.error(f"❌ Voting detection error: {e}")
            return None

    # ── Native detection fallback ─────────────────────────────────────

    @staticmethod
    def _detect_native_positions(cv2_image) -> Dict[int, Detection]:
        player_positions = {}
        for player_num, coords in PLAYER_POSITIONS.items():
            search_region = coords_to_search_region(coords['x'], coords['y'], coords['w'], coords['h'])
            try:
                detected = TemplateMatchService.find_positions(cv2_image, search_region)
                if detected:
                    player_positions[player_num] = detected[0]
                else:
                    player_positions[player_num] = Detection("NO", None, None, 1)
            except Exception as e:
                logger.error(f"❌ Native pos error P{player_num}: {e}")

        logger.info("    ✅ Found positions (native):")
        for pn, pos in player_positions.items():
            logger.info(f"        P{pn}: {pos.name}")
        return player_positions

    # ── Debug capture ─────────────────────────────────────────────────

    @staticmethod
    def _save_debug(cv2_image, positions=None):
        """Save debug captures with per-table cooldown."""
        try:
            global _DEBUG_LAST_SAVE
            now = time.time()
            try:
                tbl_key = int(cv2_image[10, 10, 0]) + int(cv2_image[50, 50, 0]) * 256
            except Exception:
                tbl_key = 0
            last = _DEBUG_LAST_SAVE.get(tbl_key, 0)
            if now - last < _DEBUG_INTERVAL:
                return
            _DEBUG_LAST_SAVE[tbl_key] = now
            _DEBUG_DIR.mkdir(parents=True, exist_ok=True)
            ts = int(now)
            cv2.imwrite(str(_DEBUG_DIR / f"table_{tbl_key}_{ts}.png"), cv2_image)
            hc = JUROJIN_POSITION_REGIONS[1]
            hero_crop = cv2_image[
                max(0, hc['y']):min(cv2_image.shape[0], hc['y'] + hc['h']),
                max(0, hc['x']):min(cv2_image.shape[1], hc['x'] + hc['w'])
            ]
            cv2.imwrite(str(_DEBUG_DIR / f"hero_{tbl_key}_{ts}.png"), hero_crop)
            pos_str = ""
            if positions:
                pos_str = " | " + ", ".join(f"S{k}={v.name}" for k, v in sorted(positions.items()))
            logger.info(f"    📸 Debug saved (tbl={tbl_key} ts={ts}){pos_str}")
        except Exception as e:
            logger.error(f"❌ Debug save error: {e}")

    # ── Other detection methods ───────────────────────────────────────

    @staticmethod
    def detect_player_cards(cv2_image) -> List[Detection]:
        return TemplateMatchService.find_player_cards(cv2_image)

    @staticmethod
    def detect_table_cards(cv2_image) -> List[Detection]:
        return TemplateMatchService.find_table_cards(cv2_image)

    @staticmethod
    def get_player_actions_detection(image: np.ndarray) -> Dict[int, List[Detection]]:
        player_actions = {}
        for player_id, region in ACTION_POSITIONS.items():
            search_region = coords_to_search_region(
                x=region[0], y=region[1], w=region[2], h=region[3],
            )
            actions = TemplateMatchService.find_jurojin_actions(image, search_region=search_region)
            player_actions[player_id] = actions
        return player_actions
