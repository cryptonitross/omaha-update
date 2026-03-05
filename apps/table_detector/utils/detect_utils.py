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
    1: {'x': 300, 'y': 375, 'w': 40, 'h': 40},
    2: {'x': 35, 'y': 330, 'w': 40, 'h': 40},
    3: {'x': 35, 'y': 173, 'w': 40, 'h': 40},
    4: {'x': 297, 'y': 120, 'w': 40, 'h': 40},
    5: {'x': 562, 'y': 168, 'w': 40, 'h': 40},
    6: {'x': 565, 'y': 332, 'w': 40, 'h': 40}
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

# Confirmed clockwise step distance from hero (seat 1) to each seat
# Derived empirically from known game states: seat4=BB when hero=BTN, seat6=CO when hero=BTN
SEAT_STEP_MAP = {
    3: 1,  # left-bottom: 1 step clockwise from hero
    4: 2,  # top-center:  2 steps
    2: 3,  # left-top:    3 steps
    5: 4,  # top-right:   4 steps
    6: 5,  # bottom-right: 5 steps
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
        Scan all seats 2-6 for Jurojin badges.
        Each detected badge implies a hero position via SEAT_STEP_MAP.
        Most voted hero position wins.
        """
        try:
            h, w = cv2_image.shape[:2]
            NO_RECT = (0, 0, 0, 0)

            # Collect votes: hero_position → list of (score, seat, badge_name)
            votes: Dict[str, List[tuple]] = {p: [] for p in VALID_POSITIONS}
            detected_per_seat: Dict[int, Detection] = {}

            for seat, step in SEAT_STEP_MAP.items():
                coords = JUROJIN_POSITION_REGIONS[seat]
                search_region = coords_to_search_region(
                    coords['x'], coords['y'], coords['w'], coords['h'],
                    image_width=w, image_height=h
                )

                try:
                    matches = TemplateMatchService.find_positions(cv2_image, search_region)
                    if matches:
                        best = matches[0]
                        if best.name in VALID_POSITIONS:
                            detected_per_seat[seat] = best
                            # Compute implied hero position
                            badge_idx = SEAT_POSITION_CYCLE.index(best.name)
                            hero_idx = (badge_idx - step) % 6
                            implied_hero = SEAT_POSITION_CYCLE[hero_idx]
                            votes[implied_hero].append((best.match_score, seat, best.name))
                            logger.info(f"        Seat {seat}: {best.name} (score={best.match_score:.3f}) → vote {implied_hero}")
                        else:
                            logger.info(f"        Seat {seat}: NO match")
                    else:
                        logger.info(f"        Seat {seat}: NO match")
                except Exception as e:
                    logger.error(f"❌ Seat {seat} error: {e}")

            if not any(votes[p] for p in VALID_POSITIONS):
                return None

            # Tally votes: sum of scores per candidate hero position
            tally = {pos: sum(s for s, _, _ in v) for pos, v in votes.items() if v}
            if not tally:
                return None

            # Pick winner
            hero_name = max(tally, key=tally.get)
            top_score = tally[hero_name]
            logger.info(f"    Vote tally: {', '.join(f'{k}={v:.2f}' for k,v in sorted(tally.items(), key=lambda x:-x[1]))}")
            logger.info(f"    → Hero winner: {hero_name} (score_sum={top_score:.3f})")

            # Build full positions dict
            player_positions = {1: Detection(hero_name, None, NO_RECT, 1.0)}
            hero_idx = SEAT_POSITION_CYCLE.index(hero_name)

            for seat, step in SEAT_STEP_MAP.items():
                if seat in detected_per_seat:
                    player_positions[seat] = detected_per_seat[seat]
                else:
                    expected_idx = (hero_idx + step) % 6
                    expected_name = SEAT_POSITION_CYCLE[expected_idx]
                    player_positions[seat] = Detection(expected_name, None, NO_RECT, 0)

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
