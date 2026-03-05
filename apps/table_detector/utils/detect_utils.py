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

# Jurojin badge search regions per seat (wider to catch badge placement variations)
# NOTE: seat 2 = left-top, seat 3 = left-bottom (matches original working code)
JUROJIN_POSITION_REGIONS = {
    1: {'x': 280, 'y': 355, 'w': 65, 'h': 65},  # hero bottom center
    2: {'x': 15,  'y': 150, 'w': 70, 'h': 70},  # left top
    3: {'x': 15,  'y': 310, 'w': 70, 'h': 70},  # left bottom
    4: {'x': 275, 'y': 100, 'w': 65, 'h': 65},  # top center
    5: {'x': 540, 'y': 145, 'w': 65, 'h': 65},  # top right
    6: {'x': 555, 'y': 330, 'w': 70, 'h': 70},  # bottom right
}

POSITION_MARGIN = 10
IMAGE_WIDTH = 784
IMAGE_HEIGHT = 584
VALID_POSITIONS = {'BB', 'SB', 'BTN', 'EP', 'MP', 'CO'}
SEAT_POSITION_CYCLE = ['BTN', 'SB', 'BB', 'EP', 'MP', 'CO']


class DetectUtils:

    # ── Main entry point ──────────────────────────────────────────────

    @staticmethod
    def detect_positions(cv2_image) -> Dict[int, Detection]:
        """
        Detect player positions using Jurojin overlay badge template matching (primary)
        with fallback to native client position labels.
        """
        t0 = time.time()
        try:
            # PRIMARY: Jurojin template matching per seat
            jurojin_result = DetectUtils._detect_jurojin_positions(cv2_image)
            if jurojin_result:
                ms = (time.time() - t0) * 1000
                summary = " | ".join(f"S{k}={v.name}" for k, v in sorted(jurojin_result.items()))
                logger.info(f"  ── Position [{ms:.0f}ms]: {summary} ──")
                DetectUtils._save_debug(cv2_image, jurojin_result)
                return jurojin_result

            # LAST RESORT: native template matching
            logger.info("  ── Jurojin not found, native fallback ──")
            DetectUtils._save_debug(cv2_image)
            return DetectUtils._detect_native_positions(cv2_image)

        except Exception as e:
            logger.error(f"❌ Position detection error: {e}")
            return {}

    # ── Jurojin template matching per seat ────────────────────────────

    @staticmethod
    def _detect_jurojin_positions(cv2_image) -> Optional[Dict[int, Detection]]:
        """
        Detect positions from Jurojin overlay colored circle badges via template matching.
        Hero (seat 1) is always at bottom center — check it directly first.
        If hero badge not found, deduce from other seats.
        Returns None if no positions detected at all.
        """
        try:
            h, w = cv2_image.shape[:2]
            NO_RECT = (0, 0, 0, 0)
            player_positions = {}

            # PRIMARY: check hero badge directly at seat 1 (always bottom center)
            hero_coords = JUROJIN_POSITION_REGIONS[1]
            hero_region = coords_to_search_region(
                hero_coords['x'], hero_coords['y'], hero_coords['w'], hero_coords['h'],
                image_width=w, image_height=h
            )
            hero_detected = TemplateMatchService.find_positions(cv2_image, hero_region)
            if hero_detected:
                best = hero_detected[0]
                player_positions[1] = best
                logger.info(f"        Seat 1 (HERO): {best.name} (score={best.match_score:.3f}) ✅ direct")
            else:
                player_positions[1] = Detection("NO", (0, 0), NO_RECT, 0)
                logger.info(f"        Seat 1 (HERO): NO match")

            # SECONDARY: detect seats 2-6 for additional context / deduction
            for seat in range(2, 7):
                coords = JUROJIN_POSITION_REGIONS[seat]
                search_region = coords_to_search_region(
                    coords['x'], coords['y'], coords['w'], coords['h'],
                    image_width=w, image_height=h
                )

                try:
                    detected = TemplateMatchService.find_positions(cv2_image, search_region)

                    if detected:
                        best = detected[0]
                        player_positions[seat] = best
                        logger.info(f"        Seat {seat}: {best.name} (score={best.match_score:.3f})")
                    else:
                        player_positions[seat] = Detection("NO", (0, 0), NO_RECT, 0)
                        logger.info(f"        Seat {seat}: NO match")

                except Exception as e:
                    logger.error(f"❌ Jurojin error seat {seat}: {e}")
                    player_positions[seat] = Detection("NO", (0, 0), NO_RECT, 0)

            # Validate: count real (non-NO) detections
            real = [(s, d) for s, d in player_positions.items() if d.name in VALID_POSITIONS]
            real_count = len(real)

            if real_count < 1:
                logger.info(f"    ⚠️ Jurojin: only {real_count} positions detected, skipping")
                return None

            # Validate: no duplicates (duplicates = overlay absent / false matches)
            names = [d.name for _, d in real]
            counts = Counter(names)
            dupes = {n: c for n, c in counts.items() if c > 1}
            if dupes:
                logger.warning(f"    ⚠️ Jurojin: duplicate positions {dupes} — overlay likely absent")
                return None

            # If hero position not found directly, deduce from other seats
            if player_positions[1].name == "NO":
                hero_name = DetectUtils._deduce_hero(
                    {s: d for s, d in player_positions.items() if s != 1}
                )
                if hero_name:
                    player_positions[1] = Detection(hero_name, None, NO_RECT, 1.0)
                    logger.info(f"    Hero deduced: {hero_name}")

            logger.info(f"    🎯 Using Jurojin overlay positions ({real_count} detected)")
            return player_positions

        except Exception as e:
            logger.error(f"❌ Error detecting Jurojin positions: {e}")
            return None

    # ── Hero deduction from other seats ───────────────────────────────

    @staticmethod
    def _deduce_hero(positions: Dict[int, Detection]) -> Optional[str]:
        """Deduce hero position from another seat's known position using cycle logic."""
        for seat, det in positions.items():
            if det.name not in VALID_POSITIONS:
                continue
            idx = SEAT_POSITION_CYCLE.index(det.name)
            hero_idx = (idx - (seat - 1)) % 6
            hero = SEAT_POSITION_CYCLE[hero_idx]
            logger.info(f"    Deduce: S{seat}={det.name} → hero={hero}")
            return hero
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
