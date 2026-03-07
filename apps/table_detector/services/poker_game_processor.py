import os

from loguru import logger

from shared.domain.game_snapshot import GameSnapshot
from table_detector.domain.captured_window import CapturedWindow
from table_detector.domain.omaha_engine import OmahaEngine, OmahaEngineException
from table_detector.services.position_service import PositionService
from table_detector.utils.detect_utils import DetectUtils
from table_detector.utils.drawing_utils import save_detection_result
from table_detector.services.rfi_range_service import RfiRangeService


class PokerGameProcessor:

    _rfi_service = None

    @classmethod
    def _get_rfi_service(cls):
        if cls._rfi_service is None:
            resources_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'resources')
            cls._rfi_service = RfiRangeService(resources_dir)
        return cls._rfi_service

    def __init__(self):
        self.debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'

    def process_window(self, captured_image: CapturedWindow, timestamp_folder) -> GameSnapshot:
        """Process captured image and return GameSnapshot."""
        window_name = captured_image.window_name

        self.validate_image(captured_image)

        game_snapshot = PokerGameProcessor.create_game_snapshot(captured_image.get_cv2_image())
        if self.debug_mode:
            save_detection_result(timestamp_folder, captured_image, game_snapshot)

        return game_snapshot

    def validate_image(self, captured_image: CapturedWindow):
        # Add size validation
        image_width, image_height = captured_image.get_size()
        if image_width != 784 or image_height != 584:
            raise ValueError(
                f"Неправильный размер картинки для окна {captured_image.window_name}. Ожидаеться: 784x584, Реальный размер: {image_width}x{image_height}. Скорее всего нужно поменять Jurojin Layout, размер окна в Jurojin должен быть: 770x577")

    @staticmethod
    def create_game_snapshot(cv2_image):
        player_cards_detections = DetectUtils.detect_player_cards(cv2_image)
        table_cards_detections = DetectUtils.detect_table_cards(cv2_image)
        position_detections = DetectUtils.detect_positions(cv2_image)
        action_detections = DetectUtils.get_player_actions_detection(cv2_image)

        moves_data = None
        try:
            recovered_positions = PositionService.get_positions(position_detections)
            position_actions = OmahaEngine.convert_to_position_actions(action_detections, recovered_positions)
            game = OmahaEngine(len(position_actions))
            game.simulate_all_moves(position_actions)
            moves_data = game.get_moves_by_street()
            logger.info(moves_data)
        except Exception as e:
            logger.debug(f"Expected exception: {e}")

        # RFI check: determine hero position and range action
        hero_position = None
        rfi_action = None
        if 1 in position_detections and position_detections[1].name != "NO":
            hero_position = position_detections[1].name
            if player_cards_detections and len(player_cards_detections) == 4:
                card_names = [c.template_name for c in player_cards_detections]
                combo_str = "".join(card_names)
                try:
                    rfi_svc = PokerGameProcessor._get_rfi_service()
                    rfi_result = rfi_svc.check_rfi(combo_str, hero_position)
                    if rfi_result:
                        rfi_action = rfi_result.action
                        logger.info(f"    RFI: {hero_position} {combo_str} → {rfi_action}")
                except Exception as e:
                    logger.debug(f"    RFI check error: {e}")

        return GameSnapshot(
            player_cards=player_cards_detections,
            table_cards=table_cards_detections,
            positions=position_detections,
            actions=action_detections,
            moves=moves_data,
            hero_position=hero_position,
            rfi_action=rfi_action
        )
