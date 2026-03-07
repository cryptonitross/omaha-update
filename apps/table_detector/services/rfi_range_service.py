import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from loguru import logger


@dataclass
class RfiResult:
    """Result of an RFI range lookup."""
    combo: str
    position: str
    weight: float
    ev: float
    action: str  # "RAISE" or "FOLD"

    def to_dict(self) -> dict:
        return {
            'combo': self.combo,
            'position': self.position,
            'weight': self.weight,
            'ev': self.ev,
            'action': self.action
        }


class RfiRangeService:
    """
    Service to look up RFI (Raise First In) ranges for PLO.

    Loads CSV files for each position and provides O(1) lookup by combo string.
    CSV format: combo,weight,ev (e.g., "AsKsQsJs,1,0.13")

    RFI applies only when all players before hero have folded:
    - EP: first to act (no folds needed)
    - MP: EP folded
    - CO: EP and MP folded
    - BTN: EP, MP, and CO folded
    """

    # Position -> CSV filename mapping
    POSITION_FILES = {
        'EP':  'PLO500v2_100_6_EP_POT.csv',
        'MP':  'PLO500v2_100_6_F_MP_POT.csv',
        'CO':  'PLO500v2_100_6_F-F_CO_POT.csv',
        'BTN': 'PLO500v2_100_6_F-F-F_BTN_POT.csv',
    }

    # Positions that can RFI (SB and BB cannot RFI in standard play)
    RFI_POSITIONS = {'EP', 'MP', 'CO', 'BTN'}

    def __init__(self, resources_dir: str):
        self._ranges: Dict[str, Dict[str, tuple]] = {}  # position -> {combo -> (weight, ev)}
        self._resources_dir = Path(resources_dir)
        self._loaded = False

    def load_ranges(self):
        """Load all RFI range CSV files into memory."""
        if self._loaded:
            return

        ranges_dir = self._resources_dir / "rfi_ranges"
        if not ranges_dir.exists():
            logger.warning(f"⚠️ RFI ranges directory not found: {ranges_dir}")
            return

        for position, filename in self.POSITION_FILES.items():
            filepath = ranges_dir / filename
            if not filepath.exists():
                logger.warning(f"⚠️ RFI range file not found: {filepath}")
                continue

            try:
                combos = {}
                with open(filepath, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        combo = row['combo']
                        weight = float(row['weight'])
                        ev = float(row['ev'])
                        combos[combo] = (weight, ev)

                self._ranges[position] = combos
                logger.info(f"✅ Loaded {len(combos)} RFI combos for {position}")

            except Exception as e:
                logger.error(f"❌ Error loading RFI range for {position}: {e}")

        self._loaded = True
        logger.info(f"✅ RFI ranges loaded for positions: {list(self._ranges.keys())}")

    def check_rfi(self, combo: str, position: str) -> Optional[RfiResult]:
        """
        Check if a combo should be raised or folded for RFI at a given position.

        Args:
            combo: 4-card combo string like "AsKsQsJs"
            position: Hero position (EP, MP, CO, BTN)

        Returns:
            RfiResult with action RAISE/FOLD, or None if position not applicable
        """
        if not self._loaded:
            self.load_ranges()

        position = position.upper()

        if position not in self.RFI_POSITIONS:
            return None

        if position not in self._ranges:
            logger.warning(f"No RFI range data for position {position}")
            return None

        range_data = self._ranges[position]

        # Normalize the combo - try the exact combo first
        normalized = self._normalize_combo(combo)

        if normalized in range_data:
            weight, ev = range_data[normalized]
            # weight > 0 means it's in the raising range
            action = "RAISE" if weight > 0 else "FOLD"
            return RfiResult(
                combo=normalized,
                position=position,
                weight=weight,
                ev=ev,
                action=action
            )

        # Combo not found in range = FOLD
        return RfiResult(
            combo=normalized,
            position=position,
            weight=0.0,
            ev=0.0,
            action="FOLD"
        )

    def _normalize_combo(self, combo: str) -> str:
        """
        Normalize a combo string to match CSV format.

        Input formats:
        - "AsKsQsJs" (already correct)
        - "as ks qs js" (lowercase with spaces)
        - "ASKSJSQS" (needs parsing)

        CSV format uses: rank+suit pairs like "AsKsQsJs"
        """
        # Remove spaces
        combo = combo.replace(" ", "")

        # If already in correct format (8 chars, alternating rank+suit)
        if len(combo) == 8:
            # Ensure proper capitalization: rank uppercase, suit lowercase
            result = ""
            for i in range(0, 8, 2):
                rank = combo[i].upper()
                suit = combo[i + 1].lower()
                result += rank + suit
            return result

        return combo

    def is_rfi_position(self, position: str) -> bool:
        """Check if the position can make an RFI decision."""
        return position.upper() in self.RFI_POSITIONS
