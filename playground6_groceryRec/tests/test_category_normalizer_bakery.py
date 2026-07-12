"""Bakery/prepared category-gap fix (2026-07-12): shelf-stable bakery + baked sweets
must win over incidental produce nouns. Pins the live eval set (must-fix) + the
must-not-regress guards, incl. the substring-safety cases that drove the keyword
choices (buns-not-bun, no-bare-pie, sweet-removed). Byte-equivalent JS map lives in
trepo-quick-ack/lib/data-access.mjs — keep both in lock-step.
"""
import importlib.util
import unittest
from pathlib import Path

CN_PATH = Path(__file__).resolve().parents[1] / "kitchen_api" / "category_normalizer.py"


def _load():
    spec = importlib.util.spec_from_file_location("category_normalizer", CN_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


class BakeryCategoryTests(unittest.TestCase):
    def setUp(self):
        self.norm = _load().normalize_kitchen_category

    def cat(self, name):
        # derive-from-name path (raw=name, scanned) — matches the backfill + freetext write path
        return self.norm(name, None, name)

    def test_must_fix_bakery_not_produce(self):
        for name, want in {
            "Potato Bread": "pantry",
            "L'Oven Fresh Potato Bread": "pantry",
            "Blueberry Bread": "pantry",
            "Blueberry Muffins": "snacks_sweets",
            "Mixed Berry Biscuits": "snacks_sweets",
            "TJ Green Onion Pancakes": "snacks_sweets",
        }.items():
            self.assertEqual(self.cat(name), want, f"{name} should be {want}")

    def test_must_not_regress(self):
        for name, want in {
            "Bananas Bunch": "produce",       # 'buns' not 'bun' — must not catch 'bunch'
            "Cilantro Bunch": "produce",
            "Strawberry Yogurt": "dairy_eggs",
            "Chicken Broth": "pantry",
            "Sweet Potato": "produce",        # bare 'sweet' removed from snacks
            "Sweet Corn": "produce",          # 'corn' added to produce
            "Sweet Onion": "produce",
            "Chicken Pieces": "meat_seafood", # bare 'pie' omitted — must not catch 'pieces'
            "Strawberries": "produce",        # 'berries' added
            "Cornbread": "pantry",            # 'bread' wins over 'corn'
            "Popcorn": "snacks_sweets",
            "Chicken Sausage": "meat_seafood",
        }.items():
            self.assertEqual(self.cat(name), want, f"{name} should stay {want}")


if __name__ == "__main__":
    unittest.main()
