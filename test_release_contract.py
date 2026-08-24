import json
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ReleaseContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "config.json").open("r", encoding="utf-8") as handle:
            cls.config = json.load(handle)

    def test_shared_protocol(self):
        defaults = self.config["defaults"]
        self.assertEqual(defaults["seeds"], [2024, 2025, 2026, 2027, 2028])
        self.assertEqual(defaults["mc_samples"], 32)
        self.assertEqual(defaults["mc_chunk"], 4)
        self.assertEqual(defaults["dim"], 64)
        self.assertEqual(defaults["train_batch"], 512)
        self.assertEqual(defaults["epochs"], 300)

    def test_dataset_specific_presets_are_not_published(self):
        self.assertEqual(set(self.config), {"defaults"})

    def test_readme_is_plain_ascii(self):
        raw = (ROOT / "README.md").read_bytes()
        self.assertNotIn(b"\r\n", raw)
        raw.decode("ascii")
        text = raw.decode("ascii")
        unsupported = "\\" + "operator" + "name"
        self.assertNotIn(unsupported, text)
        self.assertNotIn("```math", text)
        equation_blocks = re.findall(r"(?m)^\$\$\n[^\n]+\n\$\$$", text)
        self.assertEqual(len(equation_blocks), 4)
        self.assertEqual(text.splitlines().count("$$"), 8)
        self.assertIn("](paradigm.png)", text)
        self.assertEqual(text.lower().count(".png"), 1)
        paradigm = (ROOT / "paradigm.png").read_bytes()
        self.assertTrue(paradigm.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertEqual(list(ROOT.glob("formula_*.png")), [])
        self.assertNotIn("paper settings", text.lower())


if __name__ == "__main__":
    unittest.main()
