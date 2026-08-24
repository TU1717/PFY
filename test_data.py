import tempfile
import unittest
from pathlib import Path

import numpy as np

from data import InteractionDataset


class DataContractTest(unittest.TestCase):
    def make_dataset(self, root):
        (root / "train_data.txt").write_text(
            "0 0 1\n1 2 3 4 5\n", encoding="utf-8"
        )
        (root / "valid_data.txt").write_text("0 2\n1 6\n", encoding="utf-8")
        (root / "test.txt").write_text("0 3\n1 7\n", encoding="utf-8")
        return InteractionDataset.from_directory(root)

    def test_effective_p_groups_cover_every_user(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = self.make_dataset(Path(temporary))
            groups = dataset.sample_positive_groups(
                np.asarray([0, 1]), nominal_p=3, rng=np.random.default_rng(17)
            )
            self.assertEqual(set(groups), {2, 3})
            covered = np.concatenate([group.rows for group in groups.values()])
            self.assertEqual(set(covered.tolist()), {0, 1})
            self.assertEqual(groups[2].positives.shape, (1, 2))
            self.assertEqual(groups[3].positives.shape, (1, 3))
            for group in groups.values():
                for row in group.positives:
                    self.assertEqual(len(row), len(set(row.tolist())))

            audit = dataset.effective_p_audit(3)
            self.assertEqual(audit["min"], 2)
            self.assertEqual(audit["max"], 3)
            self.assertEqual(audit["users_saturated_below_nominal_p"], 1)

    def test_split_leakage_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "train_data.txt").write_text("0 1 2\n", encoding="utf-8")
            (root / "valid_data.txt").write_text("0 2\n", encoding="utf-8")
            (root / "test.txt").write_text("0 3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "leakage"):
                InteractionDataset.from_directory(root)


if __name__ == "__main__":
    unittest.main()

