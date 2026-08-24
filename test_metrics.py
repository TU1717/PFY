import math
import unittest

from metrics import metrics_for_user


class MetricTest(unittest.TestCase):
    def test_two_relevant_items(self):
        result = metrics_for_user([4, 2, 7, 9], [2, 9], [2, 4])
        self.assertAlmostEqual(result["precision@2"], 0.5)
        self.assertAlmostEqual(result["recall@2"], 0.5)
        self.assertAlmostEqual(
            result["ndcg@2"], (1.0 / math.log2(3.0)) / (1.0 + 1.0 / math.log2(3.0))
        )
        self.assertAlmostEqual(result["mrr_numerator@2"], 0.5)
        self.assertAlmostEqual(result["precision@4"], 0.5)
        self.assertAlmostEqual(result["recall@4"], 1.0)
        self.assertAlmostEqual(result["mrr_numerator@4"], 0.75)


if __name__ == "__main__":
    unittest.main()

