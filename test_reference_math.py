import unittest

import numpy as np


class ReferenceMathTest(unittest.TestCase):
    def test_hard_top_p_objective_and_gradient(self):
        scores = np.asarray(
            [[1.2, -0.4, 0.1, 0.7], [0.3, 0.9, -0.2, 0.5]],
            dtype=np.float64,
        )
        positives = np.asarray([[0, 2], [1, 3]], dtype=np.int64)
        noise = np.asarray(
            [
                [[0.0, 0.2, -0.1, 0.5], [0.1, -0.2, 0.3, 0.0]],
                [[-0.2, 0.4, 0.1, 0.0], [0.5, 0.0, -0.1, 0.2]],
                [[0.3, -0.1, 0.0, 0.2], [-0.2, 0.1, 0.4, 0.0]],
            ],
            dtype=np.float64,
        )
        scale = 0.4
        p = 2
        perturbed = scores[None, :, :] + scale * noise
        top_indices = np.argpartition(perturbed, -p, axis=-1)[..., -p:]
        choices = np.zeros_like(perturbed)
        np.put_along_axis(choices, top_indices, 1.0, axis=-1)

        target = np.zeros_like(scores)
        np.put_along_axis(target, positives, 1.0, axis=-1)
        expected_gradient = (choices.mean(axis=0) - target) / scores.shape[0]
        selected_values = np.take_along_axis(perturbed, top_indices, axis=-1)
        expected_loss = np.mean(
            selected_values.sum(axis=-1).mean(axis=0)
            - np.take_along_axis(scores, positives, axis=-1).sum(axis=-1)
        )

        self.assertTrue(np.isfinite(expected_loss))
        np.testing.assert_allclose(choices.sum(axis=-1), p)
        np.testing.assert_allclose(expected_gradient.sum(axis=-1), 0.0)
        self.assertFalse(np.allclose(expected_gradient, 0.0))


if __name__ == "__main__":
    unittest.main()

