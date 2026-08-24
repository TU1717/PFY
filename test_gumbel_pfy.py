import unittest
import inspect


try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from losses import full_catalog_gumbel_pfy_loss


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GumbelPFYTest(unittest.TestCase):
    def setUp(self):
        self.scores = torch.tensor(
            [[1.2, -0.4, 0.1, 0.7], [0.3, 0.9, -0.2, 0.5]],
            dtype=torch.float64,
            requires_grad=True,
        )
        self.positives = torch.tensor([[0, 2], [1, 3]])
        self.noise = torch.tensor(
            [
                [[0.0, 0.2, -0.1, 0.5], [0.1, -0.2, 0.3, 0.0]],
                [[-0.2, 0.4, 0.1, 0.0], [0.5, 0.0, -0.1, 0.2]],
                [[0.3, -0.1, 0.0, 0.2], [-0.2, 0.1, 0.4, 0.0]],
            ],
            dtype=torch.float64,
        )

    def direct_objective(self):
        perturbed = self.scores.unsqueeze(0) + 0.4 * self.noise
        values, indices = torch.topk(perturbed, k=2, dim=-1, sorted=False)
        target_sum = self.scores.gather(1, self.positives).sum(dim=1)
        return (values.sum(dim=-1).mean(dim=0) - target_sum).mean(), indices

    def test_public_default_uses_32_samples(self):
        signature = inspect.signature(full_catalog_gumbel_pfy_loss)
        self.assertEqual(signature.parameters["mc_samples"].default, 32)
        self.assertEqual(signature.parameters["mc_chunk"].default, 4)

    def test_scalar_and_gradient_match_direct_top_p(self):
        actual, diagnostics = full_catalog_gumbel_pfy_loss(
            self.scores,
            self.positives,
            p=2,
            mc_samples=3,
            mc_chunk=2,
            gumbel_scale=0.4,
            fixed_noise=self.noise,
        )
        expected, _ = self.direct_objective()
        actual_gradient = torch.autograd.grad(actual, self.scores, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, self.scores)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_gradient, expected_gradient)
        self.assertEqual(diagnostics["selection_mass"], 2.0)

    def test_chunking_does_not_change_result(self):
        values = []
        gradients = []
        for chunk in (1, 2, 3, 9):
            scores = self.scores.detach().clone().requires_grad_(True)
            value, _ = full_catalog_gumbel_pfy_loss(
                scores,
                self.positives,
                p=2,
                mc_samples=3,
                mc_chunk=chunk,
                gumbel_scale=0.4,
                fixed_noise=self.noise,
            )
            values.append(value.detach())
            gradients.append(torch.autograd.grad(value, scores)[0])
        for value, gradient in zip(values[1:], gradients[1:]):
            torch.testing.assert_close(value, values[0])
            torch.testing.assert_close(gradient, gradients[0])

    def test_duplicate_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "distinct"):
            full_catalog_gumbel_pfy_loss(
                self.scores,
                torch.tensor([[0, 0], [1, 3]]),
                p=2,
                mc_samples=3,
                fixed_noise=self.noise,
            )


if __name__ == "__main__":
    unittest.main()
