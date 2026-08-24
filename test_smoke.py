import argparse
import math
import tempfile
import unittest
from pathlib import Path


try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from trainer import aggregate_results, run_experiment


@unittest.skipIf(torch is None, "PyTorch is not installed")
class SmokeTrainingTest(unittest.TestCase):
    def test_aggregation_uses_sample_standard_deviation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = argparse.Namespace(
                config="config.json",
                dataset="Toy",
                data_dir="unused",
                output_dir=temporary,
                device="cpu",
                backbone="mf",
                seeds=(1, 2),
                cutoffs=(5,),
            )
            results = [
                {
                    "seed": 1,
                    "best_epoch": 1,
                    "best_validation_ndcg@5": 0.1,
                    "test_metrics": {"ndcg@5": 1.0},
                },
                {
                    "seed": 2,
                    "best_epoch": 1,
                    "best_validation_ndcg@5": 0.2,
                    "test_metrics": {"ndcg@5": 3.0},
                },
            ]
            aggregate = aggregate_results(args, Path(temporary), results)
            self.assertAlmostEqual(
                aggregate["metrics"]["ndcg@5"]["std"], math.sqrt(2.0)
            )

    def test_one_epoch_mf_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data_dir = root / "data"
            output_dir = root / "output"
            data_dir.mkdir()
            (data_dir / "train_data.txt").write_text(
                "0 0 1\n1 2 3\n2 4 5\n", encoding="utf-8"
            )
            (data_dir / "valid_data.txt").write_text(
                "0 2\n1 4\n2 6\n", encoding="utf-8"
            )
            (data_dir / "test.txt").write_text(
                "0 3\n1 5\n2 7\n", encoding="utf-8"
            )
            args = argparse.Namespace(
                config="config.json",
                dataset="Toy",
                data_dir=str(data_dir),
                output_dir=str(output_dir),
                backbone="mf",
                seeds=(2024,),
                device="cpu",
                epochs=1,
                eval_every=1,
                patience=1,
                train_batch=2,
                eval_batch=2,
                cutoffs=(1, 2, 5),
                lr=0.01,
                weight_decay=0.0,
                gradient_clip=10.0,
                dim=8,
                no_normalize=False,
                lightgcn_layers=2,
                xsimgcl_layers=3,
                p=3,
                gumbel_scale=0.2,
                mc_samples=2,
                mc_chunk=1,
                cl_rate=0.05,
                xsimgcl_eps=0.1,
                cl_temp=0.1,
                cl_layer=0,
            )
            result = run_experiment(args)
            self.assertEqual(result["status"], "FINAL_SUCCESS")
            self.assertEqual(result["num_seeds"], 1)
            self.assertEqual(result["std_definition"].split()[0], "sample")
            self.assertTrue((output_dir / "seed_2024" / "final_summary.json").is_file())
            self.assertTrue((output_dir / "final_aggregate.csv").is_file())


if __name__ == "__main__":
    unittest.main()
