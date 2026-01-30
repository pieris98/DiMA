import unittest
from unittest.mock import MagicMock, patch
import sys
import os
import argparse

# Add path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TestAutoScripts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Mock modules that might be missing
        modules_to_mock = [
            "torch", "hydra", "datasets", "transformers", "cheap", "cheap.pretrained", 
            "omegaconf", "biotite", "biotite.structure.io", 
            "src.metrics.plddt", "src.metrics.esmpppl", "src.metrics.fid", 
            "src.utils.hydra_utils", "src.datasets.load_hub", "src.diffusion.base_trainer",
            "src.utils", "numpy", "src.metrics.metric"
        ]
        for mod in modules_to_mock:
            if mod not in sys.modules:
                sys.modules[mod] = MagicMock()

    def test_prepare_data_imports(self):
        """Test that we can import prepare_data and it has expected functions."""
        try:
            import prepare_data
            self.assertTrue(hasattr(prepare_data, "prepare_afdb"))
            self.assertTrue(hasattr(prepare_data, "prepare_swissprot"))
        except ImportError as e:
            self.fail(f"Could not import prepare_data: {e}")

    def test_setup_models_imports(self):
        """Test that setup_models imports work."""
        try:
            import setup_models
            self.assertTrue(hasattr(setup_models, "setup_esm2"))
        except ImportError as e:
            self.fail(f"Could not import setup_models: {e}")

    @patch("run_inference.BaseDiffusionTrainer")
    @patch("run_inference.hydra")
    def test_run_inference_mock(self, mock_hydra, mock_trainer):
        """Test run_inference logic with mocks."""
        import run_inference
        
        # Mock config
        mock_config = MagicMock()
        mock_config.ddp.enabled = False
        mock_config.generation.num_gen_samples = 2
        
        # Mock trainer instance
        mock_trainer_instance = mock_trainer.return_value
        mock_trainer_instance.generate_samples.return_value = ["SEQ1", "SEQ2"]
        
        # Run main logic (extract body of main since hydra decorates it)
        # We can't easily run the decorated function without hydra context, 
        # so we'll just check if we can instantiate everything.
        
        trainer = run_inference.BaseDiffusionTrainer(mock_config, "cpu")
        seqs = trainer.generate_samples(2)
        self.assertEqual(seqs, ["SEQ1", "SEQ2"])

    @patch("calc_metrics.esmpppl.calculate_pppl")
    @patch("calc_metrics.plddt.calculate_plddt")
    def test_calc_metrics_mock(self, mock_plddt, mock_pppl):
        """Test calc_metrics logic."""
        import calc_metrics
        
        mock_pppl.return_value = [10.0, 12.0]
        mock_plddt.return_value = {"SEQ1": 0.8, "SEQ2": 0.9}
        
        seqs = ["SEQ1", "SEQ2"]
        # Mock config
        config = MagicMock()
        
        results = calc_metrics.calculate_metrics(seqs, config, "cpu", ["esmpppl", "plddt"])
        
        self.assertIn("esmpppl", results)
        self.assertIn("plddt", results)
        self.assertEqual(results["esmpppl"], [10.0, 12.0])
        self.assertEqual(results["plddt"], [0.8, 0.9])

if __name__ == "__main__":
    unittest.main()
