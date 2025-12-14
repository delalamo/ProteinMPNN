"""
Unit tests comparing the refactored proteinmpnn CLI output against the original ProteinMPNN.

These tests verify that the modular refactored code produces identical sequences,
scores, and recovery values as the original implementation.
"""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class TestCLIOutputMatch(unittest.TestCase):
    """Test that new CLI produces identical output to original ProteinMPNN."""

    @classmethod
    def setUpClass(cls):
        """Set up paths for testing."""
        cls.repo_root = Path(__file__).parent.parent
        cls.inputs_dir = cls.repo_root / "inputs"
        cls.original_script = cls.repo_root / "protein_mpnn_run.py"

    def _run_new_cli(self, args: list, output_dir: str) -> str:
        """Run the new modular CLI."""
        cmd = [
            "python", "-m", "proteinmpnn.cli",
            "--out_folder", output_dir,
        ] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.repo_root)
        )
        return result.stdout + result.stderr

    def _run_original(self, args: list, output_dir: str) -> str:
        """Run the original protein_mpnn_run.py."""
        cmd = [
            "python", str(self.original_script),
            "--out_folder", output_dir,
        ] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(self.repo_root)
        )
        return result.stdout + result.stderr

    def _parse_fasta(self, fasta_path: str) -> list:
        """Parse FASTA file and return list of (header, sequence) tuples."""
        sequences = []
        current_header = None
        current_seq = []

        with open(fasta_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('>'):
                    if current_header is not None:
                        sequences.append((current_header, ''.join(current_seq)))
                    current_header = line[1:]
                    current_seq = []
                else:
                    current_seq.append(line)
            if current_header is not None:
                sequences.append((current_header, ''.join(current_seq)))

        return sequences

    def _extract_scores(self, header: str) -> dict:
        """Extract score values from FASTA header."""
        scores = {}
        parts = header.split(', ')
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                # Skip git_hash as it will differ
                if key != 'git_hash':
                    try:
                        scores[key] = float(value)
                    except ValueError:
                        scores[key] = value
        return scores

    def _compare_outputs(self, new_dir: str, orig_dir: str, pdb_name: str):
        """Compare FASTA outputs from both versions."""
        new_fasta = os.path.join(new_dir, "seqs", f"{pdb_name}.fa")
        orig_fasta = os.path.join(orig_dir, "seqs", f"{pdb_name}.fa")

        self.assertTrue(os.path.exists(new_fasta), f"New output not found: {new_fasta}")
        self.assertTrue(os.path.exists(orig_fasta), f"Original output not found: {orig_fasta}")

        new_seqs = self._parse_fasta(new_fasta)
        orig_seqs = self._parse_fasta(orig_fasta)

        self.assertEqual(len(new_seqs), len(orig_seqs), "Number of sequences differs")

        for i, ((new_header, new_seq), (orig_header, orig_seq)) in enumerate(zip(new_seqs, orig_seqs)):
            # Compare sequences
            self.assertEqual(new_seq, orig_seq, f"Sequence {i} differs")

            # Compare scores (excluding git_hash)
            new_scores = self._extract_scores(new_header)
            orig_scores = self._extract_scores(orig_header)

            for key in orig_scores:
                if key in new_scores:
                    if isinstance(orig_scores[key], float):
                        self.assertAlmostEqual(
                            new_scores[key],
                            orig_scores[key],
                            places=4,
                            msg=f"Score {key} differs in sequence {i}"
                        )
                    else:
                        self.assertEqual(
                            new_scores[key],
                            orig_scores[key],
                            msg=f"Value {key} differs in sequence {i}"
                        )

    def test_example_1_monomer(self):
        """Test single PDB monomer design (example 1)."""
        pdb_path = str(self.inputs_dir / "PDB_monomers" / "pdbs" / "5L33.pdb")

        with tempfile.TemporaryDirectory() as new_dir, \
             tempfile.TemporaryDirectory() as orig_dir:

            args = [
                "--pdb_path", pdb_path,
                "--num_seq_per_target", "2",
                "--sampling_temp", "0.1",
                "--seed", "37",
                "--batch_size", "1"
            ]

            self._run_new_cli(args, new_dir)
            self._run_original(args, orig_dir)

            self._compare_outputs(new_dir, orig_dir, "5L33")

    def test_example_3_complex(self):
        """Test PDB complex with multiple chains (example 3)."""
        pdb_path = str(self.inputs_dir / "PDB_complexes" / "pdbs" / "3HTN.pdb")

        with tempfile.TemporaryDirectory() as new_dir, \
             tempfile.TemporaryDirectory() as orig_dir:

            args = [
                "--pdb_path", pdb_path,
                "--pdb_path_chains", "A B",
                "--num_seq_per_target", "2",
                "--sampling_temp", "0.1",
                "--seed", "37",
                "--batch_size", "1"
            ]

            self._run_new_cli(args, new_dir)
            self._run_original(args, orig_dir)

            self._compare_outputs(new_dir, orig_dir, "3HTN")

    def test_different_temperature(self):
        """Test with different sampling temperature."""
        pdb_path = str(self.inputs_dir / "PDB_monomers" / "pdbs" / "5L33.pdb")

        with tempfile.TemporaryDirectory() as new_dir, \
             tempfile.TemporaryDirectory() as orig_dir:

            args = [
                "--pdb_path", pdb_path,
                "--num_seq_per_target", "2",
                "--sampling_temp", "0.2",
                "--seed", "42",
                "--batch_size", "1"
            ]

            self._run_new_cli(args, new_dir)
            self._run_original(args, orig_dir)

            self._compare_outputs(new_dir, orig_dir, "5L33")


class TestCLIImports(unittest.TestCase):
    """Test that all CLI components import correctly."""

    def test_cli_import(self):
        """Test main CLI module imports."""
        from proteinmpnn.cli import main
        self.assertTrue(callable(main))

    def test_core_imports(self):
        """Test core module imports."""
        from proteinmpnn.core.model import ProteinMPNN
        from proteinmpnn.core.layers import EncLayer, DecLayer
        from proteinmpnn.core.features import ProteinFeatures
        from proteinmpnn.core.embeddings import PositionalEncodings

    def test_data_imports(self):
        """Test data module imports."""
        from proteinmpnn.data.types import ProteinBatch, SamplingConfig, SampleResult
        from proteinmpnn.data.parsers import parse_PDB
        from proteinmpnn.data.featurize import featurize_batch

    def test_config_imports(self):
        """Test config module imports."""
        from proteinmpnn.config.types import ModelConfig, DesignConfig
        from proteinmpnn.config.loaders import load_design_config

    def test_utils_imports(self):
        """Test utils module imports."""
        from proteinmpnn.utils.tensor_ops import gather_edges, gather_nodes
        from proteinmpnn.utils.helpers import loss_nll, loss_smoothed


class TestDataTypes(unittest.TestCase):
    """Test dataclass types work correctly."""

    def test_sampling_config_defaults(self):
        """Test SamplingConfig has correct defaults."""
        from proteinmpnn.data.types import SamplingConfig
        config = SamplingConfig()
        self.assertEqual(config.temperature, 1.0)

    def test_sample_result_creation(self):
        """Test SampleResult can be created."""
        from proteinmpnn.data.types import SampleResult

        result = SampleResult(
            sequence="ACDEFG",
            score=1.5,
            global_score=1.5,
            recovery=0.5,
            temperature=0.1,
            sample_id=1
        )
        self.assertEqual(result.sequence, "ACDEFG")
        self.assertEqual(result.score, 1.5)


if __name__ == '__main__':
    unittest.main()
