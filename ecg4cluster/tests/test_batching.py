"""Regression coverage for wrapper batching, without vendor code or model weights."""

import csv
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import digitize
from open_ecg_runner import OpenECGRunner


class RecordingSegmentationModel(torch.nn.Module):
    """Give each image distinct, deterministic logits and record actual batches."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.25))
        self.batches = []

    def forward(self, images):
        self.batches.append(images.detach().clone())
        red, green, blue = images.split(1, dim=1)
        return torch.cat(
            (
                red * self.scale,
                green - red * 0.25,
                blue * -0.75,
                (red + green + blue) * 0.3,
            ),
            dim=1,
        )


class SegmentBatchTests(unittest.TestCase):
    def make_runner(self, images):
        runner = object.__new__(OpenECGRunner)
        runner.device = "cpu"
        runner.segmentation_model = RecordingSegmentationModel().eval()
        runner._read_image = Mock(side_effect=images.__getitem__)
        return runner

    @staticmethod
    def image(height, width, offset):
        pixels = torch.linspace(0.0, 1.0, 3 * height * width)
        return (pixels.reshape(1, 3, height, width) + offset).requires_grad_()

    @staticmethod
    def expected_probabilities(image):
        model = RecordingSegmentationModel().eval()
        with torch.no_grad():
            return model(image).softmax(dim=1)

    def test_same_shape_images_share_one_model_call(self):
        images = {
            Path("first.jpg"): self.image(8, 12, 0.0),
            Path("second.jpg"): self.image(8, 12, 2.0),
            Path("third.jpg"): self.image(8, 12, 5.0),
        }
        runner = self.make_runner(images)

        outputs = runner.segment_batch(list(images))

        self.assertEqual(len(runner.segmentation_model.batches), 1)
        self.assertEqual(runner.segmentation_model.batches[0].shape, (3, 3, 8, 12))
        self.assertEqual(len(outputs), 3)
        for output, source in zip(outputs, images.values()):
            self.assertEqual(output.shape, (1, 4, 8, 12))
            torch.testing.assert_close(output, self.expected_probabilities(source))

    def test_mixed_shapes_are_grouped_without_padding_and_preserve_order(self):
        images = {
            Path("wide-first.jpg"): self.image(8, 12, 0.0),
            Path("tall.jpg"): self.image(12, 8, 2.0),
            Path("wide-last.jpg"): self.image(8, 12, 5.0),
        }
        runner = self.make_runner(images)

        outputs = runner.segment_batch(list(images))

        batch_shapes = [tuple(batch.shape) for batch in runner.segmentation_model.batches]
        self.assertCountEqual(batch_shapes, [(2, 3, 8, 12), (1, 3, 12, 8)])
        self.assertEqual(len(outputs), len(images))
        for output, source in zip(outputs, images.values()):
            self.assertEqual(output.shape[2:], source.shape[2:])
            torch.testing.assert_close(output, self.expected_probabilities(source))

    def test_batch_matches_individual_predictions_without_tracking_gradients(self):
        images = {
            Path("dark.jpg"): self.image(6, 10, -3.0),
            Path("light.jpg"): self.image(6, 10, 7.0),
        }
        runner = self.make_runner(images)

        together = runner.segment_batch(list(images))
        individual = [runner.segment_batch([path])[0] for path in images]

        for batched, single in zip(together, individual):
            self.assertFalse(batched.requires_grad)
            self.assertIsNone(batched.grad_fn)
            torch.testing.assert_close(batched, single)

    def test_probabilities_normalize_channels_independently_for_each_image(self):
        images = {
            Path("low.jpg"): self.image(4, 6, -10.0),
            Path("high.jpg"): self.image(4, 6, 10.0),
        }
        runner = self.make_runner(images)

        outputs = runner.segment_batch(list(images))

        for output, source in zip(outputs, images.values()):
            torch.testing.assert_close(output.sum(dim=1), torch.ones((1, 4, 6)))
            torch.testing.assert_close(output, self.expected_probabilities(source))
        self.assertFalse(torch.allclose(outputs[0], outputs[1]))

    def test_empty_batch_does_not_run_model(self):
        runner = self.make_runner({})

        self.assertEqual(runner.segment_batch([]), [])

        runner._read_image.assert_not_called()
        self.assertEqual(runner.segmentation_model.batches, [])


class MainBatchIntegrationTests(unittest.TestCase):
    def test_cli_batches_keep_metadata_profiles_and_waveforms_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "input"
            input_dir.mkdir()
            output_dir = root / "output"
            filenames = [
                "DII_tamir_eII_10.jpg",
                "DIII_adelya_2.jpg",
                "DII_aygul_eI_1.jpg",
            ]
            for filename in filenames:
                (input_dir / filename).touch()
            (input_dir / "ignored.png").touch()
            ordered_names = [filenames[1], filenames[2], filenames[0]]
            page_ids = {name: index + 1 for index, name in enumerate(ordered_names)}
            layouts = {
                "tamir_layout": {"leads": [["III"], ["V6"]], "total_rows": 2},
                "aygul_layout": {"leads": [["I", "II", "III"]], "total_rows": 1},
            }
            for name, layout in layouts.items():
                (root / f"{name}.yml").write_text(yaml.safe_dump({name: layout}))
            config = {
                "paper_speed_mm_s": 25.0,
                "gain_mm_mV": 10.0,
                "resample_size": 3000,
                "batch_size": 1,
                "target_samples": 3,
                "seed": 42,
                "image_extensions": [".jpg"],
                "metadata_regex": (
                    r"^(?:DII|DIII)_(?P<participant>[^_]+)_"
                    r"(?:(?P<experiment>eI|eII)_)?(?P<scan_index>\d+)\.jpg$"
                ),
                "experiment_order": ["eI", "eII"],
                "profiles": {
                    "tamir_adelya": {
                        "filename_regex": r"_(?:tamir|adelya)_",
                        "layout_file": "tamir_layout.yml",
                        "preferred_leads": ["V6", "III"],
                    },
                    "azamat_aygul": {
                        "filename_regex": r"_(?:azamat|aygul)_",
                        "layout_file": "aygul_layout.yml",
                        "preferred_leads": ["II", "V5"],
                    },
                },
            }
            config_path = root / "config.yml"
            config_path.write_text(yaml.safe_dump(config))
            runner = Mock(device="cpu")
            runner.segment_batch.side_effect = lambda paths: [
                torch.full((1, 4, 2, 2), float(page_ids[path.name])) for path in paths
            ]
            processed_pages = []

            def fake_digitize(probabilities, page_layouts):
                page_id = int(probabilities[0, 0, 0, 0])
                layout_name = next(iter(page_layouts))
                processed_pages.append((page_id, layout_name))
                canonical_uv = (
                    page_id * 10000
                    + torch.arange(12).reshape(12, 1) * 1000
                    + torch.arange(3).reshape(1, 3)
                ).float()
                return {
                    "layout_name": layout_name,
                    "detected_rows": page_layouts[layout_name]["total_rows"],
                    "detected_lead_labels": 3,
                    "signal": {
                        "canonical_lines": canonical_uv,
                        "layout_matching_cost": 0.0,
                    },
                }

            runner.digitize_probabilities.side_effect = fake_digitize
            runner.time_axis.return_value = np.array([0.0, 0.01, 0.02])
            runner_factory = Mock(return_value=runner)
            runner_factory.load_layout.side_effect = lambda path: yaml.safe_load(path.read_text())
            argv = [
                "digitize.py", "--input", str(input_dir), "--output", str(output_dir),
                "--config", str(config_path), "--device", "cpu", "--batch-size", "2",
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(digitize, "OpenECGRunner", runner_factory),
                patch.object(digitize, "save_plot") as save_plot,
                patch.object(digitize, "setup_logger"),
            ):
                digitize.main()

            actual_batches = [
                [path.name for path in call.args[0]]
                for call in runner.segment_batch.call_args_list
            ]
            self.assertEqual(actual_batches, [ordered_names[:2], ordered_names[2:]])
            self.assertEqual(
                processed_pages,
                [(1, "tamir_layout"), (2, "aygul_layout"), (3, "tamir_layout")],
            )
            with (output_dir / "ecg_selected_waveforms.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 9)
            expectations = [
                ("adelya", "", "2", "tamir_adelya", "V6", 11),
                ("aygul", "eI", "1", "azamat_aygul", "II", 1),
                ("tamir", "eII", "10", "tamir_adelya", "V6", 11),
            ]
            for page_index, expected in enumerate(expectations):
                participant, experiment, scan_index, profile, lead, lead_index = expected
                page_rows = rows[page_index * 3 : (page_index + 1) * 3]
                self.assertEqual([row["scan_file"] for row in page_rows], [ordered_names[page_index]] * 3)
                for row in page_rows:
                    self.assertEqual(row["participant"], participant)
                    self.assertEqual(row["experiment"], experiment)
                    self.assertEqual(row["scan_index"], scan_index)
                    self.assertEqual(row["profile"], profile)
                    self.assertEqual(row["selected_lead"], lead)
                self.assertEqual([row["sample_index"] for row in page_rows], ["0", "1", "2"])
                self.assertEqual([row["is_scan_start"] for row in page_rows], ["True", "False", "False"])
                np.testing.assert_allclose(
                    [float(row["mV"]) for row in page_rows],
                    (page_index + 1) * 10 + lead_index + np.arange(3) / 1000,
                )
                np.testing.assert_allclose(
                    [float(row["time_s"]) for row in page_rows], [0.0, 0.01, 0.02]
                )
            saved_config = yaml.safe_load((output_dir / "run_config.yml").read_text())
            self.assertEqual(saved_config["runtime"]["batch_size"], 2)
            self.assertEqual(save_plot.call_count, 3)


if __name__ == "__main__":
    unittest.main()
