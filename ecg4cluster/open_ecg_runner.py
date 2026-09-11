from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torchvision.io import decode_image


class OpenECGRunner:
    def __init__(
        self,
        repository: Path,
        device: str,
        resample_size: int,
        target_samples: int,
        initial_layout_file: Path,
    ) -> None:
        repository = repository.resolve()
        sys.path.insert(0, str(repository))

        from src.model.cropper import Cropper
        from src.model.lead_identifier import LeadIdentifier
        from src.model.perspective_detector import PerspectiveDetector
        from src.model.pixel_size_finder import PixelSizeFinder
        from src.model.signal_extractor import SignalExtractor
        from src.model.unet import UNet

        self.device = device
        self.resample_size = resample_size
        self.minimum_image_size = 512
        self.signal_extractor = SignalExtractor()
        self.perspective_detector = PerspectiveDetector(num_thetas=250)
        self.cropper = Cropper(granularity=80, percentiles=(0.02, 0.98), alpha=0.85)
        self.pixel_size_finder = PixelSizeFinder(
            min_number_of_grid_lines=30,
            max_number_of_grid_lines=70,
            lower_grid_line_factor=0.3,
        )

        self.segmentation_model = UNet(
            num_in_channels=3,
            num_out_channels=4,
            dims=[32, 64, 128, 256, 320, 320, 320, 320],
            depth=2,
        )
        checkpoint = torch.load(
            repository / "weights" / "unet_weights_07072025.pt",
            weights_only=True,
            map_location=device,
        )
        if isinstance(checkpoint, tuple):
            checkpoint = checkpoint[0]
        checkpoint = {key.replace("_orig_mod.", ""): value for key, value in checkpoint.items()}
        self.segmentation_model.load_state_dict(checkpoint)
        self.segmentation_model.to(device).eval()

        lead_config = yaml.safe_load((repository / "src" / "config" / "lead_name_unet.yml").read_text())
        lead_name_model = UNet(**lead_config["MODEL"]["KWARGS"])
        checkpoint = torch.load(
            repository / "weights" / "lead_name_unet_weights_07072025.pt",
            map_location=device,
        )
        checkpoint = {key.replace("_orig_mod.", ""): value for key, value in checkpoint.items()}
        lead_name_model.load_state_dict(checkpoint)
        lead_name_model.eval()

        self.identifier = LeadIdentifier(
            layouts=self.load_layout(initial_layout_file),
            unet=lead_name_model,
            device=device,
            debug=False,
            possibly_flipped=False,
            target_num_samples=target_samples,
        )

    @staticmethod
    def load_layout(path: Path) -> dict:
        return yaml.safe_load(path.read_text())

    @staticmethod
    def _sparse_probability(probability: torch.Tensor) -> torch.Tensor:
        probability = torch.clamp(probability - probability.mean(), min=0)
        return probability / (probability.max() + 1e-9)

    def _resample(self, image: torch.Tensor) -> torch.Tensor:
        height, width = image.shape[2:]
        minimum, maximum = min(height, width), max(height, width)
        if minimum < self.minimum_image_size:
            scale = self.minimum_image_size / minimum
            size = (int(height * scale), int(width * scale))
            return F.interpolate(image, size=size, mode="bilinear", align_corners=False)
        if maximum > self.resample_size:
            scale = self.resample_size / maximum
            size = (int(height * scale), int(width * scale))
            return F.interpolate(image, size=size, mode="bilinear", align_corners=False, antialias=True)
        return image

    @staticmethod
    def _crop_y(*tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        combined = tensors[0] + tensors[1]
        collapsed = combined.squeeze().sum(dim=combined.dim() - 3)
        positive = torch.clamp(collapsed - collapsed.mean(), min=0)
        nonzero = (positive > 0).nonzero(as_tuple=True)[0]
        if nonzero.numel() == 0:
            first, last = 0, combined.shape[2] - 1
        else:
            first, last = int(nonzero[0].item()), int(nonzero[-1].item())
        slices = (slice(None), slice(None), slice(first, last + 1), slice(None))
        return tuple(tensor[slices] for tensor in tensors)

    @torch.no_grad()
    def digitize(self, image_path: Path, layouts: dict) -> dict:
        self.identifier.layouts = layouts
        image = decode_image(str(image_path), mode="RGB").unsqueeze(0)
        image = (image - image.min()) / (image.max() - image.min())
        image = self._resample(image.to(self.device))

        probabilities = torch.softmax(self.segmentation_model(image), dim=1)
        grid = self._sparse_probability(probabilities[:, [0]])
        text = self._sparse_probability(probabilities[:, [1]])
        signal = self._sparse_probability(probabilities[:, [2]])

        alignment = self.perspective_detector(grid)
        source_points = self.cropper(signal, alignment)
        signal = self.cropper.apply_perspective(signal, source_points, fill_value=0)
        grid = self.cropper.apply_perspective(grid, source_points, fill_value=0)
        text = self.cropper.apply_perspective(text, source_points, fill_value=0)
        if signal.shape[2] > signal.shape[3]:
            signal = torch.rot90(signal, k=3, dims=(2, 3))
            grid = torch.rot90(grid, k=3, dims=(2, 3))
            text = torch.rot90(text, k=3, dims=(2, 3))
        signal, grid, text = self._crop_y(signal, grid, text)

        mm_per_pixel_x, mm_per_pixel_y = self.pixel_size_finder(grid)
        # Preserve OpenECG v1.9.3 amplitude behavior; see README's known bugs.
        average_pixels_per_mm = (1 / mm_per_pixel_x + 1 / mm_per_pixel_y) / 2
        raw_lines = self.signal_extractor(signal.squeeze())
        layout = self.identifier(
            raw_lines,
            text,
            average_pixels_per_mm,
            layout_should_include_substring=None,
        )

        return {
            "layout_name": layout["layout"],
            "detected_rows": int(layout["rows_in_layout"]),
            "detected_lead_labels": int(layout["n_detected"]),
            "signal": {
                "raw_lines": raw_lines,
                "canonical_lines": layout["canonical_lines"],
                "layout_matching_cost": layout.get("cost", 1.0),
            },
            "pixel_spacing_mm": {
                "x": mm_per_pixel_x,
                "y": mm_per_pixel_y,
                "average_pixel_per_mm": average_pixels_per_mm,
            },
        }

    def time_axis(self, result: dict, paper_speed_mm_s: float) -> np.ndarray:
        lines = self.identifier._merge_nonoverlapping_lines(result["signal"]["raw_lines"])
        counts = torch.sum(~torch.isnan(lines), dim=0).cpu().numpy()
        valid = np.flatnonzero(counts >= self.identifier.required_valid_samples)
        width_px = int(valid[-1] - valid[0] + 1) if len(valid) else lines.shape[1]
        duration_s = (width_px - 1) * float(result["pixel_spacing_mm"]["x"]) / paper_speed_mm_s
        samples = result["signal"]["canonical_lines"].shape[-1]
        return np.linspace(0.0, duration_s, samples)
