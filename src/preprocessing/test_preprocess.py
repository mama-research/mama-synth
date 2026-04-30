"""Tests for Preprocessor.determine_slice_axis, find_largest_label_slice, and
extract_slice in preprocess.py.

Run with:
    pytest src/preprocessing/test_preprocess.py -v
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import helpers – insert the preprocessing source dir before importing
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import AmbiguousFOVError, Preprocessor  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: minimal Preprocessor (no real I/O needed for unit tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def pp(tmp_path: Path) -> Preprocessor:
    """Create a Preprocessor with minimal valid config for unit testing."""
    stats = {"mean": 0.0, "std": 1.0}
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps(stats))

    img_dir = tmp_path / "images"
    seg_dir = tmp_path / "segs"
    out_dir = tmp_path / "out"
    img_dir.mkdir()
    seg_dir.mkdir()

    return Preprocessor(
        image_dir=str(img_dir),
        segmentation_dir=str(seg_dir),
        output_dir=str(out_dir),
        global_stats_path=str(stats_file),
    )


@pytest.fixture()
def pp_skip(tmp_path: Path) -> Preprocessor:
    """Preprocessor with skip_ambiguous_shapes=True."""
    stats = {"mean": 0.0, "std": 1.0}
    stats_file = tmp_path / "stats.json"
    stats_file.write_text(json.dumps(stats))

    img_dir = tmp_path / "images"
    seg_dir = tmp_path / "segs"
    img_dir.mkdir()
    seg_dir.mkdir()

    return Preprocessor(
        image_dir=str(img_dir),
        segmentation_dir=str(seg_dir),
        output_dir=str(tmp_path / "out"),
        global_stats_path=str(stats_file),
        skip_ambiguous_shapes=True,
    )


# ===========================================================================
# determine_slice_axis — Case 1: two equal, one unique
# ===========================================================================

class TestDetermineSliceAxisTwoEqual:

    def test_unique_axis_is_last(self, pp: Preprocessor) -> None:
        # (512, 512, 80) — standard axial: unique axis is 2
        assert pp.determine_slice_axis((512, 512, 80)) == 2

    def test_unique_axis_is_first(self, pp: Preprocessor) -> None:
        # (80, 512, 512) — sagittal stored first: unique axis is 0
        assert pp.determine_slice_axis((80, 512, 512)) == 0

    def test_unique_axis_is_middle(self, pp: Preprocessor) -> None:
        # (512, 80, 512) — coronal stored in middle: unique axis is 1
        assert pp.determine_slice_axis((512, 80, 512)) == 1

    def test_no_spacing_no_warning(self, pp: Preprocessor) -> None:
        # When spacing is None no cross-validation warning should fire
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            axis = pp.determine_slice_axis((512, 512, 80), spacing=None)
        assert axis == 2

    def test_spacing_agrees_no_warning(self, pp: Preprocessor, caplog) -> None:
        # spacing (0.7, 0.7, 2.5) → argmax is axis 2, agrees with shape-based axis 2
        pp.determine_slice_axis((512, 512, 80), spacing=(0.7, 0.7, 2.5))
        assert "disagrees" not in caplog.text

    def test_spacing_disagrees_warns_but_uses_shape(
        self, pp: Preprocessor, caplog
    ) -> None:
        # shape says axis 2 (unique); spacing (2.5, 0.7, 0.7) says axis 0 → warn, keep 2
        axis = pp.determine_slice_axis((512, 512, 80), spacing=(2.5, 0.7, 0.7))
        assert axis == 2
        assert "disagrees" in caplog.text

    def test_spacing_disagrees_warns_shape_axis_in_message(
        self, pp: Preprocessor, caplog
    ) -> None:
        pp.determine_slice_axis((256, 256, 64), spacing=(3.0, 0.5, 0.5))
        assert "2" in caplog.text   # shape axis 2 mentioned
        assert "0" in caplog.text   # spacing axis 0 mentioned


# ===========================================================================
# determine_slice_axis — Case 2: cubic
# ===========================================================================

class TestDetermineSliceAxisCubic:

    def test_cubic_returns_axis2(self, pp: Preprocessor) -> None:
        axis = pp.determine_slice_axis((200, 200, 200))
        assert axis == 2

    def test_cubic_logs_warning(self, pp: Preprocessor, caplog) -> None:
        pp.determine_slice_axis((200, 200, 200))
        assert "Cubic" in caplog.text or "cubic" in caplog.text

    def test_cubic_with_spacing_still_returns_axis2(self, pp: Preprocessor) -> None:
        # spacing is irrelevant for cubic; axis 2 fallback is unconditional
        axis = pp.determine_slice_axis((128, 128, 128), spacing=(1.0, 1.0, 1.0))
        assert axis == 2


# ===========================================================================
# determine_slice_axis — Case 3: all three different
# ===========================================================================

class TestDetermineSliceAxisAllDifferent:

    def test_raises_ambiguous_error(self, pp: Preprocessor) -> None:
        with pytest.raises(AmbiguousFOVError):
            pp.determine_slice_axis((320, 256, 64))

    def test_error_message_contains_shape(self, pp: Preprocessor) -> None:
        with pytest.raises(AmbiguousFOVError, match="320"):
            pp.determine_slice_axis((320, 256, 64))

    def test_error_message_mentions_skip_flag(self, pp: Preprocessor) -> None:
        with pytest.raises(AmbiguousFOVError, match="skip_ambiguous_shapes"):
            pp.determine_slice_axis((320, 256, 64))

    def test_non_3d_raises_value_error(self, pp: Preprocessor) -> None:
        with pytest.raises(ValueError, match="3-D"):
            pp.determine_slice_axis((512, 512))

    def test_skip_flag_attribute(self, pp_skip: Preprocessor) -> None:
        # The flag is stored; the raise-vs-skip decision lives in process()
        assert pp_skip.skip_ambiguous_shapes is True

    def test_no_skip_flag_attribute(self, pp: Preprocessor) -> None:
        assert pp.skip_ambiguous_shapes is False


# ===========================================================================
# find_largest_label_slice
# ===========================================================================

class TestFindLargestLabelSlice:

    def _make_seg(
        self,
        shape: Tuple[int, int, int],
        hot_slice: int,
        axis: int,
        blob_size: int = 8,
    ) -> np.ndarray:
        """Create a zero segmentation with a single labelled blob on `hot_slice`."""
        seg = np.zeros(shape, dtype=np.float32)
        slicing: list = [slice(None)] * 3
        slicing[axis] = hot_slice
        slc = tuple(slicing)
        # Place a blob so it is clearly the largest
        sub = np.index_exp[:blob_size, :blob_size]
        remaining_axes = [i for i in range(3) if i != axis]
        idx: list = [slice(None)] * 3
        idx[axis] = hot_slice
        for rank, ra in enumerate(remaining_axes):
            idx[ra] = slice(0, blob_size)
        seg[tuple(idx)] = 1.0
        return seg

    def test_correct_slice_returned(self, pp: Preprocessor) -> None:
        seg = self._make_seg(shape=(512, 512, 80), hot_slice=40, axis=2)
        slice_idx, axis = pp.find_largest_label_slice(seg)
        assert axis == 2
        assert slice_idx == 40

    def test_correct_axis_sagittal(self, pp: Preprocessor) -> None:
        seg = self._make_seg(shape=(80, 512, 512), hot_slice=20, axis=0)
        slice_idx, axis = pp.find_largest_label_slice(seg)
        assert axis == 0
        assert slice_idx == 20

    def test_correct_axis_coronal(self, pp: Preprocessor) -> None:
        seg = self._make_seg(shape=(512, 64, 512), hot_slice=10, axis=1)
        slice_idx, axis = pp.find_largest_label_slice(seg)
        assert axis == 1
        assert slice_idx == 10

    def test_returns_tuple(self, pp: Preprocessor) -> None:
        seg = self._make_seg(shape=(256, 256, 30), hot_slice=15, axis=2)
        result = pp.find_largest_label_slice(seg)
        assert isinstance(result, tuple) and len(result) == 2

    def test_ambiguous_propagates(self, pp: Preprocessor) -> None:
        seg = np.zeros((320, 256, 64), dtype=np.float32)
        with pytest.raises(AmbiguousFOVError):
            pp.find_largest_label_slice(seg)


# ===========================================================================
# extract_slice
# ===========================================================================

class TestExtractSlice:

    def test_correct_shape_axis2(self, pp: Preprocessor) -> None:
        vol = np.zeros((512, 512, 80))
        s = pp.extract_slice(vol, slice_idx=10, axis=2)
        assert s.shape == (512, 512)

    def test_correct_shape_axis0(self, pp: Preprocessor) -> None:
        vol = np.zeros((80, 512, 512))
        s = pp.extract_slice(vol, slice_idx=5, axis=0)
        assert s.shape == (512, 512)

    def test_correct_shape_axis1(self, pp: Preprocessor) -> None:
        vol = np.zeros((512, 64, 512))
        s = pp.extract_slice(vol, slice_idx=32, axis=1)
        assert s.shape == (512, 512)

    def test_values_preserved(self, pp: Preprocessor) -> None:
        rng = np.random.RandomState(0)
        vol = rng.randn(64, 64, 20).astype(np.float32)
        for idx in [0, 10, 19]:
            np.testing.assert_array_equal(
                pp.extract_slice(vol, idx, axis=2),
                vol[:, :, idx],
            )

    def test_axis0_values(self, pp: Preprocessor) -> None:
        rng = np.random.RandomState(1)
        vol = rng.randn(20, 64, 64).astype(np.float32)
        np.testing.assert_array_equal(
            pp.extract_slice(vol, 7, axis=0),
            vol[7, :, :],
        )


# ===========================================================================
# Integration: process() respects skip_ambiguous_shapes
# ===========================================================================

class TestProcessSkipAmbiguous:

    def _make_seg_file(self, tmp_path: Path, shape: Tuple[int, int, int]) -> Path:
        """Write a NIfTI segmentation with the given shape to tmp_path."""
        import nibabel as nib
        data = np.zeros(shape, dtype=np.float32)
        # Put a small blob of label so find_largest_label_slice has something to find
        data[2:4, 2:4, 2:4] = 1.0
        img = nib.Nifti1Image(data, affine=np.eye(4))
        path = tmp_path / "pat001.nii.gz"
        nib.save(img, str(path))
        return path

    def _make_phase_file(
        self, tmp_path: Path, patient_id: str, shape: Tuple[int, int, int], phase: int
    ) -> Path:
        import nibabel as nib
        folder = tmp_path / patient_id
        folder.mkdir(exist_ok=True)
        data = np.ones(shape, dtype=np.float32)
        img = nib.Nifti1Image(data, affine=np.eye(4))
        path = folder / f"{patient_id}_{phase}.nii.gz"
        nib.save(img, str(path))
        return path

    def test_ambiguous_fov_raises_by_default(self, tmp_path: Path) -> None:
        """Without skip flag, AmbiguousFOVError should propagate out of process()."""
        stats = {"mean": 0.0, "std": 1.0}
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(stats))

        shape = (320, 256, 64)   # all three differ
        seg_dir = tmp_path / "segs"
        seg_dir.mkdir()
        self._make_seg_file(seg_dir, shape)

        img_dir = tmp_path / "images"
        self._make_phase_file(img_dir, "pat001", shape, phase=0)
        self._make_phase_file(img_dir, "pat001", shape, phase=1)

        pp = Preprocessor(
            image_dir=str(img_dir),
            segmentation_dir=str(seg_dir),
            output_dir=str(tmp_path / "out"),
            global_stats_path=str(stats_file),
            skip_ambiguous_shapes=False,
        )

        with pytest.raises(AmbiguousFOVError):
            pp.process()

    def test_ambiguous_fov_skipped_with_flag(self, tmp_path: Path, caplog) -> None:
        """With skip_ambiguous_shapes=True, bad patients are skipped (empty result)."""
        stats = {"mean": 0.0, "std": 1.0}
        stats_file = tmp_path / "stats.json"
        stats_file.write_text(json.dumps(stats))

        shape = (320, 256, 64)
        seg_dir = tmp_path / "segs"
        seg_dir.mkdir()
        self._make_seg_file(seg_dir, shape)

        img_dir = tmp_path / "images"
        self._make_phase_file(img_dir, "pat001", shape, phase=0)
        self._make_phase_file(img_dir, "pat001", shape, phase=1)

        pp = Preprocessor(
            image_dir=str(img_dir),
            segmentation_dir=str(seg_dir),
            output_dir=str(tmp_path / "out"),
            global_stats_path=str(stats_file),
            skip_ambiguous_shapes=True,
        )

        df = pp.process()
        assert df.empty
        assert "pat001" in caplog.text
        assert "Skipping" in caplog.text or "ambiguous" in caplog.text.lower()
