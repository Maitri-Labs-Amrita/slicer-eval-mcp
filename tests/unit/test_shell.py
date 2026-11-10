"""Shell construction (slicer_mcp.shell) and the registration report."""

import numpy as np
import pytest

from slicer_mcp import shell
from slicer_mcp.config import load_protocol

sitk = pytest.importorskip("SimpleITK")
CLASSES = shell.classes_from(load_protocol("ct_segments"))
SPACING = (0.40625, 0.40625, 0.4)


def _ball(shape, center, radius_vox):
    k, j, i = np.indices(shape)
    return (k - center[0]) ** 2 + (j - center[1]) ** 2 + (i - center[2]) ** 2 <= radius_vox ** 2


def _two_touching_balls():
    shape = (40, 40, 60)
    a = _ball(shape, (20, 20, 20), 9)
    b = _ball(shape, (20, 20, 36), 9)
    return shell.combine_masks([(3, a), (4, b)], shape), a, b


def test_shell_is_hollow_outside_and_has_no_internal_wall():
    labels, a, b = _two_touching_balls()
    out = shell.shell_array(labels, SPACING, 1.0)
    assert not np.any(out[labels > 0])
    assert set(np.unique(out)) == {0, 3, 4}
    assert shell.radius_voxels(SPACING, 1.0) == [2, 2, 2]
    row = out[20, 20, :]
    assert list(np.nonzero(row)[0]) == [9, 10, 46, 47]


def test_overlap_goes_to_the_higher_label():
    shape = (30, 30, 30)
    a, b = _ball(shape, (15, 15, 13), 8), _ball(shape, (15, 15, 17), 8)
    labels = shell.combine_masks([(4, b), (3, a)], shape)
    assert np.all(labels[a & b] == 4)
    out = shell.shell_array(labels, SPACING, 1.0)
    da = sitk.GetArrayFromImage(sitk.BinaryDilate(sitk.GetImageFromArray(a.astype(np.uint8)), [2, 2, 2],
                                                  sitk.sitkBall)) > 0
    db = sitk.GetArrayFromImage(sitk.BinaryDilate(sitk.GetImageFromArray(b.astype(np.uint8)), [2, 2, 2],
                                                  sitk.sitkBall)) > 0
    both = da & db & (out > 0)
    assert both.any() and np.all(out[both] == 4)


def test_empty_label_map_is_an_error():
    with pytest.raises(shell.ShellError):
        shell.shell_array(np.zeros((5, 5, 5), np.uint8), SPACING)


def test_six_classes_in_label_order():
    assert [c["name"] for c in CLASSES] == ["liver", "spleen", "kidney_left", "kidney_right", "aorta", "pancreas"]
    assert [c["label"] for c in CLASSES] == [1, 2, 3, 4, 5, 6]
    assert len({tuple(c["color_rgb"]) for c in CLASSES}) == 6


def test_registration_check_fails_on_a_shifted_segmentation():
    shape = (40, 40, 40)
    ct = np.full(shape, -800.0)
    pool = _ball(shape, (20, 20, 20), 8)
    ct[pool] = 350.0
    check = load_protocol("ct_segments")["segmentation"]["registration_check"]
    on = shell.combine_masks([(3, pool)], shape)
    good = shell.registration_report(ct, on, CLASSES, check, outside={"kidney_left": 0.0})
    assert good["failures"] == [] and good["classes"]["kidney_left"]["mean_hu"] == 350.0
    shifted = np.roll(on, 30 * 2, axis=2)
    bad = shell.registration_report(ct, shifted, CLASSES, check, outside={"kidney_left": 0.3})
    assert len(bad["failures"]) == 1 and "kidney_left" in bad["failures"][0] and "outside the CT" in bad["failures"][0]


def test_an_unenhanced_scan_passes_the_registration_check():
    """The mean CT value is reported, never a reason to refuse (an unenhanced scan: blood ~0-50 HU)."""
    shape = (40, 40, 40)
    ct = np.full(shape, -800.0)
    pool = _ball(shape, (20, 20, 20), 8)
    ct[pool] = 30.0
    check = load_protocol("ct_segments")["segmentation"]["registration_check"]
    r = shell.registration_report(ct, shell.combine_masks([(3, pool)], shape), CLASSES, check, outside={"kidney_left": 0.0})
    assert r["failures"] == [] and r["classes"]["kidney_left"]["mean_hu"] == 30.0


def test_a_label_map_has_no_mesh_and_no_failures():
    shape = (20, 20, 20)
    on = shell.combine_masks([(3, _ball(shape, (10, 10, 10), 5))], shape)
    check = load_protocol("ct_segments")["segmentation"]["registration_check"]
    assert shell.registration_report(np.zeros(shape), on, CLASSES, check)["failures"] == []


def test_bbox():
    m = np.zeros((4, 5, 6), bool)
    m[1:3, 2:4, 3:5] = True
    ijk = np.diag([2.0, 3.0, 4.0, 1.0])
    box = shell.bbox_ras(m, ijk)
    assert box == {"min": [6.0, 6.0, 4.0], "max": [8.0, 9.0, 8.0], "center": [7.0, 7.5, 6.0]}
    assert shell.bbox_ras(np.zeros((2, 2, 2), bool), ijk) is None
