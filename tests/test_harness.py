"""Tests for the evaluation sample-table construction (path derivation + skipping)."""
from pathlib import Path

from idtest.evaluate.harness import build_sample_table, ALL_CONDITIONS


def _manifest(tmp_path: Path) -> list[dict]:
    # create the clean image files so the is_file() check passes
    (tmp_path / "fake01.png").write_bytes(b"x")
    (tmp_path / "real01.png").write_bytes(b"x")
    return [
        {"uid": "fake01", "split": "fake", "method": "Celeb-DF",
         "source_dataset": "Celeb-DF", "manip_type": "AEGAN", "video_id": "v1", "hard": 1,
         "path": str(tmp_path / "fake01.png")},
        {"uid": "real01", "split": "real", "method": "pristine",
         "source_dataset": "FF++", "manip_type": "AEGAN", "video_id": "v2", "hard": 0,
         "path": str(tmp_path / "real01.png")},
    ]


def test_build_sample_table_clean_only_when_no_perturbations(tmp_path):
    rows, skipped = build_sample_table(_manifest(tmp_path), tmp_path, conditions=ALL_CONDITIONS)
    # only the 2 clean images exist -> 2 rows, all 10 perturbation files skipped
    assert len(rows) == 2
    assert all(r["condition"] == "clean" for r in rows)
    assert skipped == 2 * 5
    assert rows[0]["label"] == 1 and rows[1]["label"] == 0
    assert rows[0]["hard"] == 1
    assert Path(rows[0]["path"]).name == "fake01.png"


def test_build_sample_table_derives_perturbed_paths(tmp_path):
    # create the perturbed files for the fake uid
    pert_dir = tmp_path / "perturbed" / "fake01"
    pert_dir.mkdir(parents=True)
    for t in ALL_CONDITIONS[1:]:
        (pert_dir / f"{t}.png").write_bytes(b"x")
    rows, skipped = build_sample_table(_manifest(tmp_path), tmp_path, conditions=ALL_CONDITIONS)
    by_cond = {r["condition"]: r for r in rows if r["uid"] == "fake01"}
    assert set(by_cond) == set(ALL_CONDITIONS)
    assert by_cond["clean"]["path"].endswith("fake01.png")
    assert by_cond["gaussian_blur"]["path"].endswith("perturbed/fake01/gaussian_blur.png")
    assert skipped == 5  # real01 has no perturbations


def test_build_sample_table_limit_caps_manifest_images(tmp_path):
    rows, _ = build_sample_table(_manifest(tmp_path), tmp_path,
                                 conditions=["clean"], limit=1)
    assert len(rows) == 1
    assert rows[0]["uid"] == "fake01"
