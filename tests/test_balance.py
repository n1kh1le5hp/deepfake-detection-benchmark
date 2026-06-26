"""Tests for the balancing / coverage logic (synthetic samples)."""
import numpy as np

from idtest.assemble.balance import select_fakes, select_reals
from idtest.datasets.base import Sample


def _fake(uid, method):
    return Sample(path=f"x/{uid}", split="fake", source_dataset="S", method=method,
                  manip_type="AEGAN", video_id=uid, is_video=True)


def _real(uid):
    return Sample(path=f"r/{uid}", split="real", source_dataset="S", method="pristine",
                  manip_type="AEGAN", video_id=uid, is_video=True)


def test_select_fakes_hits_target_and_keeps_all_methods(capsys):
    methods = [f"m{i:02d}" for i in range(13)]  # >= min_methods
    fakes = [_fake(f"{m}-{k}", m) for m in methods for k in range(10)]
    rng = np.random.default_rng(0)
    out = select_fakes(fakes, target=50, min_methods=13, rng=rng)
    assert len(out) == 50
    assert len({s.method for s in out}) == 13  # every method represented


def test_select_fakes_warns_when_methods_under_target(capsys):
    fakes = [_fake(f"a-{k}", "onlyA") for k in range(100)]
    rng = np.random.default_rng(0)
    out = select_fakes(fakes, target=20, min_methods=13, rng=rng)
    assert len(out) == 20
    assert "WARNING" in capsys.readouterr().out


def test_select_reals_deterministic_and_capped():
    reals = [_real(i) for i in range(200)]
    a = select_reals(reals, target=64, rng=np.random.default_rng(7))
    b = select_reals(reals, target=64, rng=np.random.default_rng(7))
    assert len(a) == 64 == len(b)
    assert [s.video_id for s in a] == [s.video_id for s in b]


def test_select_reals_fewer_than_target():
    reals = [_real(i) for i in range(10)]
    out = select_reals(reals, target=64, rng=np.random.default_rng(0))
    assert len(out) == 10  # cannot exceed availability
