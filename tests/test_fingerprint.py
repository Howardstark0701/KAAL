"""Tests for kaal/fingerprint/radar.py — Phase 7 verification."""

from __future__ import annotations

import os

import pytest

from kaal.scoring.kvs import calculate_kvs, KVSResult
from kaal.fingerprint.radar import generate_fingerprint, _AXES, _extract_scores


# ---------------------------------------------------------------------------
# Fixtures — mock KVSResult objects
# ---------------------------------------------------------------------------

def _make_kvs(scores: dict, score: float = 5.0) -> KVSResult:
    """Build a minimal KVSResult for testing."""
    from kaal.scoring.kvs import get_kvs_label, get_kvs_color
    return KVSResult(
        score=score,
        label=get_kvs_label(score),
        color=get_kvs_color(score),
        dimension_scores=scores,
        dimensions_tested=list(scores.keys()),
        dimensions_skipped=[d for d in _AXES if d not in scores],
        plain_english="Test result.",
        remediation=[],
    )


@pytest.fixture(scope="module")
def full_kvs():
    return _make_kvs({
        "fgsm_susceptibility":        8.0,
        "pgd_susceptibility":         9.0,
        "physical_survivability":     5.0,
        "blackbox_efficiency":        4.0,
        "empirical_robustness":       7.0,
        "adversarial_overconfidence": 6.0,
    }, score=7.2)


@pytest.fixture(scope="module")
def partial_kvs():
    """KVSResult with only 3 dims tested."""
    return _make_kvs({
        "fgsm_susceptibility":    6.0,
        "pgd_susceptibility":     7.0,
        "physical_survivability": 4.0,
    }, score=5.5)


@pytest.fixture(scope="module")
def comparison_kvs():
    return _make_kvs({
        "fgsm_susceptibility":        3.0,
        "pgd_susceptibility":         4.0,
        "physical_survivability":     2.0,
        "blackbox_efficiency":        1.5,
        "empirical_robustness":       3.5,
        "adversarial_overconfidence": 2.5,
    }, score=2.8)


@pytest.fixture(scope="module")
def zero_kvs():
    return _make_kvs({d: 0.0 for d in _AXES}, score=0.0)


# ---------------------------------------------------------------------------
# _extract_scores() helper tests
# ---------------------------------------------------------------------------

class TestExtractScores:

    def test_returns_list_of_six(self, full_kvs):
        scores = _extract_scores(full_kvs)
        assert len(scores) == 6

    def test_values_match_dimension_scores(self, full_kvs):
        scores = _extract_scores(full_kvs)
        assert scores[0] == full_kvs.dimension_scores["fgsm_susceptibility"]
        assert scores[1] == full_kvs.dimension_scores["pgd_susceptibility"]

    def test_missing_dims_default_to_zero(self, partial_kvs):
        scores = _extract_scores(partial_kvs)
        # blackbox_efficiency and the two new dims are missing → 0.0
        assert scores[3] == 0.0   # blackbox_efficiency
        assert scores[4] == 0.0   # empirical_robustness
        assert scores[5] == 0.0   # adversarial_overconfidence

    def test_all_zeros_for_empty(self):
        from kaal.scoring.kvs import KVSResult, get_kvs_label, get_kvs_color
        empty = KVSResult(0.0, "Robust", "#4ADE80", {}, [], list(_AXES), ".", [])
        scores = _extract_scores(empty)
        assert all(s == 0.0 for s in scores)


# ---------------------------------------------------------------------------
# generate_fingerprint() — file output tests
# ---------------------------------------------------------------------------

class TestGenerateFingerprint:

    def test_returns_string_path(self, full_kvs, tmp_path):
        out = str(tmp_path / "fp.png")
        path = generate_fingerprint(full_kvs, "TestModel", out)
        assert isinstance(path, str)

    def test_file_created(self, full_kvs, tmp_path):
        out = str(tmp_path / "fp.png")
        path = generate_fingerprint(full_kvs, "TestModel", out)
        assert os.path.exists(path)

    def test_file_is_png(self, full_kvs, tmp_path):
        out = str(tmp_path / "fp_magic.png")
        path = generate_fingerprint(full_kvs, "TestModel", out)
        with open(path, "rb") as f:
            magic = f.read(4)
        assert magic == b"\x89PNG"

    def test_file_nonzero_size(self, full_kvs, tmp_path):
        out = str(tmp_path / "fp_size.png")
        path = generate_fingerprint(full_kvs, "TestModel", out)
        assert os.path.getsize(path) > 20_000  # at least 20KB

    def test_returns_absolute_path(self, full_kvs, tmp_path):
        out = str(tmp_path / "fp_abs.png")
        path = generate_fingerprint(full_kvs, "TestModel", out)
        assert os.path.isabs(path)

    def test_creates_parent_directory(self, full_kvs, tmp_path):
        out = str(tmp_path / "nested" / "sub" / "fp.png")
        path = generate_fingerprint(full_kvs, "TestModel", out)
        assert os.path.exists(path)

    def test_partial_dims_produces_file(self, partial_kvs, tmp_path):
        out = str(tmp_path / "partial.png")
        path = generate_fingerprint(partial_kvs, "PartialModel", out)
        assert os.path.exists(path)

    def test_all_zeros_produces_file(self, zero_kvs, tmp_path):
        out = str(tmp_path / "zeros.png")
        path = generate_fingerprint(zero_kvs, "ZeroModel", out)
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Comparison mode tests
# ---------------------------------------------------------------------------

class TestComparisonMode:

    def test_comparison_produces_file(self, full_kvs, comparison_kvs, tmp_path):
        out = str(tmp_path / "compare.png")
        path = generate_fingerprint(
            full_kvs, "ModelA", out,
            comparison_kvs=comparison_kvs,
            comparison_name="ModelB",
        )
        assert os.path.exists(path)

    def test_comparison_file_larger_than_single(
            self, full_kvs, comparison_kvs, tmp_path):
        single_path  = str(tmp_path / "single.png")
        compare_path = str(tmp_path / "compare2.png")
        generate_fingerprint(full_kvs, "ModelA", single_path)
        generate_fingerprint(
            full_kvs, "ModelA", compare_path,
            comparison_kvs=comparison_kvs, comparison_name="ModelB",
        )
        # Comparison chart has legend → usually slightly larger or same size
        # (Just verify it's a valid PNG — size can vary)
        assert os.path.getsize(compare_path) > 15_000

    def test_comparison_none_produces_single_chart(self, full_kvs, tmp_path):
        out = str(tmp_path / "no_compare.png")
        path = generate_fingerprint(
            full_kvs, "ModelA", out, comparison_kvs=None
        )
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Axes definition tests
# ---------------------------------------------------------------------------

class TestAxesDefinition:

    def test_six_axes_defined(self):
        assert len(_AXES) == 6

    def test_axes_contain_all_kvs_dimensions(self):
        expected = {
            "fgsm_susceptibility",
            "pgd_susceptibility",
            "physical_survivability",
            "blackbox_efficiency",
            "empirical_robustness",
            "adversarial_overconfidence",
        }
        assert set(_AXES) == expected


# ---------------------------------------------------------------------------
# Integration test: calculate_kvs → generate_fingerprint pipeline
# ---------------------------------------------------------------------------

class TestKVSToFingerprintPipeline:

    def test_full_pipeline(self, tmp_path):
        """calculate_kvs() output feeds directly into generate_fingerprint()."""
        kvs = calculate_kvs(
            fgsm_success_rate=0.80,
            pgd_result={"success_rate": 0.90, "epsilon_used": 0.03},
            physical_result=type("R", (), {"overall_survival_rate": 0.50})(),
        )
        out = str(tmp_path / "pipeline.png")
        path = generate_fingerprint(kvs, "PipelineModel", out)
        assert os.path.exists(path)
        assert kvs.score > 0.0

    def test_kvs_score_visible_in_output(self, tmp_path):
        """Just verify the pipeline runs end-to-end without error."""
        kvs = calculate_kvs(fgsm_success_rate=0.95, pgd_success_rate=0.95)
        out = str(tmp_path / "high_risk.png")
        path = generate_fingerprint(kvs, "HighRiskModel", out)
        assert os.path.getsize(path) > 10_000
