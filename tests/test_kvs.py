"""Tests for kaal/scoring/kvs.py — Phase 7 verification."""

from __future__ import annotations

import pytest

from kaal.scoring.kvs import (
    KVSResult,
    REMEDIATION_MAP,
    calculate_kvs,
    get_kvs_color,
    get_kvs_label,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight mock result objects
# ---------------------------------------------------------------------------

def _fgsm_dict(success_rate: float, epsilon: float = 0.03) -> dict:
    return {"success_rate": success_rate, "epsilon_used": epsilon}


def _pgd_dict(success_rate: float, epsilon: float = 0.03) -> dict:
    return {"success_rate": success_rate, "epsilon_used": epsilon}


class _MockPhysical:
    def __init__(self, survival_rate: float):
        self.overall_survival_rate = survival_rate


class _MockBlackbox:
    def __init__(self, query_efficiency: float):
        self.query_efficiency = query_efficiency


# ---------------------------------------------------------------------------
# get_kvs_label() tests
# ---------------------------------------------------------------------------

class TestKVSLabel:

    @pytest.mark.parametrize("score,expected", [
        (0.0,  "Robust"),
        (1.0,  "Robust"),
        (2.0,  "Robust"),
        (2.1,  "Low Risk"),
        (4.0,  "Low Risk"),
        (4.1,  "Medium Risk"),
        (6.0,  "Medium Risk"),
        (6.1,  "High Risk"),
        (8.0,  "High Risk"),
        (8.1,  "Critical"),
        (9.5,  "Critical"),
        (9.6,  "Catastrophic"),
        (10.0, "Catastrophic"),
    ])
    def test_label_boundaries(self, score, expected):
        assert get_kvs_label(score) == expected

    def test_returns_string(self):
        assert isinstance(get_kvs_label(5.0), str)

    def test_all_labels_non_empty(self):
        for score in [0, 1, 3, 5, 7, 9, 10]:
            assert len(get_kvs_label(score)) > 0


# ---------------------------------------------------------------------------
# get_kvs_color() tests
# ---------------------------------------------------------------------------

class TestKVSColor:

    @pytest.mark.parametrize("score,expected", [
        (0.0,  "#4ADE80"),   # green
        (2.0,  "#4ADE80"),
        (2.1,  "#A3E635"),   # yellow-green
        (4.0,  "#A3E635"),
        (4.1,  "#FACC15"),   # yellow
        (6.0,  "#FACC15"),
        (6.1,  "#FB923C"),   # orange
        (8.0,  "#FB923C"),
        (8.1,  "#CC0000"),   # KAAL red
        (10.0, "#CC0000"),
    ])
    def test_color_boundaries(self, score, expected):
        assert get_kvs_color(score) == expected

    def test_returns_hex_string(self):
        color = get_kvs_color(5.0)
        assert isinstance(color, str)
        assert color.startswith("#")
        assert len(color) == 7


# ---------------------------------------------------------------------------
# KVSResult structure tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_kvs_result():
    return calculate_kvs(
        fgsm_result=_fgsm_dict(0.80, epsilon=0.03),
        pgd_result=_pgd_dict(0.90, epsilon=0.03),
        physical_result=_MockPhysical(0.50),
        min_epsilon=0.03,
    )


class TestKVSResultStructure:

    def test_returns_kvs_result(self, full_kvs_result):
        assert isinstance(full_kvs_result, KVSResult)

    def test_score_in_range(self, full_kvs_result):
        assert 0.0 <= full_kvs_result.score <= 10.0

    def test_score_is_rounded_to_1dp(self, full_kvs_result):
        assert full_kvs_result.score == round(full_kvs_result.score, 1)

    def test_label_is_string(self, full_kvs_result):
        assert isinstance(full_kvs_result.label, str)

    def test_label_matches_score(self, full_kvs_result):
        assert full_kvs_result.label == get_kvs_label(full_kvs_result.score)

    def test_color_is_hex(self, full_kvs_result):
        assert full_kvs_result.color.startswith("#")

    def test_color_matches_score(self, full_kvs_result):
        assert full_kvs_result.color == get_kvs_color(full_kvs_result.score)

    def test_dimension_scores_dict(self, full_kvs_result):
        assert isinstance(full_kvs_result.dimension_scores, dict)

    def test_dimension_scores_values_in_range(self, full_kvs_result):
        for v in full_kvs_result.dimension_scores.values():
            assert 0.0 <= v <= 10.0

    def test_dimensions_tested_is_list(self, full_kvs_result):
        assert isinstance(full_kvs_result.dimensions_tested, list)

    def test_dimensions_skipped_is_list(self, full_kvs_result):
        assert isinstance(full_kvs_result.dimensions_skipped, list)

    def test_tested_plus_skipped_equals_five(self, full_kvs_result):
        assert len(full_kvs_result.dimensions_tested) + len(full_kvs_result.dimensions_skipped) == 5

    def test_plain_english_is_one_sentence(self, full_kvs_result):
        text = full_kvs_result.plain_english
        assert isinstance(text, str) and len(text) > 0
        assert "!" not in text
        assert text.strip().endswith(".")

    def test_remediation_is_list(self, full_kvs_result):
        assert isinstance(full_kvs_result.remediation, list)


# ---------------------------------------------------------------------------
# KVS formula tests — exact spec values
# ---------------------------------------------------------------------------

class TestKVSFormula:

    def test_full_formula_all_five_dims(self):
        """Manual calculation with all five dimensions."""
        # dim1 = 0.8*10 = 8.0  weight 0.20  → 1.60
        # dim2 = 0.9*10 = 9.0  weight 0.30  → 2.70
        # dim3 = (1-0.03)*10 = 9.7  weight 0.20  → 1.94
        # dim4 = 0.5*10 = 5.0  weight 0.20  → 1.00
        # dim5 = 0.6*10 = 6.0  weight 0.10  → 0.60
        # raw = 1.60+2.70+1.94+1.00+0.60 = 7.84 → rounded = 7.8
        result = calculate_kvs(
            fgsm_result=_fgsm_dict(0.80),
            pgd_result=_pgd_dict(0.90),
            physical_result=_MockPhysical(0.50),
            blackbox_result=_MockBlackbox(0.60),
            min_epsilon=0.03,
        )
        assert result.score == pytest.approx(7.8, abs=0.15)

    def test_score_clamped_to_ten(self):
        result = calculate_kvs(
            fgsm_result=_fgsm_dict(1.0),
            pgd_result=_pgd_dict(1.0),
            physical_result=_MockPhysical(1.0),
            blackbox_result=_MockBlackbox(1.0),
            min_epsilon=0.001,
        )
        assert result.score <= 10.0

    def test_score_clamped_to_zero(self):
        result = calculate_kvs(
            fgsm_result=_fgsm_dict(0.0),
            pgd_result=_pgd_dict(0.0),
            physical_result=_MockPhysical(0.0),
            blackbox_result=_MockBlackbox(0.0),
            min_epsilon=1.0,
        )
        assert result.score >= 0.0

    def test_min_epsilon_clamped_low(self):
        """epsilon < 0.001 should be clamped to 0.001 → dim3 score ≈ 9.99."""
        result = calculate_kvs(min_epsilon=0.00001, fgsm_success_rate=0.0)
        dim3 = result.dimension_scores.get("perturbation_threshold", 0.0)
        assert dim3 > 9.9

    def test_min_epsilon_clamped_high(self):
        """epsilon > 1.0 should be clamped to 1.0 → dim3 score = 0.0."""
        result = calculate_kvs(min_epsilon=5.0, fgsm_success_rate=0.0)
        dim3 = result.dimension_scores.get("perturbation_threshold", None)
        if dim3 is not None:
            assert dim3 == pytest.approx(0.0, abs=0.01)

    def test_weight_redistribution_when_dims_skipped(self):
        """Score with missing blackbox dim should still be [0,10]."""
        result = calculate_kvs(
            fgsm_result=_fgsm_dict(0.8),
            pgd_result=_pgd_dict(0.8),
            physical_result=_MockPhysical(0.8),
            # blackbox skipped, min_epsilon auto-inferred from fgsm dict
        )
        assert 0.0 <= result.score <= 10.0
        # blackbox_efficiency must be in skipped
        assert "blackbox_efficiency" in result.dimensions_skipped
        # tested + skipped still = 5
        assert len(result.dimensions_tested) + len(result.dimensions_skipped) == 5

    def test_single_dim_produces_valid_score(self):
        result = calculate_kvs(fgsm_success_rate=0.5)
        assert 0.0 <= result.score <= 10.0
        assert len(result.dimensions_tested) == 1

    def test_no_dims_returns_zero(self):
        result = calculate_kvs()
        assert result.score == 0.0
        assert result.dimensions_tested == []
        assert len(result.dimensions_skipped) == 5


# ---------------------------------------------------------------------------
# Remediation tests
# ---------------------------------------------------------------------------

class TestRemediation:

    def test_high_fgsm_triggers_remediation(self):
        result = calculate_kvs(fgsm_success_rate=0.70)  # dim1 = 7.0 > 6.0
        assert any("preprocessing" in r or "JPEG" in r for r in result.remediation)

    def test_high_pgd_triggers_remediation(self):
        result = calculate_kvs(pgd_result=_pgd_dict(0.70))  # dim2 = 7.0 > 6.0
        assert any("adversarial training" in r for r in result.remediation)

    def test_low_threshold_triggers_remediation(self):
        # dim3 = (1 - 0.01) * 10 = 9.9 > 7.0
        result = calculate_kvs(min_epsilon=0.01, fgsm_success_rate=0.0)
        dim3 = result.dimension_scores.get("perturbation_threshold", 0.0)
        if dim3 > 7.0:
            assert any("ensemble" in r or "certified" in r for r in result.remediation)

    def test_high_physical_triggers_remediation(self):
        result = calculate_kvs(physical_result=_MockPhysical(0.80))  # dim4 = 8.0 > 6.0
        assert any("patch detection" in r or "physical" in r.lower()
                   for r in result.remediation)

    def test_high_blackbox_triggers_remediation(self):
        result = calculate_kvs(blackbox_result=_MockBlackbox(0.80))  # dim5 = 8.0 > 6.0
        assert any("rate limiting" in r or "confidence rounding" in r
                   for r in result.remediation)

    def test_robust_model_fewer_remediations(self):
        high_risk = calculate_kvs(
            fgsm_success_rate=0.9,
            pgd_result=_pgd_dict(0.9),
            physical_result=_MockPhysical(0.9),
        )
        robust = calculate_kvs(
            fgsm_success_rate=0.1,
            pgd_result=_pgd_dict(0.1),
            physical_result=_MockPhysical(0.1),
        )
        assert len(robust.remediation) <= len(high_risk.remediation)

    def test_remediation_map_has_five_keys(self):
        assert len(REMEDIATION_MAP) == 5

    def test_remediation_map_values_are_strings(self):
        for v in REMEDIATION_MAP.values():
            assert isinstance(v, str) and len(v) > 0


# ---------------------------------------------------------------------------
# Input format flexibility tests
# ---------------------------------------------------------------------------

class TestInputFormats:

    def test_fgsm_success_rate_override(self):
        result = calculate_kvs(fgsm_success_rate=0.5)
        assert result.dimension_scores.get("fgsm_susceptibility") == pytest.approx(5.0)

    def test_pgd_success_rate_override(self):
        result = calculate_kvs(pgd_success_rate=0.7)
        assert result.dimension_scores.get("pgd_susceptibility") == pytest.approx(7.0)

    def test_fgsm_dict_input(self):
        result = calculate_kvs(fgsm_result={"success_rate": 0.6})
        assert result.dimension_scores.get("fgsm_susceptibility") == pytest.approx(6.0)

    def test_physical_none_skipped(self):
        result = calculate_kvs(fgsm_success_rate=0.5, physical_result=None)
        assert "physical_survivability" in result.dimensions_skipped
