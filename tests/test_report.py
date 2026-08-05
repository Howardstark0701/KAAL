"""Tests for kaal/reporting/ — Phase 8 verification.

Covers json_report.py and pdf.py.
"""
from __future__ import annotations

import json
import os
import tempfile

import pytest
import torch
import torchvision.models as models
import numpy as np
from PIL import Image

from kaal.engine.loader import load_model
from kaal.attacks.fgsm import fgsm_attack, fgsm_attack_dataset
from kaal.attacks.pgd import pgd_attack
from kaal.attacks.physical import test_physical_robustness
from kaal.scoring.kvs import calculate_kvs
from kaal.fingerprint.radar import generate_fingerprint
from kaal.reporting.json_report import generate_json_report
from kaal.reporting.pdf import generate_pdf_report


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def kaal_model(tmp_path_factory):
    m = models.resnet18(weights=None)
    m.eval()
    path = str(tmp_path_factory.mktemp("models") / "r18.pt")
    torch.save(m, path)
    return load_model(path)


@pytest.fixture(scope="module")
def sample_tensor():
    torch.manual_seed(88)
    return torch.randn(3, 224, 224) * 0.4


@pytest.fixture(scope="module")
def fgsm_res(kaal_model, sample_tensor):
    return fgsm_attack(kaal_model, sample_tensor, epsilon=0.4)


@pytest.fixture(scope="module")
def pgd_res(kaal_model, sample_tensor):
    return pgd_attack(kaal_model, sample_tensor, epsilon=0.3, steps=8)


@pytest.fixture(scope="module")
def physical_res(kaal_model, fgsm_res):
    return test_physical_robustness(
        kaal_model, fgsm_res.adversarial_tensor,
        fgsm_res.original_class,
        transformations=["jpeg_90", "blur_3", "noise_001"],
    )


@pytest.fixture(scope="module")
def kvs_res(fgsm_res, pgd_res, physical_res):
    return calculate_kvs(
        fgsm_result={"success_rate": float(fgsm_res.success),
                     "epsilon_used": fgsm_res.epsilon_used},
        pgd_result={"success_rate": float(pgd_res.success),
                    "epsilon_used": pgd_res.epsilon_used},
        physical_result=physical_res,
        min_epsilon=0.03,
    )


@pytest.fixture(scope="module")
def fingerprint_png(kvs_res, tmp_path_factory):
    path = str(tmp_path_factory.mktemp("fp") / "fp.png")
    return generate_fingerprint(kvs_res, "TestModel", path)


@pytest.fixture(scope="module")
def model_info(kaal_model):
    return {
        "path": "test_model.pt",
        "name": "resnet18",
        "framework": kaal_model.framework,
        "input_shape": list(kaal_model.input_shape),
        "num_classes": kaal_model.num_classes,
    }


@pytest.fixture(scope="module")
def dataset_info():
    return {
        "path": "./images/",
        "total_images": 100,
        "formats": {"jpg": 87, "png": 13},
    }


@pytest.fixture(scope="module")
def json_report_path(model_info, dataset_info, kvs_res, fgsm_res,
                     pgd_res, physical_res, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("json") / "report.json")
    return generate_json_report(
        output_path=out,
        model_info=model_info,
        dataset_info=dataset_info,
        kvs_result=kvs_res,
        fgsm_result={"success_rate": float(fgsm_res.success),
                     "epsilon_used": fgsm_res.epsilon_used,
                     "avg_confidence_delta": fgsm_res.confidence_delta,
                     "plain_english": fgsm_res.plain_english},
        pgd_result={"success_rate": float(pgd_res.success),
                    "epsilon_used": pgd_res.epsilon_used,
                    "alpha_used": pgd_res.alpha_used,
                    "steps_used": pgd_res.steps_used,
                    "avg_steps_to_success": pgd_res.steps_to_success,
                    "plain_english": pgd_res.plain_english},
        physical_result=physical_res,
        audit_duration_seconds=142.0,
    )


@pytest.fixture(scope="module")
def json_report_doc(json_report_path):
    with open(json_report_path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pdf_report_path(model_info, dataset_info, kvs_res, fgsm_res,
                    pgd_res, physical_res, fingerprint_png, tmp_path_factory):
    out = str(tmp_path_factory.mktemp("pdf") / "report.pdf")
    return generate_pdf_report(
        output_path=out,
        model_info=model_info,
        dataset_info=dataset_info,
        kvs_result=kvs_res,
        fgsm_result=fgsm_res,
        pgd_result=pgd_res,
        physical_result=physical_res,
        fingerprint_path=fingerprint_png,
        audit_duration_seconds=142.0,
    )


# ===========================================================================
# JSON Report tests
# ===========================================================================

class TestJSONReport:

    def test_returns_string_path(self, json_report_path):
        assert isinstance(json_report_path, str)

    def test_file_created(self, json_report_path):
        assert os.path.exists(json_report_path)

    def test_valid_json(self, json_report_doc):
        assert isinstance(json_report_doc, dict)

    def test_top_level_keys(self, json_report_doc):
        required = {"meta", "model", "dataset", "kvs",
                    "attacks", "physical_robustness",
                    "output_files", "remediation"}
        assert required.issubset(set(json_report_doc.keys()))

    def test_meta_kaal_version(self, json_report_doc):
        assert json_report_doc["meta"]["kaal_version"] == "1.0.0"

    def test_meta_audit_id_is_uuid(self, json_report_doc):
        import uuid
        uid = json_report_doc["meta"]["audit_id"]
        assert uuid.UUID(uid)

    def test_meta_timestamp_is_iso(self, json_report_doc):
        ts = json_report_doc["meta"]["timestamp"]
        assert "T" in ts and ("Z" in ts or "+00:00" in ts)

    def test_meta_duration(self, json_report_doc):
        assert json_report_doc["meta"]["duration_seconds"] == pytest.approx(142.0, abs=0.1)

    def test_model_framework(self, json_report_doc):
        assert json_report_doc["model"]["framework"] == "pytorch"

    def test_model_input_shape_is_list(self, json_report_doc):
        assert isinstance(json_report_doc["model"]["input_shape"], list)

    def test_model_num_classes(self, json_report_doc):
        assert json_report_doc["model"]["num_classes"] == 1000

    def test_dataset_total_images(self, json_report_doc):
        assert json_report_doc["dataset"]["total_images"] == 100

    def test_dataset_formats_dict(self, json_report_doc):
        assert isinstance(json_report_doc["dataset"]["formats"], dict)

    def test_kvs_score_in_range(self, json_report_doc):
        score = json_report_doc["kvs"]["score"]
        assert 0.0 <= score <= 10.0

    def test_kvs_label_present(self, json_report_doc):
        assert isinstance(json_report_doc["kvs"]["label"], str)
        assert len(json_report_doc["kvs"]["label"]) > 0

    def test_kvs_color_is_hex(self, json_report_doc):
        assert json_report_doc["kvs"]["color"].startswith("#")

    def test_kvs_dimension_scores_present(self, json_report_doc):
        assert isinstance(json_report_doc["kvs"]["dimension_scores"], dict)

    def test_attacks_has_fgsm(self, json_report_doc):
        assert "fgsm" in json_report_doc["attacks"]

    def test_attacks_fgsm_epsilon(self, json_report_doc):
        assert "epsilon" in json_report_doc["attacks"]["fgsm"]

    def test_attacks_has_pgd(self, json_report_doc):
        assert "pgd" in json_report_doc["attacks"]

    def test_physical_has_survival_rate(self, json_report_doc):
        assert "overall_survival_rate" in json_report_doc["physical_robustness"]

    def test_physical_has_threat_rating(self, json_report_doc):
        assert json_report_doc["physical_robustness"]["physical_threat_rating"] in {
            "Lab Only", "Limited", "Field Ready"
        }

    def test_physical_per_transform_present(self, json_report_doc):
        assert isinstance(json_report_doc["physical_robustness"]["per_transform"], dict)

    def test_output_files_has_pdf(self, json_report_doc):
        assert "pdf_report" in json_report_doc["output_files"]

    def test_output_files_has_fingerprint(self, json_report_doc):
        assert "fingerprint_chart" in json_report_doc["output_files"]

    def test_remediation_is_list(self, json_report_doc):
        assert isinstance(json_report_doc["remediation"], list)

    def test_floats_rounded_to_4dp(self, json_report_doc):
        score = json_report_doc["kvs"]["score"]
        assert score == round(score, 4)

    def test_no_tensors_in_json(self, json_report_path):
        with open(json_report_path, encoding="utf-8") as f:
            raw = f.read()
        assert "tensor" not in raw.lower()

    def test_custom_audit_id(self, model_info, dataset_info, tmp_path):
        out = str(tmp_path / "id_test.json")
        path = generate_json_report(
            output_path=out,
            model_info=model_info,
            dataset_info=dataset_info,
            audit_id="test-audit-123",
        )
        with open(path) as f:
            doc = json.load(f)
        assert doc["meta"]["audit_id"] == "test-audit-123"

    def test_minimal_call_works(self, model_info, dataset_info, tmp_path):
        out = str(tmp_path / "minimal.json")
        path = generate_json_report(
            output_path=out,
            model_info=model_info,
            dataset_info=dataset_info,
        )
        assert os.path.exists(path)


# ===========================================================================
# PDF Report tests
# ===========================================================================

class TestPDFReport:

    def test_returns_string_path(self, pdf_report_path):
        assert isinstance(pdf_report_path, str)

    def test_file_created(self, pdf_report_path):
        assert os.path.exists(pdf_report_path)

    def test_is_valid_pdf(self, pdf_report_path):
        with open(pdf_report_path, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_pdf_size_reasonable(self, pdf_report_path):
        size_kb = os.path.getsize(pdf_report_path) / 1024
        assert size_kb > 5, f"PDF too small: {size_kb:.1f} KB"

    def test_returns_absolute_path(self, pdf_report_path):
        assert os.path.isabs(pdf_report_path)

    def test_creates_parent_directory(self, model_info, dataset_info,
                                      kvs_res, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "report.pdf")
        path = generate_pdf_report(
            output_path=out,
            model_info=model_info,
            dataset_info=dataset_info,
            kvs_result=kvs_res,
        )
        assert os.path.exists(path)

    def test_minimal_call_works(self, model_info, dataset_info, tmp_path):
        out = str(tmp_path / "minimal.pdf")
        path = generate_pdf_report(
            output_path=out,
            model_info=model_info,
            dataset_info=dataset_info,
        )
        assert os.path.exists(path)
        with open(path, "rb") as f:
            assert f.read(5) == b"%PDF-"

    def test_pdf_with_adversarial_image(self, model_info, dataset_info,
                                        kvs_res, fgsm_res, tmp_path):
        out = str(tmp_path / "with_adv.pdf")
        path = generate_pdf_report(
            output_path=out,
            model_info=model_info,
            dataset_info=dataset_info,
            kvs_result=kvs_res,
            fgsm_result=fgsm_res,
        )
        assert os.path.exists(path)


# ===========================================================================
# Integration: JSON + PDF from same audit data
# ===========================================================================

class TestReportIntegration:

    def test_json_and_pdf_same_audit(self, model_info, dataset_info,
                                     kvs_res, fgsm_res, pgd_res,
                                     physical_res, tmp_path):
        out_dir = str(tmp_path / "full_audit")
        json_path = generate_json_report(
            output_path=os.path.join(out_dir, "report.json"),
            model_info=model_info,
            dataset_info=dataset_info,
            kvs_result=kvs_res,
            fgsm_result={"success_rate": float(fgsm_res.success),
                         "epsilon_used": fgsm_res.epsilon_used},
            audit_duration_seconds=90.0,
        )
        pdf_path = generate_pdf_report(
            output_path=os.path.join(out_dir, "report.pdf"),
            model_info=model_info,
            dataset_info=dataset_info,
            kvs_result=kvs_res,
            fgsm_result=fgsm_res,
            pgd_result=pgd_res,
            physical_result=physical_res,
        )
        assert os.path.exists(json_path)
        assert os.path.exists(pdf_path)
        with open(json_path) as f:
            doc = json.load(f)
        assert doc["kvs"]["score"] == kvs_res.score
