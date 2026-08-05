"""MoD/DRDO-style PDF compliance report — kaal/defence/compliance_report.py

Produces a 4-page adversarial robustness assessment report from a
CertificationBundle, matching the dark styling used in patch_to_printable().

Pages:
    1 — Cover page with classification marking, title, org, date, report ID
    2 — Executive summary: model identity, hashes, KVS, determinism, impact
    3 — Attack results table with risk contribution per attack vector
    4 — Certification statement based on risk tier

Usage:
    from kaal.defence.compliance_report import generate_compliance_report
    path = generate_compliance_report(bundle, "./kaal_cert/compliance_report.pdf")
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from kaal.defence.certification import CertificationBundle

# ReportLab units — imported at module level so all helpers can use them
try:
    from reportlab.lib.units import cm, mm
except ImportError:
    # Will raise a cleaner error at call time via generate_compliance_report()
    cm = mm = 28.346  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Colour constants (matches KAAL dark theme)
# ---------------------------------------------------------------------------

_BG        = (0.039, 0.039, 0.039)    # #0A0A0A
_SURFACE   = (0.067, 0.067, 0.067)    # #111111
_BORDER    = (0.122, 0.122, 0.122)    # #1F1F1F
_TEXT      = (0.949, 0.949, 0.949)    # #F2F2F2
_MUTED     = (0.533, 0.533, 0.533)    # #888888
_RED       = (0.800, 0.000, 0.000)    # #CC0000
_DIM       = (0.267, 0.267, 0.267)    # #444444

# Risk tier colours  (r, g, b) in [0,1]
_RISK_RGB: dict[str, tuple[float, float, float]] = {
    "Robust":       (0.133, 0.773, 0.369),   # #22C55E
    "Low Risk":     (0.918, 0.702, 0.031),   # #EAB308
    "Medium Risk":  (0.976, 0.451, 0.086),   # #F97316
    "High Risk":    (0.937, 0.267, 0.267),   # #EF4444
    "Critical":     (0.800, 0.000, 0.000),   # #CC0000
    "Catastrophic": (0.498, 0.000, 0.000),   # #7F0000
}

# Certification statements per risk tier
_CERT_STATEMENTS: dict[str, str] = {
    "Robust": (
        "Model demonstrates adequate adversarial robustness for evaluated "
        "attack vectors under standard epsilon constraints. No immediate "
        "remediation is required at this time."
    ),
    "Low Risk": (
        "Model demonstrates adequate adversarial robustness for evaluated "
        "attack vectors under standard epsilon constraints. Minor susceptibility "
        "was detected; monitoring is recommended."
    ),
    "Medium Risk": (
        "Model exhibits meaningful vulnerability across one or more attack "
        "vectors. Remediation or operational deployment constraints are "
        "recommended prior to production use in adversarial environments."
    ),
    "High Risk": (
        "Model is not recommended for deployment in adversarial environments "
        "without remediation. Reliable misclassification is achievable under "
        "realistic conditions with low perturbation. Immediate remediation required."
    ),
    "Critical": (
        "Model is not recommended for deployment in adversarial environments "
        "without remediation. Adversarial attacks are trivially effective across "
        "multiple dimensions. Immediate remediation required."
    ),
    "Catastrophic": (
        "Model is not recommended for deployment in adversarial environments "
        "without remediation. The model collapses under minimal perturbation "
        "across all tested dimensions. Immediate remediation required."
    ),
}

_FRAMEWORK_STATEMENT = (
    "This assessment was conducted using KAAL (Kaal Adversarial Auditing "
    "Layer), an offline adversarial robustness framework. Results are valid "
    "for the model file identified by the SHA-256 hashes above. The KVS "
    "(KAAL Vulnerability Score) is a composite metric scaled 0.0 to 10.0 "
    "across five weighted vulnerability dimensions."
)

_FOOTER_TEXT = "KAAL-D  \u00b7  github.com/Howardstark0701/Kaal"


# ---------------------------------------------------------------------------
# generate_compliance_report()
# ---------------------------------------------------------------------------

def generate_compliance_report(
    bundle: CertificationBundle,
    output_path: str = "./kaal_cert/compliance_report.pdf",
    classification_marking: str = "FOR OFFICIAL USE ONLY",
) -> str:
    """Generate a MoD/DRDO-style PDF compliance report.

    Args:
        bundle:                  CertificationBundle from certify_model().
        output_path:             Where to save the PDF.
        classification_marking:  Classification string shown on every page
                                 header and footer. Default "FOR OFFICIAL USE ONLY".

    Returns:
        Absolute path to the saved PDF.

    Raises:
        ImportError: reportlab is not installed.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise ImportError(
            "reportlab is not installed.\n"
            "→ Install it with: pip install reportlab==4.1.0"
        )

    output_path = os.path.abspath(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    PAGE_W, PAGE_H = A4                 # 595.27 × 841.89 pts
    MARGIN         = 2.0 * cm
    CONTENT_W      = PAGE_W - 2 * MARGIN

    # ── Derived metadata ──────────────────────────────────────────────────
    fp          = bundle.model_fingerprint
    date_str    = _fmt_date(bundle.audit_timestamp)
    report_id   = f"KAAL-{fp.file_hash[:8].upper()}-{_fmt_date_compact(bundle.audit_timestamp)}"
    risk_rgb    = _RISK_RGB.get(bundle.kvs_label, _RED)

    c = rl_canvas.Canvas(output_path, pagesize=A4)

    # ================================================================
    # PAGE 1 — Cover
    # ================================================================
    _fill_bg(c, PAGE_W, PAGE_H)

    # Classification marking — top
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=True)

    # Thin top border under classification
    _hline(c, MARGIN, PAGE_H - 1.8 * cm, PAGE_W - MARGIN)

    # KAAL-D title block — centred vertically in upper half
    y_mid = PAGE_H * 0.62

    _text_centred(c, PAGE_W / 2, y_mid + 1.2 * cm,
                  "AI Model Adversarial Robustness Assessment",
                  size=18, bold=True, rgb=_TEXT)

    _text_centred(c, PAGE_W / 2, y_mid + 0.5 * cm,
                  "KAAL-D Certification Report",
                  size=13, bold=False, rgb=_RED)

    # Divider
    _hline(c, MARGIN, y_mid, PAGE_W - MARGIN, rgb=_BORDER)

    # Metadata block
    meta_y = y_mid - 0.7 * cm
    for label, value in [
        ("Organisation", bundle.org_name),
        ("Date",         date_str),
        ("Report ID",    report_id),
    ]:
        _kv_line(c, MARGIN + 1 * cm, meta_y, label, value,
                 label_w=3.5 * cm, page_w=PAGE_W, margin=MARGIN)
        meta_y -= 0.7 * cm

    # KVS badge on cover
    _kvs_cover_badge(c, PAGE_W, risk_rgb, bundle.kvs_score, bundle.kvs_label)

    # Bottom divider + classification
    _hline(c, MARGIN, 1.8 * cm, PAGE_W - MARGIN)
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=False)
    _draw_footer(c, PAGE_W)

    c.showPage()

    # ================================================================
    # PAGE 2 — Executive Summary
    # ================================================================
    _fill_bg(c, PAGE_W, PAGE_H)
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=True)
    _hline(c, MARGIN, PAGE_H - 1.8 * cm, PAGE_W - MARGIN)

    y = PAGE_H - 2.6 * cm
    _section_header(c, MARGIN, y, "EXECUTIVE SUMMARY")
    y -= 0.8 * cm

    # Model identity block
    _subsection(c, MARGIN, y, "Model Identification")
    y -= 0.55 * cm

    model_filename = Path(fp.model_path).name
    rows_identity = [
        ("Model File",           model_filename),
        ("Full Path",            _truncate(fp.model_path, 55)),
        ("File Size",            f"{fp.file_size_bytes:,} bytes"),
        ("Last Modified",        fp.modified_at[:19].replace("T", "  ")),
        ("File Hash (SHA-256)",  _wrap_hash(fp.file_hash)),
        ("Weight Hash (SHA-256)",_wrap_hash(fp.weight_hash)),
    ]
    y = _table_rows(c, MARGIN, y, rows_identity, PAGE_W, MARGIN)

    y -= 0.4 * cm
    _subsection(c, MARGIN, y, "Assessment Results")
    y -= 0.55 * cm

    det_str   = "PASS" if bundle.is_deterministic else "FAIL"
    det_rgb   = (0.133, 0.773, 0.369) if bundle.is_deterministic else _RED
    kvs_str   = f"{bundle.kvs_score:.1f} / 10.0  [{bundle.kvs_label}]"

    rows_results = [
        ("KVS Score",       kvs_str),
        ("Re-audit Score",  f"{bundle.re_audit_kvs_score:.1f} / 10.0"),
        ("Determinism",     det_str, det_rgb),
    ]
    y = _table_rows(c, MARGIN, y, rows_results, PAGE_W, MARGIN)

    y -= 0.4 * cm
    _subsection(c, MARGIN, y, "Operational Impact")
    y -= 0.55 * cm
    y = _wrapped_para(c, MARGIN + 0.3 * cm, y, bundle.operational_impact,
                      max_width=CONTENT_W - 0.3 * cm, rgb=_TEXT, size=9)

    _hline(c, MARGIN, 1.8 * cm, PAGE_W - MARGIN)
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=False)
    _draw_footer(c, PAGE_W)
    c.showPage()

    # ================================================================
    # PAGE 3 — Attack Results Table
    # ================================================================
    _fill_bg(c, PAGE_W, PAGE_H)
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=True)
    _hline(c, MARGIN, PAGE_H - 1.8 * cm, PAGE_W - MARGIN)

    y = PAGE_H - 2.6 * cm
    _section_header(c, MARGIN, y, "ATTACK RESULTS")
    y -= 0.8 * cm

    # Table header
    col_x     = [MARGIN, MARGIN + 6 * cm, MARGIN + 11.5 * cm]
    col_heads = ["Attack Vector", "Success Rate", "Risk Contribution"]
    row_h     = 0.7 * cm

    # Header row background
    c.setFillColorRGB(*_SURFACE)
    c.rect(MARGIN, y - row_h * 0.15, CONTENT_W, row_h, fill=1, stroke=0)
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(*_MUTED)
    for cx, head in zip(col_x, col_heads):
        c.drawString(cx + 0.2 * cm, y + 0.12 * cm, head.upper())
    y -= row_h
    _hline(c, MARGIN, y, PAGE_W - MARGIN, rgb=_BORDER)

    # Attack rows
    attacks_data = [
        ("FGSM (Fast Gradient Sign Method)",  bundle.fgsm_success_rate),
        ("PGD (Projected Gradient Descent)",  bundle.pgd_success_rate),
        ("Adversarial Patch",                 bundle.patch_success_rate),
    ]

    for i, (attack_name, rate) in enumerate(attacks_data):
        # Alternate row bg
        if i % 2 == 0:
            c.setFillColorRGB(*_SURFACE)
            c.rect(MARGIN, y - row_h * 0.2, CONTENT_W, row_h, fill=1, stroke=0)

        if rate is not None:
            rate_str   = f"{rate * 100:.1f}%"
            contrib    = rate * 3.33
            contrib_str = f"{contrib:.2f} pts"
            # Colour the contribution by severity
            cont_rgb = _RISK_RGB.get(
                _contrib_label(contrib), _MUTED
            )
        else:
            rate_str    = "Not tested"
            contrib_str = "—"
            cont_rgb    = _MUTED

        c.setFont("Helvetica", 9)
        c.setFillColorRGB(*_TEXT)
        c.drawString(col_x[0] + 0.2 * cm, y + 0.14 * cm, attack_name)
        c.drawString(col_x[1] + 0.2 * cm, y + 0.14 * cm, rate_str)
        c.setFillColorRGB(*cont_rgb)
        c.drawString(col_x[2] + 0.2 * cm, y + 0.14 * cm, contrib_str)

        y -= row_h
        _hline(c, MARGIN, y, PAGE_W - MARGIN, rgb=_BORDER)

    # Risk contribution formula note
    y -= 0.6 * cm
    c.setFont("Helvetica-Oblique", 8)
    c.setFillColorRGB(*_MUTED)
    c.drawString(MARGIN, y,
                 "Risk Contribution = success_rate x 3.33 pts  "
                 "(each of 3 attacks contributes up to 3.33 KVS points)")

    # KVS composition note
    y -= 0.6 * cm
    total_contrib = sum(
        (r * 3.33) for _, r in attacks_data if r is not None
    )
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(*_TEXT)
    c.drawString(MARGIN, y,
                 f"Combined attack contribution: {total_contrib:.2f} pts  "
                 f"|  KVS Score: {bundle.kvs_score:.1f} / 10.0  "
                 f"[{bundle.kvs_label}]")

    _hline(c, MARGIN, 1.8 * cm, PAGE_W - MARGIN)
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=False)
    _draw_footer(c, PAGE_W)
    c.showPage()

    # ================================================================
    # PAGE 4 — Certification Statement
    # ================================================================
    _fill_bg(c, PAGE_W, PAGE_H)
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=True)
    _hline(c, MARGIN, PAGE_H - 1.8 * cm, PAGE_W - MARGIN)

    y = PAGE_H - 2.6 * cm
    _section_header(c, MARGIN, y, "CERTIFICATION STATEMENT")
    y -= 0.8 * cm

    cert_statement = _CERT_STATEMENTS.get(
        bundle.kvs_label,
        f"Model risk level is {bundle.kvs_label}. Refer to audit report for details.",
    )

    # Certification statement paragraph
    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(*risk_rgb)
    c.drawString(MARGIN, y, f"Risk Classification: {bundle.kvs_label.upper()}")
    y -= 0.7 * cm

    y = _wrapped_para(c, MARGIN, y, cert_statement,
                      max_width=CONTENT_W, rgb=_TEXT, size=10, leading=14)

    y -= 0.6 * cm
    _hline(c, MARGIN, y, PAGE_W - MARGIN, rgb=_BORDER)
    y -= 0.6 * cm

    y = _wrapped_para(c, MARGIN, y, _FRAMEWORK_STATEMENT,
                      max_width=CONTENT_W, rgb=_MUTED, size=9, leading=13)

    # Signature block
    y -= 1.2 * cm
    _subsection(c, MARGIN, y, "Certification Details")
    y -= 0.55 * cm
    sig_rows = [
        ("Report ID",    report_id),
        ("Organisation", bundle.org_name),
        ("Audit Date",   date_str),
        ("Framework",    "KAAL v1.0.0"),
        ("File Hash",    fp.file_hash[:32] + "..."),
    ]
    y = _table_rows(c, MARGIN, y, sig_rows, PAGE_W, MARGIN)

    # Stamp box
    y -= 0.8 * cm
    stamp_w = 5.5 * cm
    stamp_h = 2.0 * cm
    stamp_x = PAGE_W - MARGIN - stamp_w
    c.setStrokeColorRGB(*risk_rgb)
    c.setFillColorRGB(*_SURFACE)
    c.setLineWidth(1.5)
    c.rect(stamp_x, y - stamp_h, stamp_w, stamp_h, fill=1, stroke=1)
    c.setFont("Helvetica-Bold", 8)
    c.setFillColorRGB(*risk_rgb)
    c.drawCentredString(stamp_x + stamp_w / 2, y - 0.55 * cm, "KAAL CERTIFIED")
    c.setFont("Helvetica", 7)
    c.setFillColorRGB(*_MUTED)
    c.drawCentredString(stamp_x + stamp_w / 2, y - 0.95 * cm, bundle.kvs_label.upper())
    c.drawCentredString(stamp_x + stamp_w / 2, y - 1.30 * cm, date_str)
    c.drawCentredString(stamp_x + stamp_w / 2, y - 1.65 * cm, bundle.org_name[:22])

    _hline(c, MARGIN, 1.8 * cm, PAGE_W - MARGIN)
    _draw_classification(c, PAGE_W, PAGE_H, classification_marking, top=False)
    _draw_footer(c, PAGE_W)
    c.showPage()

    c.save()
    return output_path


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _fill_bg(c, w: float, h: float) -> None:
    c.setFillColorRGB(*_BG)
    c.rect(0, 0, w, h, fill=1, stroke=0)


def _hline(c, x1: float, y: float, x2: float,
           rgb: tuple = _BORDER, lw: float = 0.4) -> None:
    c.setStrokeColorRGB(*rgb)
    c.setLineWidth(lw)
    c.line(x1, y, x2, y)


def _draw_classification(c, page_w: float, page_h: float,
                          text: str, top: bool) -> None:
    """Draw classification marking as a centred red bold string."""
    c.setFont("Helvetica-Bold", 8)
    c.setFillColorRGB(*_RED)
    y = page_h - 0.65 * cm if top else 0.65 * cm
    c.drawCentredString(page_w / 2, y, text)


def _draw_footer(c, page_w: float) -> None:
    """Draw the KAAL-D footer text."""
    c.setFont("Courier", 7)
    c.setFillColorRGB(*_DIM)
    c.drawCentredString(page_w / 2, 0.35 * cm, _FOOTER_TEXT)


def _section_header(c, x: float, y: float, text: str) -> None:
    c.setFont("Helvetica-Bold", 12)
    c.setFillColorRGB(*_RED)
    c.drawString(x, y, text)


def _subsection(c, x: float, y: float, text: str) -> None:
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(*_MUTED)
    c.drawString(x, y, text.upper())


def _text_centred(c, x: float, y: float, text: str,
                  size: int = 10, bold: bool = False,
                  rgb: tuple = _TEXT) -> None:
    font = "Helvetica-Bold" if bold else "Helvetica"
    c.setFont(font, size)
    c.setFillColorRGB(*rgb)
    c.drawCentredString(x, y, text)


def _kv_line(c, x: float, y: float, label: str, value: str,
             label_w: float = 120, page_w: float = 595,
             margin: float = 56) -> None:
    c.setFont("Helvetica-Bold", 9)
    c.setFillColorRGB(*_MUTED)
    c.drawString(x, y, label + ":")
    c.setFont("Helvetica", 9)
    c.setFillColorRGB(*_TEXT)
    c.drawString(x + label_w, y, value)


def _table_rows(c, x: float, y: float, rows: list,
                page_w: float, margin: float) -> float:
    """Draw label-value rows. Returns updated y position."""
    row_h  = 0.58 * cm
    val_x  = x + 4.5 * cm

    for i, row in enumerate(rows):
        label = row[0]
        value = row[1]
        val_rgb = row[2] if len(row) > 2 else _TEXT

        if i % 2 == 0:
            c.setFillColorRGB(*_SURFACE)
            c.rect(x - 0.1 * cm, y - row_h * 0.2,
                   page_w - 2 * margin + 0.2 * cm, row_h, fill=1, stroke=0)

        c.setFont("Helvetica-Bold", 8)
        c.setFillColorRGB(*_MUTED)
        c.drawString(x, y + 0.1 * cm, label)

        c.setFont("Courier", 8)
        c.setFillColorRGB(*val_rgb)
        c.drawString(val_x, y + 0.1 * cm, str(value))

        y -= row_h
        _hline(c, x - 0.1 * cm, y, page_w - margin, rgb=_BORDER, lw=0.3)

    return y - 0.1 * cm


def _wrapped_para(c, x: float, y: float, text: str,
                  max_width: float, rgb: tuple = _TEXT,
                  size: int = 9, leading: int = 13) -> float:
    """Draw word-wrapped paragraph. Returns updated y."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    c.setFont("Helvetica", size)
    c.setFillColorRGB(*rgb)

    words  = text.split()
    line   = ""
    for word in words:
        test = (line + " " + word).strip()
        if stringWidth(test, "Helvetica", size) <= max_width:
            line = test
        else:
            c.drawString(x, y, line)
            y -= leading / 72 * 72   # leading in pts
            line = word
    if line:
        c.drawString(x, y, line)
        y -= leading / 72 * 72

    return y


def _kvs_cover_badge(c, page_w: float, risk_rgb: tuple,
                     kvs_score: float, kvs_label: str) -> None:
    """Draw the KVS score prominently in the lower half of the cover page."""
    cx = page_w / 2
    y  = 10 * cm

    # Circle
    c.setStrokeColorRGB(*risk_rgb)
    c.setFillColorRGB(*_SURFACE)
    c.setLineWidth(2)
    c.circle(cx, y, 2.5 * cm, fill=1, stroke=1)

    # Score text
    c.setFont("Helvetica-Bold", 28)
    c.setFillColorRGB(*risk_rgb)
    c.drawCentredString(cx, y + 0.15 * cm, f"{kvs_score:.1f}")

    # Label below circle
    c.setFont("Helvetica-Bold", 10)
    c.setFillColorRGB(*risk_rgb)
    c.drawCentredString(cx, y - 3.3 * cm, kvs_label.upper())

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(*_MUTED)
    c.drawCentredString(cx, y - 3.8 * cm, "KVS Score (0.0 – 10.0)")


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _fmt_date(iso_ts: str) -> str:
    """Format ISO 8601 timestamp as DD MMM YYYY."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        return dt.strftime("%d %b %Y")
    except Exception:
        return iso_ts[:10]


def _fmt_date_compact(iso_ts: str) -> str:
    """Format ISO 8601 timestamp as YYYYMMDD."""
    try:
        dt = datetime.fromisoformat(iso_ts)
        return dt.strftime("%Y%m%d")
    except Exception:
        return iso_ts[:10].replace("-", "")


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else "..." + s[-(n - 3):]


def _wrap_hash(h: str) -> str:
    """Split 64-char hash into two 32-char lines separated by newline marker."""
    if len(h) <= 32:
        return h
    return h[:32] + "  " + h[32:]


def _contrib_label(contrib: float) -> str:
    """Map a risk contribution value to a KVS label for colour coding."""
    if contrib <= 0.67:  return "Robust"
    if contrib <= 1.33:  return "Low Risk"
    if contrib <= 2.00:  return "Medium Risk"
    if contrib <= 2.67:  return "High Risk"
    return "Critical"
