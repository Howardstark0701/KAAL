"""PDF Report Generator — Spec 14.1 — Phase 8, Kiro Prompt 8.2."""
from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional, Any

from kaal.engine.utils import ensure_dir

# ── Colour constants (reportlab RGB 0-1) ─────────────────────────────────────
BG      = (0.039, 0.039, 0.039)   # #0A0A0A
SURFACE = (0.067, 0.067, 0.067)   # #111111
RED     = (0.800, 0.000, 0.000)   # #CC0000
TEXT    = (0.949, 0.949, 0.949)   # #F2F2F2
MUTED   = (0.400, 0.400, 0.400)   # #666666
BORDER  = (0.122, 0.122, 0.122)   # #1F1F1F

# ── KVS score colours ─────────────────────────────────────────────────────────
def _kvs_rgb(score):
    if score is None:
        return MUTED
    score = float(score)
    if score <= 2.0:  return (0.290, 0.867, 0.502)  # #4ADE80 green
    if score <= 4.0:  return (0.639, 0.898, 0.208)  # #A3E635
    if score <= 6.0:  return (0.980, 0.800, 0.082)  # #FACC15
    if score <= 8.0:  return (0.984, 0.573, 0.188)  # #FB923C
    return RED                                        # #CC0000

# ── Main entry point ─────────────────────────────────────────────────────────
def generate_pdf_report(
    output_path: str,
    model_info: dict,
    dataset_info: dict,
    kvs_result=None,
    fgsm_result=None,
    pgd_result=None,
    patch_result=None,
    physical_result=None,
    gradcam_comparison=None,
    collapse_curve_path: Optional[str] = None,
    fingerprint_path: Optional[str] = None,
    audit_duration_seconds: float = 0.0,
    kaal_version: str = "1.0.0",
) -> str:
    """Generate an 8-page KAAL audit PDF report.

    Args:
        output_path:            Full path for the output PDF.
        model_info:             Dict: path, name, framework, input_shape, num_classes.
        dataset_info:           Dict: path, total_images, formats.
        kvs_result:             KVSResult from calculate_kvs().
        fgsm_result:            FGSMResult or aggregate dict.
        pgd_result:             PGDResult or aggregate dict.
        patch_result:           PatchResult from generate_patch().
        physical_result:        PhysicalRobustnessResult.
        gradcam_comparison:     GradCAMComparisonResult (optional).
        collapse_curve_path:    Path to PGD collapse curve PNG (optional).
        fingerprint_path:       Path to radar chart PNG (optional).
        audit_duration_seconds: Audit wall-clock time.
        kaal_version:           KAAL version string.

    Returns:
        Absolute path to the saved PDF file.
    """
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas as rl_canvas
        from reportlab.lib.units import cm
        from reportlab.lib.utils import ImageReader
    except ImportError:
        raise ImportError(
            "reportlab is not installed.\n"
            "→ pip install reportlab==4.1.0"
        )

    output_path = os.path.abspath(output_path)
    ensure_dir(os.path.dirname(output_path))

    PAGE_W, PAGE_H = A4
    c = rl_canvas.Canvas(output_path, pagesize=A4)

    ctx = _ReportContext(
        c=c, W=PAGE_W, H=PAGE_H,
        model_info=model_info,
        dataset_info=dataset_info,
        kvs_result=kvs_result,
        fgsm_result=fgsm_result,
        pgd_result=pgd_result,
        patch_result=patch_result,
        physical_result=physical_result,
        gradcam_comparison=gradcam_comparison,
        collapse_curve_path=collapse_curve_path,
        fingerprint_path=fingerprint_path,
        audit_duration_seconds=audit_duration_seconds,
        kaal_version=kaal_version,
        cm=cm,
        ImageReader=ImageReader,
    )

    _page1_cover(ctx)
    _page2_summary(ctx)
    _page3_fgsm(ctx)
    _page4_pgd(ctx)
    _page5_patch(ctx)
    _page6_physical(ctx)
    _page7_fingerprint(ctx)
    _page8_appendix(ctx)

    c.save()
    return output_path


# ── Report context (passed to each page builder) ─────────────────────────────
class _ReportContext:
    """Thin container so page builders don't take 15 arguments."""
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# ── Drawing primitives ────────────────────────────────────────────────────────
def _bg(c, W, H):
    """Fill entire page with KAAL background colour."""
    c.setFillColorRGB(*BG)
    c.rect(0, 0, W, H, fill=1, stroke=0)


def _section_heading(c, x, y, text, font_size=9):
    """Small all-caps muted section label."""
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", font_size)
    c.drawString(x, y, text.upper())


def _divider(c, x, y, width):
    """Thin horizontal rule in border colour."""
    c.setStrokeColorRGB(*BORDER)
    c.setLineWidth(0.4)
    c.line(x, y, x + width, y)


def _label_value(c, x, y, label, value, label_w=110):
    """Key-value pair: muted label + white value on same line."""
    c.setFillColorRGB(*MUTED)
    c.setFont("Helvetica", 8)
    c.drawString(x, y, label)
    c.setFillColorRGB(*TEXT)
    c.setFont("Courier", 8)
    c.drawString(x + label_w, y, str(value))


def _embed_image(c, path_or_pil, x, y, max_w, max_h, ImageReader):
    """Embed a PIL Image or file path. Returns (drawn_w, drawn_h) or (0,0)."""
    if path_or_pil is None:
        return 0, 0
    try:
        import PIL.Image as PILImage
        if isinstance(path_or_pil, PILImage.Image):
            buf = io.BytesIO()
            path_or_pil.save(buf, format="PNG")
            buf.seek(0)
            reader = ImageReader(buf)
        else:
            if not os.path.exists(str(path_or_pil)):
                return 0, 0
            reader = ImageReader(str(path_or_pil))
        iw, ih = reader.getSize()
        scale = min(max_w / iw, max_h / ih)
        dw, dh = iw * scale, ih * scale
        c.drawImage(reader, x, y - dh, width=dw, height=dh, mask="auto")
        return dw, dh
    except Exception:
        return 0, 0


# ── PAGE 1 — Cover ────────────────────────────────────────────────────────────
def _page1_cover(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)

    kvs_score = _safe(ctx.kvs_result, "score", 0.0)
    kvs_score = float(kvs_score) if kvs_score is not None else 0.0
    kvs_label = _safe(ctx.kvs_result, "label", "N/A") or "N/A"
    kvs_color = _kvs_rgb(kvs_score)
    model_name = ctx.model_info.get("name", "Unknown Model")
    audit_ts = datetime.now().strftime("%Y-%m-%d  %H:%M UTC")

    # Top bar
    c.setFillColorRGB(*SURFACE)
    c.rect(0, H - 1.2*cm, W, 1.2*cm, fill=1, stroke=0)
    c.setFillColorRGB(*RED)
    c.setFont("Courier", 9)
    c.drawString(1*cm, H - 0.75*cm, "KAAL  v" + ctx.kaal_version)

    # Main title
    c.setFillColorRGB(*TEXT)
    c.setFont("Helvetica-Bold", 18)
    c.drawCentredString(W/2, H*0.72, "ADVERSARIAL VULNERABILITY")
    c.drawCentredString(W/2, H*0.72 - 1.1*cm, "AUDIT REPORT")

    _divider(c, 2*cm, H*0.68, W - 4*cm)

    # Model info
    c.setFont("Courier", 10)
    c.setFillColorRGB(*MUTED)
    c.drawCentredString(W/2, H*0.64, "Model")
    c.setFillColorRGB(*TEXT)
    c.setFont("Courier-Bold", 11)
    c.drawCentredString(W/2, H*0.61, model_name)

    c.setFont("Courier", 8)
    c.setFillColorRGB(*MUTED)
    c.drawCentredString(W/2, H*0.58, audit_ts)

    # KVS score — large, colour-coded
    c.setFont("Helvetica-Bold", 64)
    c.setFillColorRGB(*kvs_color)
    c.drawCentredString(W/2, H*0.42, f"{kvs_score:.1f}")

    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(W/2, H*0.38, kvs_label.upper())

    c.setFont("Helvetica", 8)
    c.setFillColorRGB(*MUTED)
    c.drawCentredString(W/2, H*0.35, "KAAL VULNERABILITY SCORE  (0 = Robust  /  10 = Catastrophic)")

    # Tagline
    _divider(c, 2*cm, H*0.10, W - 4*cm)
    c.setFont("Helvetica", 8)
    c.setFillColorRGB(*MUTED)
    c.drawCentredString(W/2, H*0.07, '"What cannot be seen, cannot be defended."')

    c.showPage()


# ── PAGE 2 — Audit Summary ────────────────────────────────────────────────────
def _page2_summary(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)
    _page_header(c, W, H, cm, "AUDIT SUMMARY", ctx.kaal_version)
    y = H - 2.5*cm

    # Attack parameters table
    _section_heading(c, 1.5*cm, y, "AUDIT PARAMETERS")
    y -= 0.6*cm
    attacks = _get_attack_params(ctx)
    col_x = [1.5*cm, 5.5*cm, 9.5*cm, 13.5*cm]
    headers = ["Attack", "Epsilon", "Steps", "Success Rate"]
    c.setFont("Courier-Bold", 8)
    c.setFillColorRGB(*MUTED)
    for i, h in enumerate(headers):
        c.drawString(col_x[i], y, h)
    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.45*cm

    c.setFont("Courier", 8)
    for row in attacks:
        c.setFillColorRGB(*TEXT)
        for i, val in enumerate(row):
            c.drawString(col_x[i], y, str(val))
        y -= 0.45*cm

    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.8*cm

    # Overall findings
    _section_heading(c, 1.5*cm, y, "OVERALL FINDINGS")
    y -= 0.6*cm
    kvs = ctx.kvs_result
    if kvs is not None:
        score = _safe(kvs, "score", 0.0)
        label = _safe(kvs, "label", "N/A")
        color = _kvs_rgb(score)
        c.setFont("Helvetica-Bold", 22)
        c.setFillColorRGB(*color)
        c.drawString(1.5*cm, y, f"{score:.1f}")
        c.setFont("Helvetica", 11)
        c.drawString(3.0*cm, y + 0.1*cm, label)
        y -= 0.8*cm

        dim_scores = getattr(kvs, "dimension_scores", {})
        for dim, score_val in dim_scores.items():
            bar_w = (score_val / 10.0) * 5*cm
            c.setFillColorRGB(*_kvs_rgb(score_val))
            c.rect(1.5*cm, y - 0.1*cm, bar_w, 0.28*cm, fill=1, stroke=0)
            c.setFillColorRGB(*MUTED)
            c.setFont("Courier", 7)
            c.drawString(1.5*cm, y + 0.22*cm, dim.replace("_", " ").title())
            c.setFillColorRGB(*TEXT)
            c.drawString(7.0*cm, y + 0.1*cm, f"{score_val:.1f}")
            y -= 0.5*cm

    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.8*cm

    # Remediation
    _section_heading(c, 1.5*cm, y, "RECOMMENDED ACTIONS")
    y -= 0.6*cm
    remediations = getattr(kvs, "remediation", []) if kvs else []
    for i, rem in enumerate(remediations, 1):
        c.setFillColorRGB(*RED)
        c.setFont("Courier-Bold", 8)
        c.drawString(1.5*cm, y, f"{i}.")
        c.setFillColorRGB(*TEXT)
        c.setFont("Courier", 7.5)
        _wrap_text(c, 2.2*cm, y, rem, W - 3.7*cm, 7.5)
        y -= 0.65*cm

    c.showPage()


# ── PAGE 3 — FGSM Results ─────────────────────────────────────────────────────
def _page3_fgsm(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)
    _page_header(c, W, H, cm, "FGSM ATTACK", ctx.kaal_version)
    y = H - 2.5*cm
    fgsm = ctx.fgsm_result

    # Parameters
    _section_heading(c, 1.5*cm, y, "ATTACK PARAMETERS")
    y -= 0.55*cm
    eps = _get_val(fgsm, "epsilon_used", "epsilon", default="0.03")
    sr  = _get_val(fgsm, "success_rate", default="—")
    cd  = _get_val(fgsm, "avg_confidence_delta", "confidence_delta", default="—")
    _label_value(c, 1.5*cm, y, "Epsilon (ε)", str(eps))
    y -= 0.45*cm
    _label_value(c, 1.5*cm, y, "Success Rate",
                 f"{float(sr):.1%}" if _is_num(sr) else str(sr))
    y -= 0.45*cm
    _label_value(c, 1.5*cm, y, "Avg Confidence Drop",
                 f"{float(cd):+.4f}" if _is_num(cd) else str(cd))
    y -= 0.7*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.7*cm

    # Adversarial image (if available)
    _section_heading(c, 1.5*cm, y, "ADVERSARIAL EXAMPLE")
    y -= 0.5*cm
    adv_pil = _safe(fgsm, "adversarial_pil", None)
    if adv_pil is not None:
        cx = 1.5*cm
        dw, dh = _embed_image(c, adv_pil, cx, y, 7*cm, 5*cm, ctx.ImageReader)
        if dh > 0:
            c.setFont("Courier", 7)
            c.setFillColorRGB(*MUTED)
            adv_class = _safe(fgsm, "adversarial_class", "?")
            adv_conf  = _safe(fgsm, "adversarial_confidence", 0.0)
            c.drawString(cx, y - dh - 0.2*cm,
                         f"Predicted: class {adv_class}  ({float(adv_conf):.2%})")
            y -= (dh + 0.7*cm)

    # GradCAM comparison
    gcam = ctx.gradcam_comparison
    if gcam is not None:
        _section_heading(c, 1.5*cm, y, "GRADCAM ATTENTION  (Clean vs Adversarial)")
        y -= 0.5*cm
        sbs = getattr(gcam, "side_by_side_pil", None)
        dw, dh = _embed_image(c, sbs, 1.5*cm, y, W - 3*cm, 5*cm, ctx.ImageReader)
        if dh > 0:
            shift = getattr(gcam, "attention_shift_score", 0.0)
            c.setFont("Courier", 7)
            c.setFillColorRGB(*MUTED)
            c.drawString(1.5*cm, y - dh - 0.2*cm,
                         f"Attention shift score: {shift:.4f}")
            y -= (dh + 0.7*cm)

    # Plain English finding
    pe = _safe(fgsm, "plain_english", "")
    if pe:
        _divider(c, 1.5*cm, y, W - 3*cm)
        y -= 0.5*cm
        _section_heading(c, 1.5*cm, y, "FINDING")
        y -= 0.5*cm
        c.setFillColorRGB(*TEXT)
        c.setFont("Helvetica", 8)
        _wrap_text(c, 1.5*cm, y, pe, W - 3*cm, 8)

    c.showPage()


# ── PAGE 4 — PGD Results ──────────────────────────────────────────────────────
def _page4_pgd(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)
    _page_header(c, W, H, cm, "PGD ATTACK", ctx.kaal_version)
    y = H - 2.5*cm
    pgd = ctx.pgd_result

    _section_heading(c, 1.5*cm, y, "ATTACK PARAMETERS")
    y -= 0.55*cm
    for label, keys in [
        ("Epsilon (ε)",   ("epsilon_used",)),
        ("Alpha (α)",     ("alpha_used",)),
        ("Steps",         ("steps_used",)),
        ("Success Rate",  ("success_rate",)),
        ("Steps to Success", ("steps_to_success", "avg_steps_to_success")),
    ]:
        val = _get_val(pgd, *keys, default="—")
        if label == "Success Rate" and _is_num(val):
            val = f"{float(val):.1%}"
        _label_value(c, 1.5*cm, y, label, str(val))
        y -= 0.45*cm

    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.7*cm

    # Confidence collapse curve chart
    if ctx.collapse_curve_path and os.path.exists(ctx.collapse_curve_path):
        _section_heading(c, 1.5*cm, y, "CONFIDENCE COLLAPSE CURVE")
        y -= 0.5*cm
        dw, dh = _embed_image(c, ctx.collapse_curve_path,
                               1.5*cm, y, W - 3*cm, 7*cm, ctx.ImageReader)
        y -= (dh + 0.7*cm)

    pe = _safe(pgd, "plain_english", "")
    if pe:
        _divider(c, 1.5*cm, y, W - 3*cm)
        y -= 0.5*cm
        _section_heading(c, 1.5*cm, y, "FINDING")
        y -= 0.5*cm
        c.setFillColorRGB(*TEXT)
        c.setFont("Helvetica", 8)
        _wrap_text(c, 1.5*cm, y, pe, W - 3*cm, 8)

    c.showPage()


# ── PAGE 5 — Patch Results ────────────────────────────────────────────────────
def _page5_patch(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)
    _page_header(c, W, H, cm, "ADVERSARIAL PATCH", ctx.kaal_version)
    y = H - 2.5*cm
    patch = ctx.patch_result

    if patch is None:
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 10)
        c.drawCentredString(W/2, H/2, "Patch attack not run.")
        c.showPage()
        return

    # Patch image
    _section_heading(c, 1.5*cm, y, "TRAINED PATCH")
    y -= 0.5*cm
    patch_pil = getattr(patch, "patch_pil", None)
    img_size = 6*cm
    dw, dh = _embed_image(c, patch_pil, 1.5*cm, y, img_size, img_size, ctx.ImageReader)
    right_x = 1.5*cm + img_size + 1*cm

    # Stats beside image
    stats = [
        ("Target Class",  str(_safe(patch, "target_class", "?"))),
        ("Success Rate",  f"{_safe(patch, 'attack_success_rate', 0):.1%}"),
        ("Avg Confidence", f"{_safe(patch, 'avg_confidence_on_target', 0):.3f}"),
        ("Patch Fraction", f"{_safe(patch, 'patch_fraction_used', 0)*100:.1f}%"),
        ("Iterations",    str(_safe(patch, "iterations_used", 0))),
    ]
    sy = y
    for lbl, val in stats:
        _label_value(c, right_x, sy, lbl, val)
        sy -= 0.5*cm

    y -= max(dh, (len(stats) * 0.5*cm)) + 0.8*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.7*cm

    # Print instructions
    _section_heading(c, 1.5*cm, y, "PRINT INSTRUCTIONS")
    y -= 0.5*cm
    for line in [
        "1. Open patch_print.pdf in any PDF viewer.",
        "2. Print at 100% scale (disable 'Fit to page').",
        "3. Use the corner calibration marks to verify physical size.",
        "4. Cut out the patch and place in the camera field of view.",
    ]:
        c.setFillColorRGB(*TEXT)
        c.setFont("Helvetica", 8)
        c.drawString(1.5*cm, y, line)
        y -= 0.45*cm

    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.5*cm

    pe = _safe(patch, "plain_english", "")
    if pe:
        _section_heading(c, 1.5*cm, y, "FINDING")
        y -= 0.5*cm
        c.setFillColorRGB(*TEXT)
        c.setFont("Helvetica", 8)
        _wrap_text(c, 1.5*cm, y, pe, W - 3*cm, 8)

    c.showPage()


# ── PAGE 6 — Physical Robustness ──────────────────────────────────────────────
def _page6_physical(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)
    _page_header(c, W, H, cm, "PHYSICAL ROBUSTNESS", ctx.kaal_version)
    y = H - 2.5*cm
    phys = ctx.physical_result

    if phys is None:
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 10)
        c.drawCentredString(W/2, H/2, "Physical robustness test not run.")
        c.showPage()
        return

    overall = _safe(phys, "overall_survival_rate", 0.0)
    rating  = _safe(phys, "physical_threat_rating", "—")
    rating_color = _rating_rgb(rating)

    _label_value(c, 1.5*cm, y, "Overall Survival Rate", f"{float(overall):.1%}")
    y -= 0.5*cm

    c.setFillColorRGB(*rating_color)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(1.5*cm, y, f"Physical Threat Rating:  {rating}")
    y -= 0.8*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.7*cm

    # Per-transform table
    _section_heading(c, 1.5*cm, y, "TRANSFORM RESULTS")
    y -= 0.5*cm
    col_x = [1.5*cm, 7*cm, 11*cm, 14*cm]
    c.setFont("Courier-Bold", 7.5)
    c.setFillColorRGB(*MUTED)
    for x_, h in zip(col_x, ["Transform", "Category", "Survival", "Status"]):
        c.drawString(x_, y, h)
    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.4*cm

    per = getattr(phys, "per_transform_results", {})
    c.setFont("Courier", 7)
    for name, tr in sorted(per.items()):
        if y < 2*cm:
            break
        sr = getattr(tr, "success_rate", 0.0)
        cat = getattr(tr, "category", "")
        status_color = (0.29, 0.87, 0.50) if sr >= 0.5 else RED
        c.setFillColorRGB(*TEXT)
        c.drawString(col_x[0], y, name)
        c.drawString(col_x[1], y, cat[:16])
        c.drawString(col_x[2], y, f"{sr:.0%}")
        c.setFillColorRGB(*status_color)
        c.drawString(col_x[3], y, "SURVIVES" if sr > 0 else "BROKEN")
        y -= 0.38*cm

    pe = _safe(phys, "plain_english", "")
    if pe and y > 2*cm:
        _divider(c, 1.5*cm, y - 0.3*cm, W - 3*cm)
        y -= 0.8*cm
        _section_heading(c, 1.5*cm, y, "FINDING")
        y -= 0.5*cm
        c.setFillColorRGB(*TEXT)
        c.setFont("Helvetica", 8)
        _wrap_text(c, 1.5*cm, y, pe, W - 3*cm, 8)

    c.showPage()


# ── PAGE 7 — Vulnerability Fingerprint ───────────────────────────────────────
def _page7_fingerprint(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)
    _page_header(c, W, H, cm, "VULNERABILITY FINGERPRINT", ctx.kaal_version)
    y = H - 2.5*cm

    if ctx.fingerprint_path and os.path.exists(ctx.fingerprint_path):
        dw, dh = _embed_image(
            c, ctx.fingerprint_path,
            1.5*cm, y, W - 3*cm, H * 0.65, ctx.ImageReader,
        )
        y -= (dh + 0.6*cm)
    else:
        c.setFillColorRGB(*MUTED)
        c.setFont("Helvetica", 9)
        c.drawCentredString(W/2, y - 3*cm, "Fingerprint chart not available.")
        y -= 4*cm

    # Per-dimension one-liners
    kvs = ctx.kvs_result
    if kvs is not None and y > 2*cm:
        _divider(c, 1.5*cm, y, W - 3*cm)
        y -= 0.6*cm
        for dim, score_val in getattr(kvs, "dimension_scores", {}).items():
            if y < 2*cm:
                break
            label = dim.replace("_", " ").title()
            c.setFillColorRGB(*_kvs_rgb(score_val))
            c.setFont("Courier-Bold", 8)
            c.drawString(1.5*cm, y, f"{score_val:4.1f}")
            c.setFillColorRGB(*MUTED)
            c.setFont("Courier", 8)
            c.drawString(2.8*cm, y, label)
            y -= 0.42*cm

    c.showPage()


# ── PAGE 8 — Technical Appendix ──────────────────────────────────────────────
def _page8_appendix(ctx):
    c, W, H, cm = ctx.c, ctx.W, ctx.H, ctx.cm
    _bg(c, W, H)
    _page_header(c, W, H, cm, "TECHNICAL APPENDIX", ctx.kaal_version)
    y = H - 2.5*cm

    import sys, platform

    _section_heading(c, 1.5*cm, y, "ENVIRONMENT")
    y -= 0.55*cm
    env_rows = [
        ("KAAL Version",    ctx.kaal_version),
        ("Python",          sys.version.split()[0]),
        ("Platform",        platform.system() + " " + platform.release()),
        ("Audit Duration",  _fmt_duration(ctx.audit_duration_seconds)),
    ]
    # Try to get torch version
    try:
        import torch
        env_rows.append(("PyTorch", torch.__version__))
    except Exception:
        pass
    for lbl, val in env_rows:
        _label_value(c, 1.5*cm, y, lbl, str(val))
        y -= 0.45*cm

    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.7*cm

    _section_heading(c, 1.5*cm, y, "MODEL INFORMATION")
    y -= 0.55*cm
    mi = ctx.model_info
    for lbl, key in [
        ("Model Path",   "path"),
        ("Model Name",   "name"),
        ("Framework",    "framework"),
        ("Input Shape",  "input_shape"),
        ("Num Classes",  "num_classes"),
    ]:
        _label_value(c, 1.5*cm, y, lbl, str(mi.get(key, "—")))
        y -= 0.45*cm

    y -= 0.4*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.7*cm

    _section_heading(c, 1.5*cm, y, "DATASET STATISTICS")
    y -= 0.55*cm
    di = ctx.dataset_info
    _label_value(c, 1.5*cm, y, "Dataset Path",   str(di.get("path", "—")))
    y -= 0.45*cm
    _label_value(c, 1.5*cm, y, "Total Images",   str(di.get("total_images", "—")))
    y -= 0.45*cm
    fmt_str = "  ".join(
        f"{ext}={n}" for ext, n in di.get("formats", {}).items()
    )
    _label_value(c, 1.5*cm, y, "Formats",        fmt_str or "—")
    y -= 0.7*cm
    _divider(c, 1.5*cm, y, W - 3*cm)
    y -= 0.7*cm

    # Full attack parameters
    _section_heading(c, 1.5*cm, y, "FULL ATTACK PARAMETERS")
    y -= 0.55*cm
    for row in _get_attack_params(ctx):
        if y < 2*cm:
            break
        c.setFont("Courier", 7.5)
        c.setFillColorRGB(*TEXT)
        c.drawString(1.5*cm, y,
                     f"{row[0]:<12}  ε={row[1]}  steps={row[2]}  success={row[3]}")
        y -= 0.4*cm

    c.showPage()


# ── Shared helpers ────────────────────────────────────────────────────────────
def _page_header(c, W, H, cm, title, version):
    """Consistent top bar on every page after cover."""
    c.setFillColorRGB(*SURFACE)
    c.rect(0, H - 1.0*cm, W, 1.0*cm, fill=1, stroke=0)
    c.setFillColorRGB(*RED)
    c.setFont("Courier-Bold", 8)
    c.drawString(1*cm, H - 0.65*cm, "KAAL")
    c.setFillColorRGB(*TEXT)
    c.setFont("Helvetica-Bold", 9)
    c.drawCentredString(W/2, H - 0.65*cm, title)
    c.setFillColorRGB(*MUTED)
    c.setFont("Courier", 7)
    c.drawRightString(W - 1*cm, H - 0.65*cm, f"v{version}")


def _wrap_text(c, x, y, text, max_w, font_size, line_height=None):
    """Naïve word-wrap for a single paragraph."""
    if line_height is None:
        line_height = font_size * 1.5
    words = text.split()
    line, lines = [], []
    avg_char_w = font_size * 0.55
    max_chars = int(max_w / avg_char_w)
    for word in words:
        test = " ".join(line + [word])
        if len(test) <= max_chars:
            line.append(word)
        else:
            lines.append(" ".join(line))
            line = [word]
    if line:
        lines.append(" ".join(line))
    for l in lines:
        c.drawString(x, y, l)
        y -= line_height


def _safe(obj, *attrs, default=None):
    """Try multiple attribute names, return default if all fail."""
    if obj is None:
        return default
    for attr in attrs:
        v = getattr(obj, attr, None)
        if v is None and isinstance(obj, dict):
            v = obj.get(attr)
        if v is not None:
            return v
    return default


def _get_val(obj, *keys, default="—"):
    """Same as _safe but for one object with multiple fallback keys."""
    return _safe(obj, *keys, default=default)


def _is_num(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _rating_rgb(rating: str):
    if rating == "Field Ready":
        return RED
    if rating == "Limited":
        return (0.984, 0.573, 0.188)   # orange
    return (0.290, 0.867, 0.502)       # green = Lab Only (not a real threat)


def _fmt_duration(seconds: float) -> str:
    if seconds <= 0:
        return "—"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _get_attack_params(ctx) -> list:
    """Build rows for the attacks table: [name, epsilon, steps, success_rate]."""
    rows = []
    for name, result in [
        ("FGSM",     ctx.fgsm_result),
        ("PGD",      ctx.pgd_result),
        ("Patch",    ctx.patch_result),
    ]:
        if result is None:
            continue
        eps  = _get_val(result, "epsilon_used", default="—")
        stps = _get_val(result, "steps_used", "iterations_used", default="—")
        sr   = _get_val(result, "success_rate", "attack_success_rate", default="—")
        if _is_num(sr):
            sr = f"{float(sr):.1%}"
        rows.append([name, eps, stps, sr])
    return rows
