"""
PDF report generation.

Reuses the exact ReportLab layout/logic from the original Gradio app.py's
generate_pdf(): title, side-by-side original + Grad-CAM images, prediction
result, probability table, disclaimer.

The only change is the calling convention — this takes explicit typed
arguments (built by the API route from the /predict response + the
original uploaded image) instead of a single Gradio gr.State dict.
"""

import base64
import io
import os
import uuid
from datetime import datetime

from PIL import Image
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from app.config import STATIC_TEMP_DIR


def generate_pdf_report(
    model_name: str,
    pred_class: str,
    confidence: float,
    probabilities: dict[str, float],
    original_pil_image: Image.Image,
    gradcam_base64: str,
) -> str:
    """
    Build the PDF report and save it to app/static/temp/.
    Returns the absolute file path (the route layer turns this into a
    FileResponse / download).
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("ReportTitle", parent=styles["Title"], fontSize=18, spaceAfter=4)
    elements = []

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    elements.append(Paragraph("🐟 Fish Species Classification Report", title_style))
    elements.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
    elements.append(Paragraph(f"Model used: <b>{model_name}</b>", styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Original image
    orig_buf = io.BytesIO()
    original_pil_image.convert("RGB").save(orig_buf, format="PNG")
    orig_buf.seek(0)

    # Grad-CAM image (decoded from Base64 sent in by the route)
    # cam_bytes = base64.b64decode(gradcam_base64)
    # cam_buf = io.BytesIO(cam_bytes)

    # img_table = Table([[
    #     RLImage(orig_buf, width=70 * mm, height=70 * mm),
    #     RLImage(cam_buf, width=70 * mm, height=70 * mm),
    # ]])
    # img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    # elements.append(img_table)

    # caption_table = Table([["Original Image (resized)", "Grad-CAM Heatmap (XAI)"]],
    #                        colWidths=[75 * mm, 75 * mm])
    # caption_table.setStyle(TableStyle([
    #     ("ALIGN", (0, 0), (-1, -1), "CENTER"),
    #     ("FONTSIZE", (0, 0), (-1, -1), 9),
    #     ("TEXTCOLOR", (0, 0), (-1, -1), colors.grey),
    # ]))
    # elements.append(caption_table)
    
    
    img_table = Table([[
        RLImage(orig_buf, width=90 * mm, height=90 * mm),
        # RLImage(cam_buf, width=70 * mm, height=70 * mm),
    ]])
    img_table.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    elements.append(img_table)

    caption_table = Table([["Original Image (resized)"]],
                           colWidths=[100 * mm])
    caption_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.grey),
    ]))
    elements.append(caption_table)
    
   
    
    elements.append(Spacer(1, 16))

    elements.append(Paragraph("Prediction Result", styles["Heading2"]))
    elements.append(Paragraph(f"Predicted species: <b>{pred_class}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: <b>{confidence:.2%}</b>", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Class Probability Breakdown", styles["Heading2"]))
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    data = [["Species", "Probability"]] + [[k, f"{v:.2%}"] for k, v in sorted_probs]
    t = Table(data, colWidths=[100 * mm, 40 * mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2196F3")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F5F5")]),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
    ]))
    elements.append(t)
    elements.append(Spacer(1, 16))

    disclaimer = (
        "Disclaimer: This report is generated automatically by a deep-learning "
        "image classification system for research/demonstration purposes. "
        "The Grad-CAM heatmap highlights the image regions that most influenced "
        "the model's decision and should be treated as an explainability aid, "
        "not a guarantee of correctness."
    )
    elements.append(Paragraph(disclaimer, styles["Italic"]))

    doc.build(elements)
    buf.seek(0)

    safe_name = pred_class.replace(" ", "_")
    unique_suffix = uuid.uuid4().hex[:8]
    filename = f"fish_report_{safe_name}_{unique_suffix}.pdf"
    out_path = os.path.join(STATIC_TEMP_DIR, filename)

    with open(out_path, "wb") as f:
        f.write(buf.read())

    return out_path