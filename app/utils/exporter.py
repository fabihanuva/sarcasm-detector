"""
app/utils/exporter.py
---------------------
Generate PDF or plain-text analysis reports.
"""

import io
import json
from datetime import datetime


def export_as_text(records: list) -> bytes:
    """Return a UTF-8 plain-text report as bytes."""
    lines = [
        "═" * 60,
        "     SARCASM DETECTOR — ANALYSIS REPORT",
        f"     Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
        "═" * 60,
        "",
    ]
    for i, r in enumerate(records, 1):
        lines += [
            f"[{i}]  {r['created_at'][:19]}",
            f"  Text    : {r['input_text'][:120]}{'…' if len(r['input_text'])>120 else ''}",
            f"  Score   : {r['score']}%",
            f"  Result  : {r['label']}  (risk: {r['risk']})",
            f"  Markers : {', '.join(r['markers']) if r['markers'] else 'none'}",
            "",
        ]
    return "\n".join(lines).encode("utf-8")


def export_as_pdf(records: list) -> bytes:
    """Return a PDF report as bytes using ReportLab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.enums import TA_CENTER

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"],
        fontSize=18, spaceAfter=6, textColor=colors.HexColor("#1a1a2e"),
        alignment=TA_CENTER
    )
    sub_style = ParagraphStyle(
        "Sub", parent=styles["Normal"],
        fontSize=9, textColor=colors.grey, alignment=TA_CENTER, spaceAfter=16
    )
    body_style = ParagraphStyle(
        "Body2", parent=styles["Normal"], fontSize=9, leading=13
    )

    story = [
        Paragraph("Sarcasm Detector — Analysis Report", title_style),
        Paragraph(
            f"Generated {datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')} | "
            f"{len(records)} record(s)",
            sub_style,
        ),
    ]

    # Table header
    header = ["#", "Text (truncated)", "Score", "Result", "Risk"]
    rows   = [header]
    for i, r in enumerate(records, 1):
        rows.append([
            str(i),
            r["input_text"][:60] + ("…" if len(r["input_text"]) > 60 else ""),
            f"{r['score']}%",
            r["label"],
            r["risk"].upper(),
        ])

    tbl = Table(rows, colWidths=[1*cm, 9*cm, 2*cm, 2.5*cm, 2*cm])
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1,  0), colors.HexColor("#1a1a2e")),
        ("TEXTCOLOR",    (0, 0), (-1,  0), colors.white),
        ("FONTNAME",     (0, 0), (-1,  0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.HexColor("#f9f9f9"), colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",   (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 4),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 0.5*cm))

    # Detail paragraphs
    for i, r in enumerate(records, 1):
        colour = "#c0392b" if r["risk"] == "high" else "#27ae60"
        story.append(Paragraph(
            f'<font color="{colour}"><b>[{i}] {r["label"]} — {r["score"]}%</b></font>',
            body_style,
        ))
        story.append(Paragraph(f'<i>{r["input_text"][:300]}</i>', body_style))
        if r["markers"]:
            story.append(Paragraph(
                f'Sarcasm markers: {", ".join(r["markers"])}', body_style
            ))
        story.append(Spacer(1, 0.3*cm))

    doc.build(story)
    return buffer.getvalue()
