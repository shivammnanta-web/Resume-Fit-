"""Script to generate 13 varied PDF test fixtures for Phase 1 golden & unit testing.

Variants covered:
1. single_col_1.pdf — Standard 1-column software engineer resume
2. single_col_2.pdf — Standard 1-column data scientist resume
3. single_col_3.pdf — Standard 1-column ML engineer resume
4. two_col_1.pdf — 2-column layout (sidebar + main)
5. two_col_2.pdf — 2-column layout variant
6. table_layout_1.pdf — Table-structured education & skills sections
7. table_layout_2.pdf — Experience & project matrices in tables
8. scanned_image_only.pdf — Raster image embedded without extractable text
9. encrypted_pw.pdf — Password-protected PDF (AES encrypted)
10. empty_blank.pdf — Zero-text blank PDF
11. very_short.pdf — 1-section minimal resume
12. non_english_heavy.pdf — Non-English (Hindi/Russian/Devanagari) resume
13. oversized_pages.pdf — 12-page document (exceeds 10-page limit)
"""

from __future__ import annotations

import io
from pathlib import Path

import pypdf
from PIL import Image, ImageDraw
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

FIXTURES_DIR = Path(__file__).resolve().parent / "pdfs"


def ensure_fixtures_dir() -> Path:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIXTURES_DIR


def make_single_column_1(out_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Aarav Sharma</b>", styles["Title"]))
    story.append(
        Paragraph(
            "Email: aarav.sharma@example.com | Phone: +91 9876543210 | Location: Bengaluru, India",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>PROFESSIONAL SUMMARY</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Backend Engineer with 3 years of experience specializing in distributed Python microservices, PostgreSQL databases, and high-throughput REST APIs.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>WORK EXPERIENCE</b>", styles["Heading2"]))
    story.append(Paragraph("<b>Senior Backend Engineer | CloudScale Tech</b>", styles["Normal"]))
    story.append(Paragraph("Jan 2022 – Present | Bengaluru, India", styles["Italic"]))
    story.append(
        Paragraph(
            "• Architected and deployed microservices handling 20,000 requests/sec using Flask, PostgreSQL, and Redis.",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            "• Implemented automated CI/CD pipelines with Docker and GitHub Actions, cutting release cycles by 40%.",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            "• Optimized SQL queries and added pgvector indexing for semantic document similarity search.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 8))

    story.append(Paragraph("<b>Junior Developer | InfoSystems Ltd</b>", styles["Normal"]))
    story.append(Paragraph("Jun 2020 – Dec 2021 | Pune, India", styles["Italic"]))
    story.append(
        Paragraph(
            "• Developed REST APIs with Python and SQLAlchemy for financial ledger services.",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            "• Wrote unit and integration test suites achieving 88% test coverage with pytest.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>EDUCATION</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "<b>Bachelor of Technology in Computer Science</b> | Graphic Era University (2016 – 2020)",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>TECHNICAL SKILLS</b>", styles["Heading2"]))
    story.append(Paragraph("Languages: Python, SQL, C++, Bash", styles["Normal"]))
    story.append(
        Paragraph("Frameworks: Flask, SQLAlchemy, FastAPI, scikit-learn", styles["Normal"])
    )
    story.append(Paragraph("Databases: PostgreSQL, pgvector, Redis", styles["Normal"]))
    story.append(Paragraph("DevOps: Docker, Linux, Git, GitHub Actions", styles["Normal"]))

    doc.build(story)


def make_single_column_2(out_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Priya Patel</b>", styles["Title"]))
    story.append(
        Paragraph("priya.patel@datasci.org | +91 9123456789 | Mumbai, India", styles["Normal"])
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>CAREER OBJECTIVE</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Data Scientist with 2.5 years of experience building machine learning pipelines, predictive modeling, and NLP applications using Python and scikit-learn.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>PROFESSIONAL EXPERIENCE</b>", styles["Heading2"]))
    story.append(Paragraph("<b>Data Scientist | DataInsights Corp</b>", styles["Normal"]))
    story.append(Paragraph("Jan 2023 – Present | Mumbai, India", styles["Italic"]))
    story.append(
        Paragraph(
            "• Trained and evaluated scikit-learn random forests and gradient boosted trees for customer churn prediction.",
            styles["Normal"],
        )
    )
    story.append(
        Paragraph(
            "• Extracted semantic embeddings using Sentence-Transformers and stored them in PostgreSQL with pgvector.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>PROJECTS</b>", styles["Heading2"]))
    story.append(Paragraph("<b>Semantic Resume Search Engine</b>", styles["Normal"]))
    story.append(
        Paragraph(
            "• Built hybrid TF-IDF lexical and dense vector retrieval with all-MiniLM-L6-v2 embeddings.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>EDUCATION</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Bachelor of Computer Applications (AI & Data Science) | Graphic Era University (2019 – 2022)",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>CORE COMPETENCIES</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Python, Pandas, NumPy, scikit-learn, Sentence-Transformers, SQL, PostgreSQL, Git",
            styles["Normal"],
        )
    )

    doc.build(story)


def make_single_column_3(out_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Rohan Verma</b>", styles["Title"]))
    story.append(
        Paragraph("rohan.verma@mlops.net | +91 9988776655 | Dehradun, India", styles["Normal"])
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>SUMMARY</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "Machine Learning Engineer with 4 years experience in MLOps, model deployment, and deep learning.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>EXPERIENCE</b>", styles["Heading2"]))
    story.append(Paragraph("<b>MLOps Engineer | DeepScale AI</b>", styles["Normal"]))
    story.append(Paragraph("03/2020 – 08/2023 | Dehradun, India", styles["Italic"]))
    story.append(
        Paragraph(
            "• Deployed PyTorch and ONNX models into production with Docker and Gunicorn.",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>ACADEMIC BACKGROUND</b>", styles["Heading2"]))
    story.append(
        Paragraph(
            "B.Tech in Computer Science | Dehradun Institute of Technology (2016 – 2020)",
            styles["Normal"],
        )
    )
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>SKILLS</b>", styles["Heading2"]))
    story.append(Paragraph("Python, PyTorch, Docker, Linux, Git, Flask, CI/CD", styles["Normal"]))

    doc.build(story)


def make_two_column_1(out_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Neha Gupta</b> — Full Stack Developer", styles["Title"]))
    story.append(Paragraph("neha.gupta@dev.io | +91 9811223344 | Delhi, India", styles["Normal"]))
    story.append(Spacer(1, 14))

    col1 = [
        Paragraph("<b>EDUCATION</b>", styles["Heading3"]),
        Paragraph(
            "BCA (AI & Data Science)<br/>Graphic Era University<br/>2019 – 2022", styles["Normal"]
        ),
        Spacer(1, 10),
        Paragraph("<b>SKILLS</b>", styles["Heading3"]),
        Paragraph(
            "• Python<br/>• Flask<br/>• JavaScript<br/>• SQL<br/>• PostgreSQL<br/>• Docker<br/>• Git",
            styles["Normal"],
        ),
    ]

    col2 = [
        Paragraph("<b>PROFESSIONAL EXPERIENCE</b>", styles["Heading3"]),
        Paragraph(
            "<b>Full Stack Engineer | WebWorks</b><br/><i>Jan 2022 – Present</i>", styles["Normal"]
        ),
        Paragraph(
            "• Built dynamic web portals using Flask, Jinja2, and HTMX with responsive UI.<br/>• Integrated PostgreSQL 15 database with SQLAlchemy 2.0 ORM.<br/>• Authored RESTful APIs with Pydantic validation.",
            styles["Normal"],
        ),
        Spacer(1, 10),
        Paragraph("<b>PROJECTS</b>", styles["Heading3"]),
        Paragraph(
            "<b>Resume-Job Matching Engine</b><br/>• Implemented TF-IDF and dense embeddings with hybrid score fusion.",
            styles["Normal"],
        ),
    ]

    table = Table([[col1, col2]], colWidths=[180, 360])
    table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("RIGHTPADDING", (0, 0), (0, 0), 15),
                ("LEFTPADDING", (1, 0), (1, 0), 15),
            ]
        )
    )

    story.append(table)
    doc.build(story)


def make_two_column_2(out_path: Path) -> None:
    make_two_column_1(out_path)


def make_table_layout_1(out_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(out_path), pagesize=letter, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36
    )
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>Vikram Malhotra</b>", styles["Title"]))
    story.append(Paragraph("vikram@enterprise.org | +91 9777888999", styles["Normal"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>WORK EXPERIENCE</b>", styles["Heading2"]))
    exp_data = [
        ["Role", "Company", "Dates", "Responsibilities"],
        [
            "Software Developer",
            "FinTech Global",
            "Jan 2021 – Present",
            "Built payment reconciliation APIs in Python and PostgreSQL. Reduced latency by 25%.",
        ],
        [
            "Associate Engineer",
            "CodeMatrix",
            "Jun 2019 – Dec 2020",
            "Maintained backend services and wrote automated tests in pytest.",
        ],
    ]
    t = Table(exp_data, colWidths=[100, 100, 110, 230])
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0D7377")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )
    story.append(t)
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>EDUCATION</b>", styles["Heading2"]))
    edu_data = [
        ["Degree", "Institution", "Year", "Score"],
        ["B.Tech CS", "Graphic Era University", "2015 – 2019", "8.6 CGPA"],
    ]
    t2 = Table(edu_data, colWidths=[120, 200, 100, 80])
    t2.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ]
        )
    )
    story.append(t2)
    story.append(Spacer(1, 14))

    story.append(Paragraph("<b>TECHNICAL SKILLS</b>", styles["Heading2"]))
    story.append(Paragraph("Python, SQL, PostgreSQL, Flask, Git, Docker, Linux", styles["Normal"]))

    doc.build(story)


def make_table_layout_2(out_path: Path) -> None:
    make_table_layout_1(out_path)


def make_scanned_image_only(out_path: Path) -> None:
    """Generate an image-only PDF containing no extractable text stream."""
    img = Image.new("RGB", (612, 792), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw simple rasterized shapes resembling scanned document text
    draw.rectangle([50, 50, 300, 70], fill=(50, 50, 50))
    for y in range(100, 600, 25):
        draw.line([(50, y), (550, y)], fill=(120, 120, 120), width=4)

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="JPEG")
    img_buffer.seek(0)

    c = canvas.Canvas(str(out_path), pagesize=letter)
    # Save the canvas with only the raster image
    from reportlab.lib.utils import ImageReader

    c.drawImage(ImageReader(img_buffer), 0, 0, width=612, height=792)
    c.save()


def make_encrypted(out_path: Path) -> None:
    """Generate an AES-encrypted PDF."""
    temp_buf = io.BytesIO()
    c = canvas.Canvas(temp_buf, pagesize=letter)
    c.drawString(100, 700, "Secret Resume Content for John Doe")
    c.drawString(100, 680, "Experience: 5 years software engineering")
    c.save()
    temp_buf.seek(0)

    reader = pypdf.PdfReader(temp_buf)
    writer = pypdf.PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    writer.encrypt("resumefit")
    with open(out_path, "wb") as f:
        writer.write(f)


def make_empty(out_path: Path) -> None:
    c = canvas.Canvas(str(out_path), pagesize=letter)
    c.showPage()
    c.save()


def make_very_short(out_path: Path) -> None:
    c = canvas.Canvas(str(out_path), pagesize=letter)
    c.drawString(100, 700, "Dev Candidate - Just Learning Python")
    c.save()


def _register_unicode_font() -> str:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    candidates = [
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"),
        Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/Library/Fonts/Arial.ttf"),
    ]
    for p in candidates:
        if p.exists():
            try:
                fname = "DocUnicode"
                if fname not in pdfmetrics.getRegisteredFontNames():
                    pdfmetrics.registerFont(TTFont(fname, str(p)))
                return fname
            except Exception:
                pass
    return "Helvetica"


def make_non_english(out_path: Path) -> None:
    font_name = _register_unicode_font()
    c = canvas.Canvas(str(out_path), pagesize=letter)
    c.setFont(font_name, 11)
    c.drawString(100, 700, "Резюме: Иван Иванов - Инженер-программист")
    c.drawString(100, 680, "Опыт работы: Разработка программного обеспечения на Python и C++.")
    c.drawString(100, 660, "Навыки: Машинное обучение, Базы данных SQL, Архитектура.")
    c.save()


def make_oversized_pages(out_path: Path) -> None:
    c = canvas.Canvas(str(out_path), pagesize=letter)
    for p in range(1, 13):
        c.drawString(100, 700, f"Page {p} of 12: Candidate Curriculum Vitae")
        c.drawString(
            100, 650, "Extensive publications and career details stretching across 12 pages."
        )
        c.showPage()
    c.save()


def generate_all_fixtures() -> dict[str, Path]:
    fdir = ensure_fixtures_dir()
    generators = {
        "single_col_1.pdf": make_single_column_1,
        "single_col_2.pdf": make_single_column_2,
        "single_col_3.pdf": make_single_column_3,
        "two_col_1.pdf": make_two_column_1,
        "two_col_2.pdf": make_two_column_2,
        "table_layout_1.pdf": make_table_layout_1,
        "table_layout_2.pdf": make_table_layout_2,
        "scanned_image_only.pdf": make_scanned_image_only,
        "encrypted_pw.pdf": make_encrypted,
        "empty_blank.pdf": make_empty,
        "very_short.pdf": make_very_short,
        "non_english_heavy.pdf": make_non_english,
        "oversized_pages.pdf": make_oversized_pages,
    }

    created = {}
    for fname, gen in generators.items():
        fpath = fdir / fname
        gen(fpath)
        created[fname] = fpath

    return created


if __name__ == "__main__":
    files = generate_all_fixtures()
    print(f"Generated {len(files)} test PDF fixtures in {FIXTURES_DIR}")
