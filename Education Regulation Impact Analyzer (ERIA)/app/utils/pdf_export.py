from fpdf import FPDF
import tempfile

class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 16)
        self.cell(0, 10, "ERIA Analysis Report", 0, 1, "C")
        self.ln(5)

def add_section(pdf, title):
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, title, 0, 1)
    pdf.ln(2)

def add_subsection(pdf, subtitle):
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, subtitle, 0, 1)

def clean_text(text):
    return (
        text.replace("•", "-")
            .replace("’", "'")
            .replace("“", '"')
            .replace("”", '"')
            .encode("latin-1", "ignore")
            .decode("latin-1")
    )

def add_bullets(pdf, items):
    pdf.set_font("Arial", "", 11)
    for item in items:
        pdf.multi_cell(0, 6, f"- {clean_text(item)}")
    pdf.ln(2)

def generate_pdf(data):

    raw = data.get("raw", data)

    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()

    # -------------------------
    # SUMMARY
    # -------------------------
    summary = raw.get("summary", {})

    add_section(pdf, "Summary")

    for key, value in summary.items():
        add_subsection(pdf, key.replace("_", " ").title())
        add_bullets(pdf, value)

    # -------------------------
    # STAKEHOLDERS
    # -------------------------
    stakeholders = raw.get("stakeholder_report", {})

    add_section(pdf, "Stakeholders")

    for key, value in stakeholders.items():
        add_subsection(pdf, key.title())
        add_bullets(pdf, value)

    # -------------------------
    # RISKS
    # -------------------------
    risks = raw.get("sentiment_risk", {})

    add_section(pdf, "Risks & Sentiment")

    for key, value in risks.items():
        add_subsection(pdf, key.replace("_", " ").title())
        add_bullets(pdf, value)

    # -------------------------
    # IMPACT
    # -------------------------
    impact = raw.get("impact_assessment", {})

    add_section(pdf, "Impact Assessment")

    for key, value in impact.items():
        add_subsection(pdf, key.title())
        add_bullets(pdf, value)

    # -------------------------
    # TIMELINE
    # -------------------------
    chronology = raw.get("chronology", {})

    add_section(pdf, "Timeline")

    for key, value in chronology.items():
        add_subsection(pdf, key.replace("_", " ").title())
        add_bullets(pdf, value)

    # -------------------------
    # SAVE FILE
    # -------------------------
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(temp_file.name)

    return temp_file.name