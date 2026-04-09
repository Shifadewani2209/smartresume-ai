import os
import json
from io import BytesIO

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from PyPDF2 import PdfReader
from docx import Document
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import simpleSplit


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

client = OpenAI(api_key=api_key)


# -----------------------------
# Streamlit page config
# -----------------------------
st.set_page_config(page_title="SmartResume AI", page_icon="📄", layout="wide")

st.title("SmartResume AI")
st.caption("Upload Resume + Job Description and get AI-powered match analysis with PDF report download")


# -----------------------------
# File text extraction helpers
# -----------------------------
def extract_text_from_pdf(uploaded_file):
    try:
        uploaded_file.seek(0)
        reader = PdfReader(uploaded_file)
        text = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
        return "\n".join(text).strip()
    except Exception as e:
        return f"ERROR_READING_PDF: {e}"


def extract_text_from_docx(uploaded_file):
    try:
        uploaded_file.seek(0)
        doc = Document(uploaded_file)
        text = [para.text for para in doc.paragraphs if para.text.strip()]
        return "\n".join(text).strip()
    except Exception as e:
        return f"ERROR_READING_DOCX: {e}"


def extract_text_from_txt(uploaded_file):
    try:
        uploaded_file.seek(0)
        return uploaded_file.read().decode("utf-8").strip()
    except UnicodeDecodeError:
        try:
            uploaded_file.seek(0)
            return uploaded_file.read().decode("latin-1").strip()
        except Exception as e:
            return f"ERROR_READING_TXT: {e}"


def extract_text(uploaded_file):
    if uploaded_file is None:
        return ""

    file_name = uploaded_file.name.lower()

    if file_name.endswith(".pdf"):
        return extract_text_from_pdf(uploaded_file)
    elif file_name.endswith(".docx"):
        return extract_text_from_docx(uploaded_file)
    elif file_name.endswith(".txt"):
        return extract_text_from_txt(uploaded_file)
    else:
        return "UNSUPPORTED_FILE_TYPE"


# -----------------------------
# PDF Report Generator
# -----------------------------
def create_pdf_report(result):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    width, height = A4
    x_margin = 50
    y = height - 50
    line_height = 16
    max_width = width - 2 * x_margin

    def write_line(text="", font="Helvetica", size=11, spacing=line_height):
        nonlocal y
        if y < 60:
            pdf.showPage()
            y = height - 50
        pdf.setFont(font, size)
        pdf.drawString(x_margin, y, str(text))
        y -= spacing

    def write_wrapped_text(text, font="Helvetica", size=11, spacing=line_height):
        nonlocal y
        pdf.setFont(font, size)
        lines = simpleSplit(str(text), font, size, max_width)
        for line in lines:
            if y < 60:
                pdf.showPage()
                y = height - 50
                pdf.setFont(font, size)
            pdf.drawString(x_margin, y, line)
            y -= spacing

    write_line("Resume Match Analysis Report", font="Helvetica-Bold", size=16, spacing=24)
    write_line(f"Match Score: {result['match_score']}/100", font="Helvetica-Bold", size=12, spacing=20)

    write_line("Overall Verdict", font="Helvetica-Bold", size=13, spacing=18)
    write_wrapped_text(result["overall_verdict"])
    y -= 8

    write_line("Strengths", font="Helvetica-Bold", size=13, spacing=18)
    for item in result["strengths"]:
        write_wrapped_text(f"- {item}")
    y -= 8

    write_line("Missing Keywords", font="Helvetica-Bold", size=13, spacing=18)
    for item in result["missing_keywords"]:
        write_wrapped_text(f"- {item}")
    y -= 8

    write_line("Improved Professional Summary", font="Helvetica-Bold", size=13, spacing=18)
    write_wrapped_text(result["improved_summary"])
    y -= 8

    write_line("Improved Resume Bullets", font="Helvetica-Bold", size=13, spacing=18)
    for item in result["improved_bullets"]:
        write_wrapped_text(f"- {item}")
    y -= 8

    write_line("Likely Interview Questions", font="Helvetica-Bold", size=13, spacing=18)
    for i, item in enumerate(result["interview_questions"], start=1):
        write_wrapped_text(f"{i}. {item}")

    pdf.save()
    buffer.seek(0)
    return buffer


# -----------------------------
# AI Analysis
# -----------------------------
def analyze_resume(resume_text, job_description_text):
    prompt = f"""
You are an expert ATS evaluator, recruiter assistant, and resume optimization AI.

Compare the resume against the job description and return ONLY valid JSON.

Use this exact structure:
{{
  "match_score": 0,
  "overall_verdict": "",
  "strengths": ["", "", "", "", ""],
  "missing_keywords": ["", "", "", "", ""],
  "improved_summary": "",
  "improved_bullets": ["", "", ""],
  "interview_questions": ["", "", "", "", ""]
}}

Rules:
- match_score must be an integer from 0 to 100
- overall_verdict must be 1 short paragraph
- strengths must contain exactly 5 points
- missing_keywords must contain 5 to 10 important keywords or skills
- improved_summary must be 3 to 4 lines and ATS-friendly
- improved_bullets must contain exactly 3 strong bullet points
- interview_questions must contain exactly 5 likely questions
- Keep output concise and realistic
- Return JSON only
- No markdown
- No explanation outside JSON

Resume:
{resume_text}

Job Description:
{job_description_text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a precise hiring assistant. Return clean JSON only."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2
    )

    content = response.choices[0].message.content.strip()

    start = content.find("{")
    end = content.rfind("}") + 1

    if start == -1 or end == 0:
        raise ValueError("Could not parse JSON from AI response.")

    json_text = content[start:end]
    return json.loads(json_text)


# -----------------------------
# UI Layout
# -----------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Resume Input")
    resume_file = st.file_uploader(
        "Upload Resume File",
        type=["pdf", "docx", "txt"],
        key="resume_uploader"
    )
    resume_text_manual = st.text_area(
        "Or paste resume text",
        height=220,
        placeholder="Paste resume text here if you do not want to upload a file..."
    )

with right_col:
    st.subheader("Job Description Input")
    jd_file = st.file_uploader(
        "Upload Job Description File",
        type=["pdf", "docx", "txt"],
        key="jd_uploader"
    )
    jd_text_manual = st.text_area(
        "Or paste job description text",
        height=220,
        placeholder="Paste job description here if you do not want to upload a file..."
    )


# -----------------------------
# Extract text
# -----------------------------
resume_text = ""
jd_text = ""

if resume_file is not None:
    resume_text = extract_text(resume_file)
elif resume_text_manual.strip():
    resume_text = resume_text_manual.strip()

if jd_file is not None:
    jd_text = extract_text(jd_file)
elif jd_text_manual.strip():
    jd_text = jd_text_manual.strip()


# -----------------------------
# Preview extracted text
# -----------------------------
with st.expander("Preview Resume Text"):
    if resume_text:
        st.text_area("Resume Preview", resume_text[:5000], height=200)
    else:
        st.write("No resume content loaded yet.")

with st.expander("Preview Job Description Text"):
    if jd_text:
        st.text_area("Job Description Preview", jd_text[:5000], height=200)
    else:
        st.write("No job description content loaded yet.")


# -----------------------------
# Validation messages
# -----------------------------
if resume_text == "UNSUPPORTED_FILE_TYPE" or jd_text == "UNSUPPORTED_FILE_TYPE":
    st.error("Unsupported file type detected. Please upload only PDF, DOCX, or TXT files.")

if "ERROR_READING_" in resume_text:
    st.error(f"Resume file could not be read: {resume_text}")

if "ERROR_READING_" in jd_text:
    st.error(f"Job description file could not be read: {jd_text}")


# -----------------------------
# Analyze button
# -----------------------------
if st.button("Analyze Resume Match"):
    if not resume_text or resume_text == "UNSUPPORTED_FILE_TYPE" or "ERROR_READING_" in resume_text:
        st.warning("Please upload or paste a valid resume.")
    elif not jd_text or jd_text == "UNSUPPORTED_FILE_TYPE" or "ERROR_READING_" in jd_text:
        st.warning("Please upload or paste a valid job description.")
    else:
        with st.spinner("Analyzing documents with Gen AI..."):
            try:
                result = analyze_resume(resume_text, jd_text)

                st.success("Analysis completed successfully.")
                st.markdown("---")

                st.metric("Match Score", f"{result['match_score']}/100")

                st.subheader("Overall Verdict")
                st.write(result["overall_verdict"])

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Strengths")
                    for item in result["strengths"]:
                        st.write(f"- {item}")

                with col2:
                    st.subheader("Missing Keywords")
                    for item in result["missing_keywords"]:
                        st.write(f"- {item}")

                st.subheader("Improved Professional Summary")
                st.info(result["improved_summary"])

                st.subheader("Improved Resume Bullets")
                for item in result["improved_bullets"]:
                    st.write(f"- {item}")

                st.subheader("Likely Interview Questions")
                for i, item in enumerate(result["interview_questions"], start=1):
                    st.write(f"{i}. {item}")

                final_output = f"""
Match Score: {result['match_score']}/100

Overall Verdict:
{result['overall_verdict']}

Strengths:
""" + "\n".join([f"- {x}" for x in result["strengths"]]) + """

Missing Keywords:
""" + "\n".join([f"- {x}" for x in result["missing_keywords"]]) + f"""

Improved Professional Summary:
{result['improved_summary']}

Improved Resume Bullets:
""" + "\n".join([f"- {x}" for x in result["improved_bullets"]]) + """

Interview Questions:
""" + "\n".join([f"- {x}" for x in result["interview_questions"]])

                st.subheader("Copy-Friendly Report")
                st.text_area("Analysis Report", final_output, height=300)

                pdf_buffer = create_pdf_report(result)

                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_buffer,
                    file_name="smartresume_ai_report.pdf",
                    mime="application/pdf"
                )

            except json.JSONDecodeError:
                st.error("The AI response was not valid JSON. Please try again.")
            except Exception as e:
                st.error(f"Something went wrong: {e}")