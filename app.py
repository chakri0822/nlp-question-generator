import streamlit as st
import random
from PyPDF2 import PdfReader
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from xml.sax.saxutils import escape

# ---------------- SETUP ---------------- #
st.set_page_config(page_title="NLP Question Generator", layout="wide")

# ----------- LOAD SPACY ----------- #
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

nlp = load_spacy()

# ----------- LOAD NLTK ----------- #
@st.cache_resource
def load_nltk():
    try:
        nltk.data.find('tokenizers/punkt')
    except:
        nltk.download('punkt')

load_nltk()

# ---------------- STYLE ---------------- #
st.markdown("""
<style>
.stButton>button {
    background-color:#f4b942;
    color:black;
    border-radius:10px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- FUNCTIONS ---------------- #
def extract_text(files):
    text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            pdf = PdfReader(file)
            for page in pdf.pages:
                if page.extract_text():
                    text += page.extract_text()
        else:
            text += file.read().decode("utf-8", errors="ignore")
    return text


def extract_topics(text):
    if not text.strip():
        return []
    vectorizer = TfidfVectorizer(stop_words='english', max_features=20)
    X = vectorizer.fit_transform([text])
    return vectorizer.get_feature_names_out()


def clean_text(text):
    return text.replace("•", "").replace("\n", " ")


# ---------------- SESSION STATE ---------------- #
if "text" not in st.session_state:
    st.session_state.text = ""

# ---------------- SIDEBAR ---------------- #
st.sidebar.title("NLP AI System")

files = st.sidebar.file_uploader("Upload Modules", accept_multiple_files=True)

if files:
    st.session_state.text = extract_text(files)
    st.sidebar.success("Files Loaded!")

page = st.sidebar.radio("Navigate", [
    "Generator",
    "Topics",
    "Gap Analyzer",
    "Improver",
    "Type Detector",
    "Ranker",
    "Insights"
])

# ---------------- GENERATOR ---------------- #
if page == "Generator":
    st.title("Question Generator")

    num_q = st.slider("Number of Questions", 5, 20, 5)
    marks = st.selectbox("Marks per Question", [2, 5, 10])
    q_type = st.selectbox("Question Type", ["Descriptive", "MCQ", "Mixed"])

    if st.button("Generate Question Paper"):

        text = st.session_state.text

        if not text.strip():
            st.error("Upload valid files first!")
            st.stop()

        sentences = sent_tokenize(text)

        if len(sentences) == 0:
            st.error("Not enough content!")
            st.stop()

        selected = random.sample(sentences, min(num_q, len(sentences)))

        questions = []

        for i, s in enumerate(selected):
            s = clean_text(s)

            if q_type == "Descriptive":
                if marks == 2:
                    q = f"Q{i+1}. Define: {s} ({marks} Marks)"
                elif marks == 5:
                    q = f"Q{i+1}. Explain briefly: {s} ({marks} Marks)"
                else:
                    q = f"Q{i+1}. Analyze in detail: {s} ({marks} Marks)"

            elif q_type == "MCQ":
                words = s.split()
                correct = words[0] if len(words) > 2 else "NLP"

                options = list(set(words[:4]))
                while len(options) < 4:
                    options.append(f"Option{len(options)}")

                random.shuffle(options)

                q = f"Q{i+1}. What is related to: {s}\n"
                for idx, opt in enumerate(options):
                    q += f"{chr(65+idx)}) {opt}\n"
                q += f"Answer: {correct}"

            else:
                if i % 2 == 0:
                    q = f"Q{i+1}. Explain: {s} ({marks} Marks)"
                else:
                    words = s.split()
                    correct = words[0] if words else "NLP"

                    options = list(set(words[:4]))
                    while len(options) < 4:
                        options.append(f"Option{len(options)}")

                    random.shuffle(options)

                    q = f"Q{i+1}. What is related to: {s}\n"
                    for idx, opt in enumerate(options):
                        q += f"{chr(65+idx)}) {opt}\n"
                    q += f"Answer: {correct}"

            questions.append(q)

        st.subheader("Generated Questions")
        for q in questions:
            st.write(q)

        # -------- PDF -------- #
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer)
        styles = getSampleStyleSheet()

        content = []
        content.append(Paragraph("NLP Question Paper", styles['Title']))
        content.append(Spacer(1, 20))

        for q in questions:
            safe_q = escape(q)
            content.append(Paragraph(safe_q, styles['Normal']))
            content.append(Spacer(1, 10))

        doc.build(content)
        buffer.seek(0)

        st.download_button("Download PDF", buffer, "question_paper.pdf")

# ---------------- OTHER PAGES ---------------- #
elif page == "Topics":
    st.title("Topic Extractor")
    if st.button("Extract Topics"):
        st.write(extract_topics(st.session_state.text))

elif page == "Gap Analyzer":
    st.title("Gap Analyzer")
    syllabus = st.text_area("Paste syllabus")
    if st.button("Analyze"):
        text = st.session_state.text
        missing = [w for w in syllabus.split() if w.lower() not in text.lower()]
        st.write("Missing Topics:", missing[:20])

elif page == "Improver":
    st.title("Question Improver")
    q = st.text_area("Enter question")
    if st.button("Improve"):
        st.write("Improved:", "Explain in detail: " + q)

elif page == "Type Detector":
    st.title("Type Detector")
    q = st.text_area("Enter question")
    if st.button("Detect"):
        if "define" in q.lower():
            st.write("Definition")
        elif "analyze" in q.lower():
            st.write("Analytical")
        else:
            st.write("General")

elif page == "Ranker":
    st.title("Ranker")
    papers = st.text_area("Paste papers")
    if st.button("Rank"):
        st.write(sorted(papers.split("\n"), key=len, reverse=True))

elif page == "Insights":
    st.title("Insights")
    words = st.session_state.text.split()
    if st.button("Show Insights"):
        st.write("Total Words:", len(words))
        st.write("Unique Words:", len(set(words)))
