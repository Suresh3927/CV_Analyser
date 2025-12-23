import streamlit as st
import pickle
import re
import nltk
import pdfplumber
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from fpdf import FPDF
import nltk
nltk.download('averaged_perceptron_tagger')


# ✅ Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ✅ Load trained model, vectorizer, and label encoder
try:
    model = pickle.load(open("resume_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    label_encoder = pickle.load(open("label_encoder.pkl", "rb"))
except Exception as e:
    st.error(f"⚠ Error loading model files: {e}")
    st.stop()

# ✅ Initialize NLP tools
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ✅ Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = '\n'.join(page.extract_text() for page in pdf.pages if page.extract_text())
        return text if text.strip() else None
    except Exception as e:
        st.error(f"⚠ PDF Processing Error: {e}")
        return None

# ✅ Improved Preprocessing Function
def preprocess_text(text):
    text = re.sub(r'[^\w\s#.+-]', ' ', text).lower().strip()
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

# ✅ Extract skills from resume
def extract_skills(text):
    skills_list = ['python', 'machine learning', 'flask', 'django', 'sql', 'data science', 'java', 'aws', 'react']
    skills_found = [skill for skill in skills_list if skill in text.lower()]
    return skills_found

# ✅ Streamlit UI - Custom Styling
st.markdown("""
    <style>
        .main { background-color: #f5f5f5; }
        h1 { color: #2E86C1; text-align: center; }
    </style>
""", unsafe_allow_html=True)

st.title("🔍 Resume Job Role Predictor")
st.write("Upload your resume (PDF or TXT), and the model will predict the most suitable job category.")

# ✅ File Upload
uploaded_file = st.file_uploader("Upload Resume", type=["txt", "pdf"])

if uploaded_file is not None:
    resume_text = None

    try:
        if uploaded_file.type == "application/pdf":
            resume_text = extract_text_from_pdf(uploaded_file)
        else:
            resume_text = uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"⚠ Error reading file: {e}")

    if not resume_text:
        st.error("⚠ Could not extract text. Try another file.")
    else:
        # ✅ Preprocess and predict
        processed_text = preprocess_text(resume_text)
        vectorized_text = vectorizer.transform([processed_text]).toarray()
        skills = extract_skills(resume_text)

        try:
            prediction = model.predict(vectorized_text)
            category = label_encoder.inverse_transform(prediction)[0]

            st.write("🏆 **Predicted Job Category:**", f"*{category}*")
            st.write("💡 **Extracted Skills:**", ", ".join(skills) if skills else "No specific skills detected")

            # ✅ **Plot Class Distribution**
            category_counts = np.bincount(label_encoder.transform(label_encoder.classes_))
            plt.figure(figsize=(10, 5))
            sns.barplot(x=label_encoder.classes_, y=category_counts)
            plt.xticks(rotation=90)
            plt.title("Class Distribution in Training Data")
            plt.ylabel("Count")
            st.pyplot(plt)

            # ✅ **Download Report**
            def generate_pdf():
                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Resume Analysis Report", ln=True, align='C')
                pdf.ln(10)
                pdf.cell(200, 10, txt=f"Predicted Job Category: {category}", ln=True)
                pdf.ln(5)
                pdf.cell(200, 10, txt=f"Extracted Skills: {', '.join(skills) if skills else 'None'}", ln=True)
                pdf_file_path = "resume_report.pdf"
                pdf.output(pdf_file_path)
                return pdf_file_path

            if st.button("📥 Download Report"):
                pdf_path = generate_pdf()
                with open(pdf_path, "rb") as file:
                    st.download_button(label="Download Resume Report 📄", data=file, file_name="resume_report.pdf", mime="application/pdf")
        except Exception as e:
            st.error(f"⚠ Prediction Error: {e}")