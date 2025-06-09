import re
import fitz  # PyMuPDF for PDF text extraction
import nltk
import numpy as np
import pickle
import psycopg2  # Use psycopg2-binary for PostgreSQL connection
import tensorflow as tf
from flask import Flask, request, jsonify
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required nltk data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

app = Flask(__name__)

# Database configuration
DB_HOST = "w5ejy.h.filess.io"
DB_PORT = 5434
DB_NAME = "CapstoneML_askrequire"
DB_USER = "CapstoneML_askrequire"
DB_PASSWORD = "4212b39e981348968d21a833646f0f70cba2bbf6"
DB_SCHEMA = "job_portal"

# Load saved TensorFlow model and tokenizer
model = tf.keras.models.load_model("jobmatch_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_len = 200  # Length used for padding sequences

# Initialize nltk preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    """Preprocess the input text for model prediction."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\n|\r|\t', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

def extract_text_from_pdf(file_stream):
    """Extract text from a PDF file."""
    doc = fitz.open(stream=file_stream, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def get_jobs_from_db():
    """Fetch job postings from PostgreSQL database and preprocess texts."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        cur = conn.cursor()
        query = f"""
        SELECT id_lowongan, posisi, tentang, syarat, skill
        FROM {DB_SCHEMA}.lowongan
        WHERE tentang IS NOT NULL
        LIMIT 500
        """
        cur.execute(query)
        rows = cur.fetchall()
        cur.close()
        conn.close()
        
        jobs = []
        for row in rows:
            job_id = row[0]  # Get id_lowongan
            job_title = row[1]
            combined_text_raw = " ".join(str(col) for col in row[2:] if col)
            clean_text = preprocess_text(combined_text_raw)
            jobs.append({"id_lowongan": job_id, "Job Title": job_title, "clean_text": clean_text})
        return jobs
    except Exception as e:
        print("DB connection or query error:", e)
        return []

# Cache job data on startup
job_dataset = get_jobs_from_db()

@app.route('/predict', methods=['POST'])
def predict():
    """Predict job matches based on the uploaded CV."""
    if 'file' not in request.files:
        return jsonify({"error": "Missing file in request"}), 400
    file = request.files['file']
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "File must have .pdf extension"}), 400

    try:
        pdf_text = extract_text_from_pdf(file.read())
    except Exception as e:
        return jsonify({"error": f"Failed to extract text from PDF: {str(e)}"}), 500

    clean_resume = preprocess_text(pdf_text)

    if not job_dataset:
        return jsonify({"error": "Job dataset not loaded or DB error"}), 500

    all_job_texts = [job['clean_text'] for job in job_dataset]
    resume_seq = tokenizer.texts_to_sequences([clean_resume])
    job_seq = tokenizer.texts_to_sequences(all_job_texts)

    from tensorflow.keras.preprocessing.sequence import pad_sequences
    resume_pad = pad_sequences(resume_seq, maxlen=max_len, padding='post')
    job_pad = pad_sequences(job_seq, maxlen=max_len, padding='post')

    resume_pad_tiled = np.repeat(resume_pad, len(job_dataset), axis=0)

    predictions = model.predict([resume_pad_tiled, job_pad], verbose=0).flatten()

    results = []
    for job, score in zip(job_dataset, predictions):
        results.append({
            "id_lowongan": job["id_lowongan"],  # Include id_lowongan
            "Job Title": job["Job Title"],
            "match_score": float(score)
        })

    results_sorted = sorted(results, key=lambda x: x["match_score"], reverse=True)

    return jsonify({
        "top_matches": results_sorted[:20]
    })

if __name__ == '__main__':
    app.run(debug=True)
