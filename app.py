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
import os

# Create NLTK data directory if it doesn't exist
nltk_data_dir = os.path.expanduser('~/nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

def download_nltk_data():
    datasets = ['stopwords', 'punkt', 'wordnet', 'omw-1.4']
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
            print(f"NLTK {dataset} already exists")
        except LookupError:
            try:
                print(f"Downloading NLTK {dataset}...")
                nltk.download(dataset, download_dir=nltk_data_dir, quiet=True)
                print(f"Successfully downloaded NLTK {dataset}")
            except Exception as e:
                print(f"Failed to download NLTK {dataset}: {e}")

# Download NLTK data
download_nltk_data()

app = Flask(__name__)

# DB config
DB_HOST = "w5ejy.h.filess.io"
DB_PORT = 5434
DB_NAME = "CapstoneML_askrequire"
DB_USER = "CapstoneML_askrequire"
DB_PASSWORD = "4212b39e981348968d21a833646f0f70cba2bbf6"
DB_SCHEMA = "job_portal"

# Variables
model = None
tokenizer = None
max_len = 200
stop_words = set()
lemmatizer = None
job_dataset = []

def initialize_models():
    global model, tokenizer, stop_words, lemmatizer
    try:
        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model("jobmatch_model.h5")
        print("Model loaded successfully")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        return False
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        print("NLTK tools initialized successfully")
    except Exception as e:
        print(f"Error initializing NLTK tools: {e}")
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        lemmatizer = None
        print("Using fallback preprocessing without NLTK")
    return True

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\n|\r|\t', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    try:
        tokens = word_tokenize(text)
    except:
        tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    if lemmatizer:
        try:
            tokens = [lemmatizer.lemmatize(w) for w in tokens]
        except:
            pass
    return " ".join(tokens)

def extract_text_from_pdf(file_stream):
    try:
        doc = fitz.open(stream=file_stream, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"PDF extraction failed: {str(e)}")

def get_jobs_from_db():
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
            job_id = row[0]
            job_title = row[1]
            combined_text_raw = " ".join(str(col) for col in row[2:] if col)
            clean_text = preprocess_text(combined_text_raw)
            jobs.append({"id_lowongan": job_id, "Job Title": job_title, "clean_text": clean_text})
        print(f"Successfully loaded {len(jobs)} jobs from database")
        return jobs
    except Exception as e:
        print("DB connection or query error:", e)
        return []

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Models not loaded properly"}), 500
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
    try:
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
                "id_lowongan": job["id_lowongan"],
                "Job Title": job["Job Title"],
                "match_score": float(score)
            })
        results_sorted = sorted(results, key=lambda x: x["match_score"], reverse=True)
        return jsonify({
            "top_matches": results_sorted[:20]
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

print("Initializing application...")
if initialize_models():
    print("Models initialized successfully")
    job_dataset = get_jobs_from_db()
    print("Application ready!")
else:
    print("Failed to initialize models")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
