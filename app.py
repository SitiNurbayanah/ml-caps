import os
import re
import fitz  # PyMuPDF
import pickle
import numpy as np
import psycopg2
from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Setup
nltk_data_dir = os.path.expanduser('~/nltk_data')
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Download NLTK datasets if not already
def download_nltk_data():
    for item in ['stopwords', 'punkt', 'wordnet', 'omw-1.4']:
        try:
            nltk.data.find(f'tokenizers/{item}' if item == 'punkt' else f'corpora/{item}')
        except LookupError:
            nltk.download(item, download_dir=nltk_data_dir)

download_nltk_data()

app = Flask(__name__)

# Database config
DB_CONFIG = {
    "host": "w5ejy.h.filess.io",
    "port": 5434,
    "dbname": "CapstoneML_askrequire",
    "user": "CapstoneML_askrequire",
    "password": "4212b39e981348968d21a833646f0f70cba2bbf6",
    "schema": "job_portal"
}

# Globals
model = None
tokenizer = None
stop_words = set()
lemmatizer = None
max_len = 200
job_dataset = []

def initialize_models():
    global model, tokenizer, stop_words, lemmatizer
    try:
        import tensorflow as tf
        print("Loading model...")
        model = tf.keras.models.load_model("jobmatch_model.h5")
        print("Model loaded.")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"[ERROR] Model load failed: {e}")
        return False

    try:
        stop_words.update(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
    except Exception as e:
        print(f"[WARN] NLTK init failed, fallback used: {e}")
        stop_words.update(['the', 'a', 'and', 'in', 'on', 'for'])
        lemmatizer = None
    return True

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-z\s]', ' ', text.lower())
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
        text = ''.join([page.get_text() for page in doc])
        doc.close()
        return text
    except Exception as e:
        raise Exception(f"PDF extraction failed: {e}")

def get_jobs_from_db():
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            dbname=DB_CONFIG["dbname"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        cur = conn.cursor()
        query = f"""
        SELECT id_lowongan, posisi, tentang, syarat, skill
        FROM {DB_CONFIG['schema']}.lowongan
        WHERE tentang IS NOT NULL
        LIMIT 500
        """
        cur.execute(query)
        jobs = []
        for row in cur.fetchall():
            job_id, title, tentang, syarat, skill = row
            combined_text = " ".join(str(x) for x in [tentang, syarat, skill] if x)
            jobs.append({
                "id_lowongan": job_id,
                "Job Title": title,
                "clean_text": preprocess_text(combined_text)
            })
        cur.close()
        conn.close()
        return jobs
    except Exception as e:
        print(f"[ERROR] DB fetch failed: {e}")
        return []

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "jobs_loaded": len(job_dataset)
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model or tokenizer not loaded"}), 500
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename.endswith(".pdf"):
        return jsonify({"error": "Only PDF allowed"}), 400

    try:
        pdf_text = extract_text_from_pdf(file.read())
        clean_resume = preprocess_text(pdf_text)

        if not job_dataset:
            return jsonify({"error": "No job data available"}), 500

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        resume_seq = tokenizer.texts_to_sequences([clean_resume])
        resume_pad = pad_sequences(resume_seq, maxlen=max_len, padding='post')

        job_texts = [j["clean_text"] for j in job_dataset]
        job_seq = tokenizer.texts_to_sequences(job_texts)
        job_pad = pad_sequences(job_seq, maxlen=max_len, padding='post')

        resume_pad = np.repeat(resume_pad, len(job_dataset), axis=0)
        preds = model.predict([resume_pad, job_pad], verbose=0).flatten()

        results = sorted([
            {
                "id_lowongan": j["id_lowongan"],
                "Job Title": j["Job Title"],
                "match_score": float(score)
            } for j, score in zip(job_dataset, preds)
        ], key=lambda x: x["match_score"], reverse=True)

        return jsonify({"top_matches": results[:20]})

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

# Run init
print("Initializing...")
if initialize_models():
    job_dataset = get_jobs_from_db()
    print(f"Loaded {len(job_dataset)} jobs.")
else:
    print("Initialization failed.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
