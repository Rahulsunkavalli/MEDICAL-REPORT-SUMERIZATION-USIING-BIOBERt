from flask import Flask, render_template, request, jsonify, send_file
from dotenv import load_dotenv
import os
from utils.pdf_utils import extract_text_from_pdf
from utils.summarizer import summarize_medical, is_medical_content, translate_text
import google.generativeai as genai
import re
from gtts import gTTS
import io
from googletrans import Translator
from transformers import BertTokenizer, BertForTokenClassification
from transformers import pipeline
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.tokenize import sent_tokenize
import nltk
# Load environment variables
load_dotenv()
UPLOAD_FOLDER = 'uploads'
SUMMARY_FOLDER = 'summaries'
AUDIO_DIR = 'static/audio'
nltk.download('punkt')
for folder in [UPLOAD_FOLDER, SUMMARY_FOLDER, AUDIO_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

#
# Download NLTK Punkt tokenizer
nltk.download('punkt')

# Load BioBERT model and tokenizer
MODEL_NAME = "dmis-lab/biobert-base-cased-v1.1"
TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
EXTRACTIVE_MODEL = AutoModel.from_pretrained(MODEL_NAME)

def extractive_summary(text):
    """Generate an extractive summary using BioBERT."""
    # Split text into sentences
    sentences = sent_tokenize(text)
    
    # Tokenize sentences for BioBERT
    inputs = TOKENIZER.batch_encode_plus(
        sentences, return_tensors='pt', max_length=512, truncation=True, padding='longest'
    )
    
    # Pass through the BioBERT model
    outputs = EXTRACTIVE_MODEL(**inputs)
    sentence_embeddings = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    
    # Compute cosine similarity between sentence embeddings
    similarity_matrix = cosine_similarity(sentence_embeddings)
    np.fill_diagonal(similarity_matrix, 0)  # Ignore self-similarity
    
    # Rank sentences based on similarity scores
    sentence_scores = np.sum(similarity_matrix, axis=0)
    ranked_sentences = sorted(
        ((sentence_scores[i], s) for i, s in enumerate(sentences)), reverse=True
    )
    
    # Select the top-ranked sentences (e.g., 60% of total)
    num_sentences = int(len(sentences) * 0.6)
    summary = ' '.join([s for _, s in ranked_sentences[:num_sentences]])
    
    return summary if summary.strip() else "No valid summary generated."
# Initialize Google Generative AI
genai.configure(api_key=os.getenv("AIzaSyAJONLVfr744J2u3OEyNW5mMrESRuQdCUg"))
gen_model = genai.GenerativeModel('gemini-pro')

# Directories and configuration
AUDIO_DIR = 'static/audio'
if not os.path.exists(AUDIO_DIR):
    os.makedirs(AUDIO_DIR)

# Initialize Flask app
app = Flask(__name__)

uploaded_text = ""

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    token = request.json.get('token')
    # Process the token here (e.g., verify with Google and authenticate the user)
    return jsonify(success=True), 200


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF file upload and extract text."""
    global uploaded_text
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        extracted_text = extract_text_from_pdf(file)
        if is_medical_content(extracted_text):
            uploaded_text = extracted_text
            return jsonify({"text": "PDF uploaded successfully and text extracted."}), 200
        else:
            return jsonify({"error": "The file does not contain valid medical content."}), 400
    return jsonify({"error": "Invalid file format. Please upload a PDF."}), 400
@app.route('/summarize', methods=['POST'])
def summarize():
    """
    Generate a summary based on the uploaded report and summarization style.
    """
    global uploaded_text
    data = request.get_json()
    language = data.get('language', 'en')  # Get selected language
    summary_style = data.get('type', 'extractive')  # Default to 'extractive'

    if not uploaded_text:
        return jsonify({"error": "No PDF uploaded. Please upload a report first."}), 400

    # Extract basic information from the text
    patient_details = extract_patient_details(uploaded_text)

    # Generate summary based on the selected style
    if summary_style.lower() == 'extractive':
        summary = extractive_summary(uploaded_text)  # BioBERT-based extractive summary
    else:
        summary = summarize_medical(uploaded_text, summary_style)  # Fallback to other summary types

    # Combine patient details with the summary
    full_summary = f"""
    Patient Name: {patient_details.get('patient_name', 'N/A')}
    Doctor Name: {patient_details.get('doctor_name', 'N/A')}
    Date of Report: {patient_details.get('report_date', 'N/A')}
    Previous History: {patient_details.get('previous_history', 'N/A')}
    
    Summary:
    {summary}
    """

    # Translate summary to the selected language if it's not English
    if language != 'en':
        full_summary = translate_text(full_summary, language)

    # Generate audio of the summary
    audio_file = generate_audio_from_text(full_summary, language)

    return jsonify({"summary": full_summary, "audio_url": audio_file}), 200


def extract_patient_details(text):
    """Extract patient name and report date from the given medical report text."""
    # Patterns to extract patient name and report date
    patient_name_pattern = r"(?i)Patient\s*Name\s*:\s*([A-Za-z\s\.]+)"
    report_date_pattern = r"(?i)(Collected|Reported)\s*:\s*([0-9]{2}/[A-Za-z]{3}/[0-9]{4})"
    
    # Search for patient name
    patient_name_match = re.search(patient_name_pattern, text)
    patient_name = patient_name_match.group(1).strip() if patient_name_match else "Not found"

    # Search for report date (takes the first occurrence, which might be 'Collected' date)
    report_date_match = re.search(report_date_pattern, text)
    report_date = report_date_match.group(2) if report_date_match else "Not found"

    # Return extracted details
    patient_details = {
        "patient_name": patient_name,
        "report_date": report_date,
    }

    return patient_details

@app.route('/identify_problems', methods=['POST'])
def identify_problems():
    """Identify medical problems and provide explanations."""
    global uploaded_text
    data = request.get_json()
    language = data.get('language', 'en')  # Get selected language

    if not uploaded_text:
        return jsonify({"error": "No PDF uploaded. Please upload a report first."}), 400

    problems_explanation = explain_medical_problems(uploaded_text)

    # Translate explanation to the selected language if it's not English
    if language != 'en':
        problems_explanation = translate_text(problems_explanation, language)

    # Generate audio of the explanation
    audio_file = generate_audio_from_text(problems_explanation, language)

    return jsonify({"explanation": problems_explanation, "audio_url": audio_file}), 200
@app.route('/ask_question', methods=['POST'])
def ask_question():
    """Answer a user's question based on the uploaded report."""
    global uploaded_text
    data = request.get_json()
    question = data.get('question')

    if not uploaded_text:
        return jsonify({"error": "No PDF uploaded. Please upload a report first."}), 400

    if not question:
        return jsonify({"error": "No question provided."}), 400

    # Extract relevant content based on the question
    answer = generate_answer(question, uploaded_text)

    return jsonify({"answer": answer}), 200

@app.route('/explain', methods=['POST'])
def explain():
    """Provide explanations for medical problems."""
    global uploaded_text
    data = request.get_json()
    language = data.get('language', 'en')  # Get selected language

    if not uploaded_text:
        return jsonify({"error": "No PDF uploaded. Please upload a report first."}), 400

    # Extract and explain medical problems
    problems_explanation = explain_medical_problems(uploaded_text)

    # Translate explanation to the selected language if it's not English
    if language != 'en':
        problems_explanation = translate_text(problems_explanation, language)

    # Generate audio of the explanation
    audio_file = generate_audio_from_text(problems_explanation, language)

    return jsonify({"explanation": problems_explanation, "audio_url": audio_file}), 200
def explain_medical_problems(text):
    """Identify medical problems and explain them."""
    MAX_LENGTH = 1500
    
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]
    
    prompt = f"Identify any medical problems in the following report and explain their symptoms, causes, cures, and suggested treatments: {text}"
    response = gen_model.generate_content(prompt)

    if response and hasattr(response, 'text') and response.text:
        # Remove * symbols from the response text
        cleaned_response = response.text.replace('*', '')
        return cleaned_response
    else:
        return "No problems identified or unable to generate explanation."

def generate_answer(question, report_text):
    """Generate an answer to the user's question based on the uploaded report."""
    try:
        # Preprocess the question and report text (e.g., remove unnecessary parts)
        question = question.lower().strip()
        report_text = report_text.lower()

        # Use a simple approach to check if the question contains specific keywords related to diseases, symptoms, etc.
        if "symptoms" in question:
            # Extract symptoms from the report using NER or predefined pattern matching
            symptoms = extract_symptoms(report_text)
            if symptoms:
                return f"The reported symptoms are: {', '.join(symptoms)}."
            else:
                return "Sorry, no symptoms were found in the report."

        elif "treatment" in question:
            # Extract treatment-related information from the report
            treatment = extract_treatment(report_text)
            if treatment:
                return f"The recommended treatment is: {treatment}."
            else:
                return "Sorry, no treatment information was found in the report."

        elif "diagnosis" in question:
            # Extract diagnosis-related information from the report
            diagnosis = extract_diagnosis(report_text)
            if diagnosis:
                return f"The diagnosis is: {diagnosis}."
            else:
                return "Sorry, no diagnosis information was found in the report."

        else:
            return "Sorry, I could not understand the question. Please try asking something specific like symptoms, treatment, or diagnosis."
    except Exception as e:
        return f"Error generating answer: {str(e)}"

def extract_symptoms(text):
    """Extract symptoms mentioned in the medical report."""
    # Placeholder for symptom extraction logic (could use NLP or regex)
    # For now, we're just simulating with a hardcoded response
    return ["fever", "cough", "fatigue"]

def extract_treatment(text):
    """Extract treatment options mentioned in the medical report."""
    # Placeholder for treatment extraction logic
    return "Pain relievers, bed rest"

def extract_diagnosis(text):
    """Extract diagnosis information from the medical report."""
    # Placeholder for diagnosis extraction logic
    return "Viral Infection"

def translate_text(text, language):
    """Translate text to the selected language using googletrans."""
    translator = Translator()
    translated = translator.translate(text, dest=language)
    return translated.text

def generate_audio_from_text(text, language):
    """Generate audio from text using gTTS and save it to a file."""
    try:
        tts = gTTS(text=text, lang=language, slow=False)
        audio_filename = f"{str(hash(text))}.mp3"  # Generate a unique filename based on the text hash
        audio_file_path = os.path.join(AUDIO_DIR, audio_filename)  # Path to save the audio file
        tts.save(audio_file_path)  # Save the audio to the file
        return audio_file_path  # Return the file path
    except Exception as e:
        print(f"Error generating audio: {str(e)}")
        return None

if __name__ == "__main__":
    app.run(debug=True)
