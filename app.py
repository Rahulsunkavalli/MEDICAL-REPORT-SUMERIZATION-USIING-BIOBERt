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

# Load environment variables
load_dotenv()

# Load BioBERT for Named Entity Recognition
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = BertForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1")
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Initialize Google Generative AI
genai.configure(api_key=os.getenv("API"))
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
    """Generate a summary based on the uploaded report and summarization style."""
    global uploaded_text
    data = request.get_json()
    language = data.get('language', 'en')  # Get selected language
    summary_style = data.get('type', 'Brief')  # Get summary style

    if not uploaded_text:
        return jsonify({"error": "No PDF uploaded. Please upload a report first."}), 400

    # Extract basic information from the text
    patient_details = extract_patient_details(uploaded_text)
    
    # Generate summary using the summarizer
    summary = summarize_medical(uploaded_text, summary_style)

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
        return response.text
    else:
        return "No problems identified or unable to generate explanation."

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
