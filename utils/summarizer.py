import google.generativeai as genai
from googletrans import Translator, LANGUAGES
import os
# Configure Google Generative AI
genai.configure(api_key=os.getenv("API_KEY"))  # Ensure the API key is correctly loaded
model = genai.GenerativeModel('gemini-pro')
translator = Translator()

def summarize_medical(text, style):
    if style == "Brief":
        prompt = f"Summarize the following medical report briefly: {text}"
    elif style == "Detailed":
        prompt = f"Provide a detailed summary of the following medical report: {text}"
    elif style == "Key Points":
        prompt = f"Summarize the key points of the following medical report: {text}"
    
    response = model.generate_content(prompt)
    
    if response and hasattr(response, 'text') and response.text:
        return response.text
    else:
        return "No summary could be generated."

def translate_text(text, target_lang):
    try:
        if text:
            translated = translator.translate(text, dest=target_lang)
            return translated.text
        else:
            return "Translation could not be performed."
    except Exception as e:
        return str(e)

def is_medical_content(text):
    medical_keywords = ["diagnosis", "treatment", "symptoms", "disease", "prescription", "therapy", "medical"]
    return any(keyword in text.lower() for keyword in medical_keywords)

def explain_medical_problems(text):
    MAX_LENGTH = 1500
    if len(text) > MAX_LENGTH:
        text = text[:MAX_LENGTH]
    
    prompt = f"Identify any medical problems in the following report and explain their symptoms, causes, cures, and suggested treatments: {text}"
    response = model.generate_content(prompt)

    if response and hasattr(response, 'text') and response.text:
        return response.text
    else:
        return "No problems identified or unable to generate explanation."
