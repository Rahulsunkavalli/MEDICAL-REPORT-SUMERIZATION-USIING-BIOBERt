from gtts import gTTS
import io

def text_to_speech(text, language='en'):
    """Convert text to speech and return as a file-like object."""
    tts = gTTS(text, lang=language)
    audio_fp = io.BytesIO()
    tts.save(audio_fp)
    audio_fp.seek(0)  # Reset the pointer to the beginning
    return audio_fp
