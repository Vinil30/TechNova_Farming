import os
import base64
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from langsmith import traceable
load_dotenv()


class TTSModel:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

        self.prompt = """
Speak in Hindi.
Use simple Indian rural Hindi.
Speak in a calm, friendly, and supportive tone.
Sound like an experienced female agricultural advisor guiding a farmer.
Maintain clear pronunciation and moderate speaking speed.
Pause slightly between steps for clarity.
Avoid sounding robotic or dramatic.
"""

    def clean_text(self, text: str) -> str:
        if not text:
            return ""

        text = re.sub(r"[#*`>-]", "", text)
        text = re.sub(r"\d+\.", "", text)
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()
    
    @traceable(name="Gemini TTS Generation")
    def synthesize(self, text: str):
        cleaned_text = self.clean_text(text)
        full_text = self.prompt + "\n\n" + cleaned_text

        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-tts",
                contents=full_text,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        # language_code="hi-IN",
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Charon"
                            )
                        )
                    ),
                ),
            )

            # 🔥 SAFE CHECKS
            if (
                not response
                or not response.candidates
                or not response.candidates[0].content
                or not response.candidates[0].content.parts
                or not response.candidates[0].content.parts[0].inline_data
            ):
                print("TTS returned empty response")
                return None

            audio_bytes = response.candidates[0].content.parts[0].inline_data.data
            audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

            return audio_base64

        except Exception as e:
            print("TTS ERROR:", str(e))
            return None
