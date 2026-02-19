import os
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class TTSModel:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

        self.prompt = """
        Speak in a calm, friendly, and supportive tone.
        Sound like an experienced agricultural advisor guiding a farmer.
        Maintain clear pronunciation and moderate speaking speed.
        Pause slightly between steps for clarity.
        Emphasize warnings gently but clearly.
        Avoid sounding robotic or overly dramatic.
        Keep the tone practical, confident, and reassuring.
        """

    def synthesize(self, text: str):

        full_text = self.prompt + "\n\n" + text

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=full_text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Charon"
                        )
                    )
                ),
            ),
        )

        audio_bytes = response.candidates[0].content.parts[0].inline_data.data
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")

        return audio_base64
