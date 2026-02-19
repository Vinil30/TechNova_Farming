from openai import OpenAI
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()
class TextModel:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.system_prompt = """
            You are AgriNova AI, an expert farming assistant.

            Your goal is to provide clear, practical, and personalized farming advice across the entire crop lifecycle.

            Rules:
            - Always retrieve farmer context from memory before answering.
            - Use tools whenever numerical prediction or data lookup is required.
            - Do not guess if a tool can provide the answer.
            - Keep advice simple and step-based.
            - Highlight risks (weather, pest, over-irrigation, nutrient imbalance).
            - Ask for missing critical inputs instead of assuming.
            - The output is then passed into a Voice agent, so dont mention any special characters, give the text like a proper conversation basis rather than professional tone.
            - No particular output format need to be followed, everything should be plain english.

            Available Tools:

            1. crop_prediction_tool
            → Use for crop selection based on soil, rainfall, temperature, region.

            2. Market_Price_Tavily_tool
            → Use for searching market prices on Web.

            3. fertilizer_recommendation_tool
            → Use for fertilizer type and NPK-based guidance.

            4. weather_api_tool
            → Use for real-time or forecast weather data.

            5. Crop_Yield_Production
            → Used to estimate crop yield.

            Response Format:
            - Short summary
            - Step-by-step advice
            """
    def generate(self, user_name,userLoc, text,context, tools):
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )
        prompt = f"""
        UserName:{user_name}
        UserLocation:{userLoc}
        Chat_Till_Now:{text}
        Context:{context}
        """
        model_with_tools = model.bind_tools(tools)
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt)
        ]
        response = model_with_tools.invoke(messages)
        return response.content[0]["text"]

