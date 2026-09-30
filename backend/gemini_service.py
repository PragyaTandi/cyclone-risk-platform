import os
import json
from google import genai
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=API_KEY)


SYSTEM_PROMPT = """
You are an AI disaster-management advisory assistant.

You receive structured outputs from a quantitative cyclone-risk model.

Your task is to interpret those outputs and generate a concise
pre-landfall action advisory for municipal disaster-management authorities.

You must:
1. Identify the highest-risk zones.
2. Identify critical infrastructure at risk.
3. Explain the major predicted hazards.
4. Prioritize actions based on supplied risk values and infrastructure criticality.
5. Generate concise recommendations for authorities.
6. Clearly distinguish predictions from recommendations.

Rules:
- Do not invent numerical predictions.
- Do not modify supplied risk values.
- Do not claim certainty.
- Do not create infrastructure that is not present in the input.
- Base recommendations only on supplied data.
- Prioritize life safety and critical infrastructure.
- Recommendations must be practical but should not assume specific resources, equipment, personnel, or government capabilities unless they are explicitly provided.
- Do not invent evacuation routes, shelters, emergency units, equipment, or response resources.
- Phrase recommendations as suggested actions for authorities, not as actions that have already been executed.

Return the response using these sections:

SITUATION:
TOP PRIORITIES:
INFRASTRUCTURE AT RISK:
RECOMMENDED ACTIONS:
PUBLIC ADVISORY:
"""


def generate_advisory(risk_data):
    prompt = f"""
{SYSTEM_PROMPT}

Here is the structured cyclone-risk data:

{json.dumps(risk_data, indent=2)}

Generate the pre-landfall advisory.
"""

    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash"
    ]

    last_error = None

    for model in models_to_try:
        try:
            print(f"Trying Gemini model: {model}")

            response = client.models.generate_content(
                model=model,
                contents=prompt
            )

            print(f"Success with: {model}")
            return response.text

        except Exception as e:
            print(f"Failed with {model}: {e}")
            last_error = e

    raise last_error