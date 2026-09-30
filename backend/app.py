from flask import Flask, jsonify, request
from dotenv import load_dotenv
from google import genai
import os

# Load .env file
load_dotenv()

# Create Flask app
app = Flask(__name__)

# Get Gemini API key
API_KEY = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=API_KEY)


@app.route("/")
def home():
    return jsonify({
        "message": "Coastal AI backend is running"
    })


@app.route("/advisory", methods=["POST"])
def advisory():

    # Receive risk data from GEE/dashboard
    risk_data = request.get_json()

    # Check whether data was received
    if not risk_data:
        return jsonify({
            "error": "No risk data received"
        }), 400

    # Create prompt for Gemini
    prompt = f"""
You are a coastal disaster early-warning assistant.

Analyze ONLY the risk data provided below.

Risk data:
{risk_data}

Generate a concise emergency advisory containing:

1. Risk level
2. Main reason for the risk
3. Key vulnerable factors
4. Recommended actions for local disaster authorities
5. A short public warning message

Important rules:
- Do not invent weather conditions.
- Do not invent infrastructure damage.
- Do not invent locations or affected facilities.
- Do not claim that flooding is currently happening.
- Clearly distinguish predicted/modelled risk from confirmed events.
- Base every factual statement on the supplied risk data.
"""

    # Send risk data to Gemini
    interaction = client.interactions.create(
        model="gemini-3.8-flash",
        input=prompt
    )

    # Return Gemini advisory
    return jsonify({
        "location": risk_data.get("location"),
        "advisory": interaction.output_text
    })


if __name__ == "__main__":
    app.run(debug=True)