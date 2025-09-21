import os
from dotenv import load_dotenv
load_dotenv()
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.genai import types

# Load Gemini API key from .env
google_gemini_api_key = os.getenv("GOOGLE_GEMINI_API_KEY")

# Initialize Gemini client with API key
client = genai.Client(api_key=google_gemini_api_key)

use_mock = os.getenv("USE_MOCK", "true").lower() == "true"

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

def mock_ai_questions(job_description):
    return [
        f"What motivates you about the role of {job_description}?",
        "Describe a project where you faced significant challenges.",
        "How do you stay updated with the latest industry trends?",
        "What are your salary expectations?",
        "Where do you see yourself in five years?"
    ]



def generate_ai_questions(job_description):
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[types.Part.from_text(f"Generate interview questions for the job: {job_description}")],
            config=types.GenerateContentConfig(
                max_output_tokens=150,
                temperature=0.7,
            ),
        )
        print("AI response received:", response.text)
        text_output = response.text
        questions = [q.strip() for q in text_output.split('\n') if q.strip()]
        return questions
    except Exception as e:
        print("Error generating questions:", e)
        return []


@app.route('/')
def home():
    return "ReadySetHire backend running"

@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    job_description = data.get('job_description', '')
    if use_mock:
        questions = mock_ai_questions(job_description)
    else:
        questions = generate_ai_questions(job_description)
    return jsonify({"questions": questions})

if __name__ == '__main__':
    app.run(debug=True)
