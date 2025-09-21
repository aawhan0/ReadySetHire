from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can communicate with backend

def generate_ai_questions(job_description):
    api_key = os.getenv('AI_API_KEY')
    headers = {"Authorization": f"Bearer {api_key}"}
    # The API endpoint and payload depend on the provider you choose
    payload = {
        "model": "text-generation-model",
        "prompt": f"Generate interview questions for the job: {job_description}",
        "max_tokens": 150,
    }
    response = requests.post("https://api.example.com/generate", json=payload, headers=headers)
    response_json = response.json()
    questions = response_json.get("questions", [])
    return questions
@app.route('/')
def home():
    return "ReadySetHire backend running"

# Sample API endpoint for AI question generation (to be expanded)
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    job_description = data.get('job_description', '')
    # Placeholder response
    questions = generate_ai_questions(job_description)
    return jsonify({"questions": questions})

if __name__ == '__main__':
    app.run(debug=True)
