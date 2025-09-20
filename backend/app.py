from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS so frontend can communicate with backend

@app.route('/')
def home():
    return "ReadySetHire Backend is Running"

# Sample API endpoint for AI question generation (to be expanded)
@app.route('/generate-questions', methods=['POST'])
def generate_questions():
    data = request.json
    job_description = data.get('job_description', '')
    # Placeholder response
    questions = [
        f"What interests you about this role based on: {job_description}?",
        "Can you tell me about your strengths related to this job?"
    ]
    return jsonify({"questions": questions})

if __name__ == '__main__':
    app.run(debug=True)
