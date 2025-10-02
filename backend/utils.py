import re

def parse_gemini_output(raw_texts):
    text = raw_texts.replace('r\n', '\n').strip()

    pattern = r'(?:^\d\+\.\s+|\n-\s+)'
    raw_questions = re.split(pattern, text)

    if raw_questions and (not raw_questions[0].strip() or len(raw_questions[0].strip()) <5):
        raw_questions.pop(0)

    questions = []
    for q in raw_questions:
        cleaned= ''.join(q.strip().split())
        if len(cleaned) > 10:
            questions.append(cleaned)

    return questions