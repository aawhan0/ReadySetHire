import React, { useState } from 'react';

function InterviewSetup() {
  const [jobDescription, setJobDescription] = useState('');
  const [questions, setQuestions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setQuestions([]);
    setError(null);

    try {
      const response = await fetch('http://localhost:5000/generate-questions', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ job_description: jobDescription }),
      });
      const data = await response.json();
      setQuestions(data.questions || []);
    } catch (error) {
      console.error('Error fetching questions:', error);
      setError('Failed to get questions from backend.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>ReadySetHire: Interview Setup</h2>
      <form onSubmit={handleSubmit}>
        <label>
          Job Description:
          <textarea
            value={jobDescription}
            onChange={(e) => setJobDescription(e.target.value)}
            rows={6}
            cols={50}
            required
          />
        </label>
        <br />
        <button disabled={loading}>
          {loading ? 'Generating Questions...' : 'Generate Questions'}
        </button>
      </form>

      {error && <p style={{ color: 'red' }}>{error}</p>}

      <div>
        {questions.length > 0 && (
          <>
            <h3>Generated Questions: </h3>
            <ol style={{ paddingLeft: '20px', fontFamily: 'Arial, sans-serif', lineHeight: '1.6' }}>
              {questions.map((q, idx) => (
                <li key={idx} style={{ marginBottom: '12px', fontSize: '16px', fontWeight: '500' }}>
                  {q}
                </li>
              ))}
            </ol>
          </>
        )}
      </div>
    </div>
  );
}

export default InterviewSetup;
