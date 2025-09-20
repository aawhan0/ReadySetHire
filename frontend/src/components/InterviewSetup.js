import React, {useState} from 'react';

function InterviewSetup(){
    const [jobDescription, setJobDescription] = useState('');
    const [questions, setQuestions] = useState([]);
    const [loading, setLoading] = useState(false);

    const handleSubmit = async(e) => {
        e.preventDefault();
        setLoading(true);
        setQuestions([]);

        try{
            const response = await fetch('http://localhost:5000/generate-questions', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ job_description: jobDescription}),
            })
            const data= await response.json();
            setQuestions(data.questions || []);
        } catch(error) {
            console.error('Error fetching questions:', error);
            alert('Failed to get questions from backend');
        } finally {
            setLoading(false);
        }
    }

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
                    {loading ? 'Generating Questions...': 'Generate Questions'}
                </button>
            </form>

            <div>
                {questions.length >0 && (
                    <>
                        <h3>Generated Questions: </h3>
                        <ul>
                            {questions.map((q, idx) => (
                                <li key= {idx}>{q}</li>
                            ))}
                        </ul>
                    </>
                )}
            </div>
        </div>
    );
}

export default InterviewSetup;