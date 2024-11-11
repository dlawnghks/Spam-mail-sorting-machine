import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [title, setTitle] = useState('');
  const [content, setContent] = useState('');
  const [result, setResult] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();

    try {
      const response = await axios.post('http://localhost:5000/predict', {
        title: title,
        content: content,
      });
      setResult(response.data.result);
    } catch (error) {
      console.error('Error predicting email:', error);
    }
  };

  return (
    <div className="App">
      <h2>이메일 스팸 판별기</h2>
      <form onSubmit={handleSubmit}>
        <div>
          <label>이메일 제목:</label>
          <input
            type="text"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
            required
          />
        </div>
        <div>
          <label>이메일 내용:</label>
          <textarea
            value={content}
            onChange={(e) => setContent(e.target.value)}
            required
          />
        </div>
        <button type="submit">판별하기</button>
      </form>
      {result && (
        <div>
          <h3>판별 결과:</h3>
          <p>{result}</p>
        </div>
      )}
    </div>
  );
}

export default App;
