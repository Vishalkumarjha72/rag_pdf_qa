import { useState } from "react";
import { askQuestion } from "./api";

/**
 * Lets the user ask a question against the currently uploaded document
 * (identified by `namespace`, passed down from App). Disabled until
 * a namespace exists.
 */
function QueryPanel({ namespace }) {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("idle"); // idle | asking | success | error
  const [errorMessage, setErrorMessage] = useState("");
  const [answer, setAnswer] = useState(null); // { answer, sources }

  async function handleAsk() {
    if (!question.trim() || !namespace) return;

    setStatus("asking");
    setErrorMessage("");

    try {
      const result = await askQuestion(question, namespace);
      setAnswer(result);
      setStatus("success");
    } catch (error) {
      setStatus("error");
      const detail = error?.response?.data?.detail;
      setErrorMessage(detail || "Query failed. Is the backend running?");
    }
  }

  function handleKeyDown(event) {
    if (event.key === "Enter") {
      handleAsk();
    }
  }

  return (
    <div className="panel">
      <h2>2. Ask a question</h2>

      {!namespace && (
        <p className="status-hint">Upload a PDF first to enable questions.</p>
      )}

      <input
        type="text"
        placeholder="Ask something about the document..."
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={!namespace || status === "asking"}
      />

      <button
        onClick={handleAsk}
        disabled={!namespace || !question.trim() || status === "asking"}
      >
        {status === "asking" ? "Thinking..." : "Ask"}
      </button>

      {status === "error" && (
        <p className="status-error">{errorMessage}</p>
      )}

      {status === "success" && answer && (
        <div className="answer-block">
          <h3>Answer</h3>
          <p>{answer.answer}</p>

          {answer.sources?.length > 0 && (
            <>
              <h3>Sources</h3>
              <ul className="sources-list">
                {answer.sources.map((source, index) => (
                  <li key={index}>
                    <strong>{source.source}</strong> — page {source.page}{" "}
                    (score: {source.score.toFixed(3)})
                    <div className="source-text">{source.text}</div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default QueryPanel;
