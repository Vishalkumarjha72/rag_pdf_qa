import { useState } from "react";
import { askQuestion } from "./api";

/**
 * Lets the user have a multi-turn conversation about the currently
 * uploaded document (identified by `namespace`). Disabled until a
 * namespace exists.
 *
 * V2 changes from V1:
 *  - Renders the full back-and-forth (`turns`), not just the latest answer,
 *    so the user can see the conversation building up.
 *  - Sends/receives `sessionId` so the backend's LangGraph checkpointer
 *    can carry memory across turns. Ownership: App.jsx owns the actual
 *    sessionId value (same "lift state up" pattern as namespace); this
 *    component reports changes to it via onSessionIdChange, mirroring how
 *    UploadPanel reports namespace changes via onUploadSuccess.
 *  - Has a "New Conversation" button to reset sessionId back to null.
 *
 * Note on resetting `turns`: when a NEW document is uploaded, App.jsx
 * renders this component with a different `key` (key={namespace}), which
 * makes React unmount/remount it fresh — that's what resets `turns` in
 * that case, not any effect here. We deliberately avoid syncing turns to
 * sessionId via useEffect, since calling a state setter synchronously
 * inside an effect can trigger cascading re-renders (flagged by the
 * react-hooks/set-state-in-effect lint rule) — clearing state in response
 * to a user action belongs in the event handler that caused it, which is
 * exactly what handleNewConversation does below.
 */
function QueryPanel({ namespace, sessionId, onSessionIdChange }) {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("idle"); // idle | asking | error
  const [errorMessage, setErrorMessage] = useState("");
  const [turns, setTurns] = useState([]); // [{ question, answer, sources }]

  async function handleAsk() {
    if (!question.trim() || !namespace) return;

    const askedQuestion = question;
    setStatus("asking");
    setErrorMessage("");

    try {
      const result = await askQuestion(askedQuestion, namespace, sessionId);

      setTurns((prevTurns) => [
        ...prevTurns,
        { question: askedQuestion, answer: result.answer, sources: result.sources },
      ]);
      onSessionIdChange(result.session_id);
      setQuestion("");
      setStatus("idle");
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

  function handleNewConversation() {
    setTurns([]);
    onSessionIdChange(null);
  }

  return (
    <div className="panel">
      <div className="panel-header-row">
        <h2>2. Ask a question</h2>
        {turns.length > 0 && (
          <button className="new-conversation-btn" onClick={handleNewConversation}>
            New Conversation
          </button>
        )}
      </div>

      {!namespace && (
        <p className="status-hint">Upload a PDF first to enable questions.</p>
      )}

      {turns.length > 0 && (
        <div className="conversation-history">
          {turns.map((turn, index) => (
            <div key={index} className="turn">
              <p className="turn-question">
                <strong>You:</strong> {turn.question}
              </p>
              <p className="turn-answer">
                <strong>Answer:</strong> {turn.answer}
              </p>

              {turn.sources?.length > 0 && (
                <details className="turn-sources">
                  <summary>{turn.sources.length} source(s)</summary>
                  <ul className="sources-list">
                    {turn.sources.map((source, sourceIndex) => (
                      <li key={sourceIndex}>
                        <strong>{source.source}</strong> — page {source.page}{" "}
                        (score: {source.score.toFixed(3)})
                        <div className="source-text">{source.text}</div>
                      </li>
                    ))}
                  </ul>
                </details>
              )}
            </div>
          ))}
        </div>
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

      {status === "error" && <p className="status-error">{errorMessage}</p>}
    </div>
  );
}

export default QueryPanel;
