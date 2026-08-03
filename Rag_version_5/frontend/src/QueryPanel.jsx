import { useState } from "react";
import { askQuestion } from "./api";

/**
 * Lets the user have a multi-turn conversation about the currently
 * uploaded document (identified by `namespace`). Disabled until a
 * namespace exists.
 *
 * V3 changes from V2:
 *  - Answers now STREAM in token-by-token instead of appearing all at
 *    once. A turn is added to `turns` immediately with an empty answer,
 *    then askQuestion()'s onToken callback appends text to it as tokens
 *    arrive — the UI updates on every token, not just once at the end.
 *  - `sources` and `metadata` (confidence + which chunks were cited) only
 *    arrive in the final "done" event, so a turn's answer can be fully
 *    visible before its sources/metadata are filled in.
 *  - Since only one question can be in flight at a time (the Ask button
 *    is disabled while status === "asking"), onToken/onDone always know
 *    the turn they're updating is the LAST one in the array — no need to
 *    track an explicit turn index.
 */
function QueryPanel({ namespace, sessionId, onSessionIdChange }) {
  const [question, setQuestion] = useState("");
  const [status, setStatus] = useState("idle"); // idle | asking | error
  const [errorMessage, setErrorMessage] = useState("");
  const [turns, setTurns] = useState([]); // [{ question, answer, sources, metadata }]

  function appendToLastAnswer(text) {
    setTurns((prevTurns) => {
      const updated = [...prevTurns];
      const lastIndex = updated.length - 1;
      updated[lastIndex] = { ...updated[lastIndex], answer: updated[lastIndex].answer + text };
      return updated;
    });
  }

  function fillInLastTurnResult(payload) {
    setTurns((prevTurns) => {
      const updated = [...prevTurns];
      const lastIndex = updated.length - 1;
      updated[lastIndex] = {
        ...updated[lastIndex],
        sources: payload.sources,
        metadata: payload.metadata,
      };
      return updated;
    });
  }

  async function handleAsk() {
    if (!question.trim() || !namespace) return;

    const askedQuestion = question;
    setStatus("asking");
    setErrorMessage("");
    setQuestion("");

    // Placeholder turn, added immediately so the user's question shows up
    // right away — its answer starts empty and fills in as tokens stream.
    setTurns((prevTurns) => [
      ...prevTurns,
      { question: askedQuestion, answer: "", sources: [], metadata: null },
    ]);

    await askQuestion(askedQuestion, namespace, sessionId, {
      onToken: appendToLastAnswer,
      onDone: (payload) => {
        fillInLastTurnResult(payload);
        onSessionIdChange(payload.session_id);
        setStatus("idle");
      },
      onError: (detail) => {
        setStatus("error");
        setErrorMessage(detail || "Query failed. Is the backend running?");
      },
    });
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
          {turns.map((turn, index) => {
            const isStreamingThisTurn =
              status === "asking" && index === turns.length - 1;

            return (
              <div key={index} className="turn">
                <p className="turn-question">
                  <strong>You:</strong> {turn.question}
                </p>
                <p className="turn-answer">
                  <strong>Answer:</strong>{" "}
                  {turn.answer || (isStreamingThisTurn ? "…" : "")}
                </p>

                {turn.metadata && (
                  <p className={`confidence-badge confidence-${turn.metadata.confidence}`}>
                    Confidence: {turn.metadata.confidence}
                    {turn.metadata.cited_chunk_indices.length > 0 &&
                      ` — used ${turn.metadata.cited_chunk_indices.length} of ${turn.sources.length} retrieved source(s)`}
                  </p>
                )}

                {turn.sources?.length > 0 && (
                  <details className="turn-sources">
                    <summary>{turn.sources.length} source(s)</summary>
                    <ul className="sources-list">
                      {turn.sources.map((source, sourceIndex) => {
                        const wasCited = turn.metadata?.cited_chunk_indices?.includes(sourceIndex);
                        return (
                          <li key={sourceIndex} className={wasCited ? "source-cited" : ""}>
                            <strong>{source.source}</strong> — page {source.page}{" "}
                            (score: {source.score.toFixed(3)})
                            {wasCited && <span className="cited-tag"> · cited</span>}
                            <div className="source-text">{source.text}</div>
                          </li>
                        );
                      })}
                    </ul>
                  </details>
                )}
              </div>
            );
          })}
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
