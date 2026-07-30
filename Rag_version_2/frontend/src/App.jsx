import { useState } from "react";
import UploadPanel from "./UploadPanel";
import QueryPanel from "./QueryPanel";
import "./App.css";

function App() {
  // namespace identifies which uploaded PDF the QueryPanel should search.
  // null until a successful upload happens.
  const [namespace, setNamespace] = useState(null);
  const [activeFilename, setActiveFilename] = useState(null);

  // V2: sessionId identifies the ongoing conversation with the backend's
  // LangGraph checkpointer. null means "no conversation started yet" —
  // QueryPanel omits it on the next question, and the backend generates
  // a fresh one.
  const [sessionId, setSessionId] = useState(null);

  function handleUploadSuccess(newNamespace, filename) {
    setNamespace(newNamespace);
    setActiveFilename(filename);
    // A new document means the old conversation's history no longer
    // applies (it was about a different PDF) — start fresh.
    setSessionId(null);
  }

  return (
    <div className="app-container">
      <header>
        <h1>RAG PDF Q&A</h1>
        {activeFilename && (
          <p className="active-doc">Active document: {activeFilename}</p>
        )}
      </header>

      <main>
        <UploadPanel onUploadSuccess={handleUploadSuccess} />
        {/*
          key={namespace}: when a new PDF is uploaded, namespace changes,
          which makes React unmount the old QueryPanel and mount a brand
          new instance — resetting its internal `turns` state for free,
          instead of needing an effect inside QueryPanel to sync it.
        */}
        <QueryPanel
          key={namespace}
          namespace={namespace}
          sessionId={sessionId}
          onSessionIdChange={setSessionId}
        />
      </main>
    </div>
  );
}

export default App;
