import { useState } from "react";
import UploadPanel from "./UploadPanel";
import QueryPanel from "./QueryPanel";
import "./App.css";

function App() {
  // namespace identifies which uploaded PDF the QueryPanel should search.
  // null until a successful upload happens.
  const [namespace, setNamespace] = useState(null);
  const [activeFilename, setActiveFilename] = useState(null);

  function handleUploadSuccess(newNamespace, filename) {
    setNamespace(newNamespace);
    setActiveFilename(filename);
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
        <QueryPanel namespace={namespace} />
      </main>
    </div>
  );
}

export default App;
