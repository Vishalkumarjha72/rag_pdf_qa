import { useState } from "react";
import { uploadPdf } from "./api";

/**
 * Lets the user select and upload a PDF.
 * On successful upload, calls onUploadSuccess(namespace, filename)
 * so the parent (App) knows which document to query against.
 */
function UploadPanel({ onUploadSuccess }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("idle"); // idle | uploading | success | error
  const [errorMessage, setErrorMessage] = useState("");
  const [uploadedInfo, setUploadedInfo] = useState(null); // { filename, chunks_indexed }

  function handleFileChange(event) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setStatus("idle");
    setErrorMessage("");
  }

  async function handleUpload() {
    if (!selectedFile) return;

    setStatus("uploading");
    setErrorMessage("");

    try {
      const result = await uploadPdf(selectedFile);
      setStatus("success");
      setUploadedInfo({
        filename: result.filename,
        chunks_indexed: result.chunks_indexed,
      });
      onUploadSuccess(result.namespace, result.filename);
    } catch (error) {
      setStatus("error");
      const detail = error?.response?.data?.detail;
      setErrorMessage(detail || "Upload failed. Is the backend running?");
    }
  }

  return (
    <div className="panel">
      <h2>1. Upload a PDF</h2>

      <input
        type="file"
        accept="application/pdf"
        onChange={handleFileChange}
        disabled={status === "uploading"}
      />

      <button
        onClick={handleUpload}
        disabled={!selectedFile || status === "uploading"}
      >
        {status === "uploading" ? "Uploading..." : "Upload"}
      </button>

      {status === "success" && uploadedInfo && (
        <p className="status-success">
          Indexed "{uploadedInfo.filename}" ({uploadedInfo.chunks_indexed} chunks).
          You can ask questions now.
        </p>
      )}

      {status === "error" && (
        <p className="status-error">{errorMessage}</p>
      )}
    </div>
  );
}

export default UploadPanel;
