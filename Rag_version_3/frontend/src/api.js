import axios from "axios";

// Base URL for the FastAPI backend.
// In V1 we hardcode localhost; this becomes an env var once we containerize the frontend.
const API_BASE_URL = "http://localhost:8000";

const apiClient = axios.create({
  baseURL: API_BASE_URL,
});

/**
 * Uploads a PDF file to the backend for ingestion.
 * Backend chunks + embeds the PDF and returns a unique `namespace`
 * that must be used for subsequent queries against this document.
 *
 * @param {File} file - the PDF file selected by the user
 * @returns {Promise<{namespace: string, filename: string, chunks_indexed: number}>}
 */
export async function uploadPdf(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await apiClient.post("/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });

  return response.data;
}

/**
 * Asks a question against a previously uploaded document.
 *
 * V2: also sends/receives session_id so the backend can carry conversation
 * memory across turns. Pass `null` (or omit) for the first question of a
 * new conversation — the backend generates one and returns it.
 *
 * @param {string} question
 * @param {string} namespace - namespace returned from uploadPdf()
 * @param {string|null} sessionId - session_id from a prior askQuestion() call, or null for a new conversation
 * @returns {Promise<{answer: string, sources: Array, session_id: string}>}
 */
export async function askQuestion(question, namespace, sessionId = null) {
  const response = await apiClient.post("/query", {
    question,
    namespace,
    session_id: sessionId,
  });

  return response.data;
}
