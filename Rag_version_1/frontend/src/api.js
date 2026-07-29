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
 * @param {string} question
 * @param {string} namespace - namespace returned from uploadPdf()
 * @returns {Promise<{answer: string, sources: Array}>}
 */
export async function askQuestion(question, namespace) {
  const response = await apiClient.post("/query", {
    question,
    namespace,
  });

  return response.data;
}
