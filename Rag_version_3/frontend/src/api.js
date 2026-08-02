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
 * Asks a question against a previously uploaded document, STREAMING the
 * answer back token-by-token via Server-Sent Events.
 *
 * V3 change from V2: axios can't easily read a streaming response body in
 * the browser, so this uses the native fetch() API instead, reading the
 * response as a stream and parsing it manually. Because there's no single
 * return value anymore (tokens arrive over time), this is callback-based
 * instead of returning a Promise<result> like uploadPdf() does — the
 * caller (QueryPanel) reacts to each event as it happens rather than
 * awaiting one final answer.
 *
 * The backend sends three kinds of SSE events, each a JSON object on its
 * own "data: " line, separated by a blank line:
 *   {"type": "token", "text": "..."}   -- zero or more, as the answer streams in
 *   {"type": "done", "session_id", "sources", "metadata"}  -- always last, on success
 *   {"type": "error", "detail": "..."} -- instead of "done", if something failed
 *
 * @param {string} question
 * @param {string} namespace - namespace returned from uploadPdf()
 * @param {string|null} sessionId - session_id from a prior call, or null for a new conversation
 * @param {{onToken?: (text: string) => void, onDone?: (payload: object) => void, onError?: (detail: string) => void}} callbacks
 */
export async function askQuestion(question, namespace, sessionId, { onToken, onDone, onError }) {
  let response;
  try {
    response = await fetch(`${API_BASE_URL}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, namespace, session_id: sessionId }),
    });
  } catch (err) {
    onError?.("Query failed. Is the backend running?");
    return;
  }

  if (!response.ok || !response.body) {
    onError?.("Query failed. Is the backend running?");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  // Chunks from the network don't line up with SSE event boundaries —
  // one read() can deliver half an event, or several at once. `buffer`
  // accumulates raw text across reads; we only process text up through
  // the last complete "\n\n" (the SSE event separator), and keep
  // whatever's left (a partial event) for the next chunk.
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    const events = buffer.split("\n\n");
    buffer = events.pop(); // last piece may be incomplete — hold it for next time

    for (const rawEvent of events) {
      const line = rawEvent.trim();
      if (!line.startsWith("data:")) continue;

      let payload;
      try {
        payload = JSON.parse(line.slice("data:".length).trim());
      } catch {
        continue; // defensively skip anything malformed rather than crash the stream
      }

      if (payload.type === "token") {
        onToken?.(payload.text);
      } else if (payload.type === "done") {
        onDone?.(payload);
      } else if (payload.type === "error") {
        onError?.(payload.detail);
      }
    }
  }
}
