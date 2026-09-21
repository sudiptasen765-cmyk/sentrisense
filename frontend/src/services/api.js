import axios from "axios";

// Falls back to the live production endpoint if VITE_API_BASE_URL is not
// set in .env — see .env.example. Override this for local/staging testing
// against a different API Gateway stage.
const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "https://av9stuspv9.execute-api.ap-south-1.amazonaws.com";

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 20000, // distilbert cold starts can be slow; give it real room
  headers: { "Content-Type": "application/json" },
});

/**
 * Calls POST /predict.
 * @param {string} text - the review text to analyze
 * @param {"logreg"|"distilbert"} model
 * @returns {Promise<{sentiment: string, confidence: number, model: string}>}
 * @throws {Error} with a `.response` shaped like the API's error body when
 *   the server returns a structured error ({error, message}), or a plain
 *   network error otherwise.
 */
export async function analyzeSentiment(text, model = "logreg") {
  const { data } = await client.post("/predict", { text, model });
  return data;
}
