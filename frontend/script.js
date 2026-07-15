/**
 * AIDL Buildathon Forge 2026 — Day 3: Computer Vision
 * =====================================================
 *
 * frontend/script.js
 *
 * What this file does
 * --------------------
 * Polls the FastAPI backend's GET /count endpoint at a fixed
 * interval and updates the dashboard with the latest hand detection
 * result: finger count, detection status, FPS, and last-updated time.
 *
 * This is intentionally simple polling (setInterval + fetch) rather
 * than WebSockets — easy for students to read end-to-end, and a
 * natural discussion point for "Future Improvements" in the README.
 */

// Base URL of the FastAPI backend. Update this if you run uvicorn on
// a different host or port (see backend/.env / .env.example).
const API_BASE_URL = "http://127.0.0.1:8000";

// How often to poll the backend while polling is active, in milliseconds.
const POLL_INTERVAL_MS = 1000;

// Holds the interval ID returned by setInterval, so we can cancel it
// when the user clicks "Stop Polling". `null` means polling is off.
let pollIntervalId = null;

// ---------------------------------------------------------------------------
// DOM references
// ---------------------------------------------------------------------------
const fingerCountEl = document.getElementById("finger-count");
const statusDotEl = document.getElementById("status-dot");
const statusTextEl = document.getElementById("status-text");
const handDetectedValueEl = document.getElementById("hand-detected-value");
const fpsValueEl = document.getElementById("fps-value");
const updatedAtValueEl = document.getElementById("updated-at-value");

const startButton = document.getElementById("start-button");
const stopButton = document.getElementById("stop-button");
const refreshButton = document.getElementById("refresh-button");

// ---------------------------------------------------------------------------
// Rendering helpers
// ---------------------------------------------------------------------------

/**
 * Update the dashboard to reflect a successful /count response.
 * @param {{hand_detected: boolean, finger_count: number|null, fps: number, updated_at: number}} data
 */
function renderSuccess(data) {
  fingerCountEl.textContent = data.hand_detected ? String(data.finger_count) : "--";

  statusDotEl.className = "status-dot status-dot--live";
  statusTextEl.textContent = data.hand_detected ? "Hand detected" : "Live — no hand in frame";

  handDetectedValueEl.textContent = data.hand_detected ? "Yes" : "No";
  fpsValueEl.textContent = data.fps.toFixed(1);

  const updatedDate = new Date(data.updated_at * 1000);
  updatedAtValueEl.textContent = updatedDate.toLocaleTimeString();
}

/**
 * Update the dashboard to reflect a failed request (backend
 * unreachable, network error, or non-2xx response).
 */
function renderOffline() {
  fingerCountEl.textContent = "--";

  statusDotEl.className = "status-dot status-dot--offline";
  statusTextEl.textContent = "Backend unreachable";

  handDetectedValueEl.textContent = "--";
  fpsValueEl.textContent = "--";
  updatedAtValueEl.textContent = "--";
}

// ---------------------------------------------------------------------------
// Networking
// ---------------------------------------------------------------------------

/**
 * Fetch the latest detection result from the backend and update the
 * dashboard accordingly. Safe to call whether or not polling is active.
 */
async function fetchCountOnce() {
  try {
    const response = await fetch(`${API_BASE_URL}/count`);

    if (!response.ok) {
      throw new Error(`Backend responded with status ${response.status}`);
    }

    const data = await response.json();
    renderSuccess(data);
  } catch (error) {
    console.error("Failed to fetch /count:", error);
    renderOffline();
  }
}

// ---------------------------------------------------------------------------
// Polling controls
// ---------------------------------------------------------------------------

/** Start polling the backend on a fixed interval. */
function startPolling() {
  if (pollIntervalId !== null) {
    return; // Already polling.
  }

  fetchCountOnce(); // Fetch immediately, don't wait for the first interval tick.
  pollIntervalId = setInterval(fetchCountOnce, POLL_INTERVAL_MS);

  startButton.disabled = true;
  stopButton.disabled = false;
}

/** Stop polling the backend. */
function stopPolling() {
  if (pollIntervalId !== null) {
    clearInterval(pollIntervalId);
    pollIntervalId = null;
  }

  startButton.disabled = false;
  stopButton.disabled = true;
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

startButton.addEventListener("click", startPolling);
stopButton.addEventListener("click", stopPolling);
refreshButton.addEventListener("click", fetchCountOnce);

// Fetch once on page load so the dashboard isn't empty before the
// user clicks anything, without starting continuous polling yet.
fetchCountOnce();