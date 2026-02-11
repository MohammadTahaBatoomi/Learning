// Face Authentication Front-End Demo
// Requires: face-api.js (loaded via CDN) and camera permissions. No backend needed.

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const statusEl = document.getElementById('status');
const resultEl = document.getElementById('result');
const loaderEl = document.getElementById('loader');
const toastEl = document.getElementById('toast');
const tabs = document.querySelectorAll('.tab');
const panels = document.querySelectorAll('.panel');
const manualForm = document.getElementById('manual-form');
const registerForm = document.getElementById('register-form');

const MODEL_URL = 'https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/weights';
const STORAGE_KEY = 'face-auth-users';
const MATCH_THRESHOLD = 0.45; // smaller = stricter
const DETECTION_INTERVAL = 450; // ms between detections

let stream = null;
let detectTimer = null;
let modelsReady = false;
let dims = { width: 0, height: 0 };
let storedEmbeddings = [];

function setStatus(msg) {
  statusEl.textContent = msg;
}

function setResult(msg, tone = 'muted') {
  resultEl.textContent = msg;
  resultEl.style.color = tone === 'success' ? '#5dd4a0' : tone === 'danger' ? '#ff6b6b' : 'var(--muted)';
}

function showLoader(state) {
  loaderEl.classList.toggle('active', state);
}

function toast(msg) {
  toastEl.textContent = msg;
  toastEl.classList.add('show');
  setTimeout(() => toastEl.classList.remove('show'), 2600);
}

function loadStoredEmbeddings() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return parsed.map(item => ({
      name: item.name,
      descriptor: new Float32Array(item.descriptor)
    }));
  } catch (err) {
    console.error('Failed to load embeddings', err);
    return [];
  }
}

function saveEmbedding(name, descriptor) {
  const existing = loadStoredEmbeddings();
  existing.push({ name, descriptor: Array.from(descriptor) });
  localStorage.setItem(STORAGE_KEY, JSON.stringify(existing));
  storedEmbeddings = loadStoredEmbeddings();
}

async function loadModels() {
  setStatus('Loading face models…');
  await Promise.all([
    faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
    faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
    faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL)
  ]);
  modelsReady = true;
  setStatus('Models loaded. Align your face in frame.');
}

async function initCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    setStatus('Camera not supported. Use manual login.');
    toast('Camera not available on this device.');
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'user' }, audio: false });
    video.srcObject = stream;
    await video.play();
    dims = { width: video.videoWidth, height: video.videoHeight };
    overlay.width = dims.width;
    overlay.height = dims.height;
    setStatus('Camera ready. Hold still for verification.');
  } catch (err) {
    console.error('Camera error', err);
    setStatus('Camera access denied. Use manual login.');
    toast('Camera permission needed for face login.');
  }
}

function clearCanvas() {
  const ctx = overlay.getContext('2d');
  ctx.clearRect(0, 0, overlay.width, overlay.height);
}

function drawBox(box, tone) {
  const ctx = overlay.getContext('2d');
  ctx.strokeStyle = tone === 'success' ? '#5dd4a0' : '#ff6b6b';
  ctx.lineWidth = 3;
  ctx.beginPath();
  if (ctx.roundRect) {
    ctx.roundRect(box.x, box.y, box.width, box.height, 10);
  } else {
    ctx.rect(box.x, box.y, box.width, box.height);
  }
  ctx.stroke();
}

function bestMatch(descriptor) {
  if (!storedEmbeddings.length) return { match: null, distance: 1 };
  let best = { match: null, distance: Number.MAX_VALUE };
  storedEmbeddings.forEach(entry => {
    const dist = faceapi.euclideanDistance(descriptor, entry.descriptor);
    if (dist < best.distance) best = { match: entry, distance: dist };
  });
  return best;
}

async function detectFace() {
  if (!modelsReady || !video.srcObject) return;
  showLoader(true);
  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.5 }))
    .withFaceLandmarks()
    .withFaceDescriptor();
  showLoader(false);

  clearCanvas();

  if (!detection) {
    setResult('No face detected. Center your face.', 'muted');
    return;
  }

  const resized = faceapi.resizeResults(detection, dims);
  drawBox(resized.detection.box, 'muted');

  const { match, distance } = bestMatch(resized.descriptor);
  if (match && distance < MATCH_THRESHOLD) {
    setResult(`Access granted · Welcome ${match.name} (${distance.toFixed(2)})`, 'success');
    drawBox(resized.detection.box, 'success');
  } else {
    setResult('Access denied · Face not recognized', 'danger');
    drawBox(resized.detection.box, 'danger');
  }
}

function startDetectionLoop() {
  if (detectTimer) clearInterval(detectTimer);
  detectTimer = setInterval(detectFace, DETECTION_INTERVAL);
}

function stopDetectionLoop() {
  if (detectTimer) clearInterval(detectTimer);
}

function handleManualLogin(e) {
  e.preventDefault();
  const data = new FormData(manualForm);
  const email = data.get('email');
  const password = data.get('password');
  if (email && password) {
    setResult('Manual access granted (fallback).', 'success');
    toast('Logged in manually.');
  }
}

async function handleRegister(e) {
  e.preventDefault();
  if (!modelsReady) return toast('Models are still loading.');
  if (!video.srcObject) return toast('Camera unavailable.');

  const data = new FormData(registerForm);
  const name = data.get('name').trim();
  if (!name) return toast('Enter a display name.');

  showLoader(true);
  const detection = await faceapi
    .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.55 }))
    .withFaceLandmarks()
    .withFaceDescriptor();
  showLoader(false);

  if (!detection) {
    setResult('Unable to capture face. Try again.', 'danger');
    return;
  }
  saveEmbedding(name, detection.descriptor);
  setResult(`Registered ${name}. You can now login with face.`, 'success');
  toast('Face registered locally.');
}

function wireTabs() {
  tabs.forEach(btn => {
    btn.addEventListener('click', () => {
      tabs.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const target = btn.dataset.target;
      panels.forEach(p => p.classList.toggle('hidden', p.id !== target));
    });
  });
}

async function bootstrap() {
  storedEmbeddings = loadStoredEmbeddings();
  wireTabs();
  manualForm.addEventListener('submit', handleManualLogin);
  registerForm.addEventListener('submit', handleRegister);

  await Promise.all([loadModels(), initCamera()]);
  if (video.srcObject) startDetectionLoop();
}

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopDetectionLoop();
  } else if (video.srcObject) {
    startDetectionLoop();
  }
});

window.addEventListener('beforeunload', () => stopDetectionLoop());

bootstrap();
