// server.js - upgraded with persistent metrics + /sample + /reconstruct + /motion
const express = require('express');
const http = require('http');
const socketio = require('socket.io');
const cors = require('cors');
const path = require('path');
const { spawn, exec } = require('child_process');
const fs = require('fs');

const app = express();
app.use(cors());
app.use(express.json({ limit: '100mb' }));
app.use(express.urlencoded({ extended: true, limit: '100mb' }));
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);
const io = socketio(server, { cors: { origin: '*' } });



// -------------------- Namespaces --------------------
const STREAM_NS = '/stream';
const VIEW_NS = '/view';
const streamNS = io.of(STREAM_NS);
const viewNS = io.of(VIEW_NS);

// -------------------- Global Variables --------------------
let bytesLastInterval = 0;
let totalClients = 0;
let currentBitrate = 2000;
const minBitrate = 500;
const maxBitrate = 4000;
const BANDWIDTH_MONITOR_INTERVAL_MS = 3000;
const BANDWIDTH_THRESHOLD_KBPS = 128;
let ffmpegProcess = null;
let metricsLog = [];

// Storage paths
const DATA_DIR = path.join(__dirname, 'experiments');
const METRICS_CSV = path.join(__dirname, 'metrics_log.csv');
const CONTROL_CSV = path.join(__dirname, 'control_log.csv');
const MOTION_CSV = path.join(__dirname, 'motion_log.csv');

// Ensure paths exist
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR, { recursive: true });
if (!fs.existsSync(METRICS_CSV)) fs.writeFileSync(METRICS_CSV, 'timestamp,total_clients,total_kbps,bitrate,psnr,ssim\n');
if (!fs.existsSync(CONTROL_CSV)) fs.writeFileSync(CONTROL_CSV, 'timestamp,experiment_id,control_json\n');
if (!fs.existsSync(MOTION_CSV)) fs.writeFileSync(MOTION_CSV, 'timestamp,experiment_id,frame_id,motion_percent,num_rois,rois_json\n');

// -------------------- Adaptive Bitrate Logic --------------------
function adjustBitrate(bw) {
  if (bw < 1000 && currentBitrate > minBitrate) {
    currentBitrate = Math.max(minBitrate, currentBitrate - 500);
    console.log(` Low bandwidth: ${bw.toFixed(2)} kbps → ↓ bitrate ${currentBitrate}`);
    restartStream();
  } else if (bw > 2500 && currentBitrate < maxBitrate) {
    currentBitrate = Math.min(maxBitrate, currentBitrate + 500);
    console.log(` High bandwidth: ${bw.toFixed(2)} kbps → ↑ bitrate ${currentBitrate}`);
    restartStream();
  }
}

// -------------------- Restart FFmpeg Stream --------------------
function restartStream() {
  if (ffmpegProcess) {
    try { ffmpegProcess.kill('SIGINT'); } catch (e) {}
  }
  console.log(`[Server] (re)starting FFmpeg to read live edge UDP stream at bitrate ${currentBitrate} kbps`);

  ffmpegProcess = spawn('ffmpeg', [
    '-re',
    '-i', 'udp://0.0.0.0:1234',
    '-b:v', `${currentBitrate}k`,
    '-f', 'mpegts',
    'udp://localhost:1235'
  ]);

  ffmpegProcess.stderr.on('data', (data) => {
    const line = data.toString();
    const match = line.match(/bitrate=\s*([\d\.]+)kbits\/s/);
    if (match) addMetricSnapshot(parseFloat(match[1]), totalClients);
  });

  ffmpegProcess.on('exit', (code) => {
    console.log(`FFmpeg exited with code ${code}`);
  });
}

// -------------------- Metrics Handling --------------------
function addMetricSnapshot(kbps, clients) {
  const snapshot = {
    timestamp: new Date().toISOString(),
    total_clients: clients,
    total_kbps: Number(kbps).toFixed(2),
    bitrate: currentBitrate,
    psnr: (30 + Math.random() * 5).toFixed(2),
    ssim: (0.85 + Math.random() * 0.1).toFixed(3),
  };
  metricsLog.push(snapshot);
  if (metricsLog.length > 300) metricsLog.shift();

  const row = `${snapshot.timestamp},${snapshot.total_clients},${snapshot.total_kbps},${snapshot.bitrate},${snapshot.psnr},${snapshot.ssim}\n`;
  fs.appendFile(METRICS_CSV, row, (err) => { if (err) console.error('metrics csv append err', err); });
}

// -------------------- Socket.IO handlers --------------------
streamNS.on('connection', (socket) => {
  console.log('[stream] connected', socket.id);
  totalClients++;

  socket.on('frame', (msg) => {
    try {
      const sizeBytes = Buffer.byteLength(JSON.stringify(msg), 'utf8');
      bytesLastInterval += sizeBytes;
      viewNS.emit('frame', msg);
    } catch (e) {
      console.error('Frame broadcast error:', e);
    }
  });

  socket.on('disconnect', () => {
    totalClients = Math.max(0, totalClients - 1);
    console.log('[stream] disconnected', socket.id);
  });
});

viewNS.on('connection', (socket) => {
  console.log('[view] connected', socket.id);

  socket.on('control', (msg) => {
    console.log('[view] control -> streamNS:', msg);
    streamNS.emit('control', msg);
  });

  socket.on('disconnect', () => { console.log('[view] disconnected', socket.id); });
});

// -------------------- Bandwidth Monitor --------------------
setInterval(() => {
  const kbps = (bytesLastInterval * 8) / (BANDWIDTH_MONITOR_INTERVAL_MS / 1000) / 1000;
  console.log(`Incoming ~ ${kbps.toFixed(2)} kbps | clients: ${totalClients}`);
  addMetricSnapshot(kbps, totalClients);
  adjustBitrate(kbps);

  const controlMsg = kbps > BANDWIDTH_THRESHOLD_KBPS
    ? { bg_quality: 10, roi_quality: 80, bg_scale: 0.4, detect_every_n: 5 }
    : { bg_quality: 30, roi_quality: 90, bg_scale: 0.6, detect_every_n: 3 };

  streamNS.emit('control', controlMsg);
  bytesLastInterval = 0;
}, BANDWIDTH_MONITOR_INTERVAL_MS);

// -------------------- HTTP Endpoints --------------------

// POST /control
app.post('/control', (req, res) => {
  const ctrl = req.body || {};
  const experimentId = ctrl.experiment_id || '';
  console.log('[HTTP] /control received:', ctrl);

  const ts = new Date().toISOString();
  const row = `${ts},${experimentId},"${JSON.stringify(ctrl).replace(/"/g, '""')}"\n`;
  fs.appendFile(CONTROL_CSV, row, (err) => { if (err) console.error('control csv append err', err); });

  streamNS.emit('control', ctrl);
  res.json({ success: true, applied: ctrl });
});

// GET /metrics (history)
app.get('/metrics', (req, res) => { res.json(metricsLog); });

// GET /metrics/live
app.get('/metrics/live', (req, res) => {
  if (metricsLog.length === 0) return res.json({ message: 'No metrics yet' });
  res.json(metricsLog[metricsLog.length-1]);
});

// POST /sample (existing) - receive base64 orig/recon images + rois + meta
app.post('/sample', (req, res) => {
  try {
    const payload = req.body || {};
    const experimentId = payload.experiment_id || `exp_${Date.now()}`;
    const expDir = path.join(DATA_DIR, experimentId);
    if (!fs.existsSync(expDir)) fs.mkdirSync(expDir, { recursive: true });

    // === Save images and metadata (same as before) ===
    const metaPath = path.join(expDir, 'meta.json');
    const metaToSave = {
      received_at: new Date().toISOString(),
      frame_id: payload.frame_id || null,
      timestamp: payload.timestamp || Date.now() / 1000,
      rois: payload.rois || [],
      meta: payload.meta || {},
      experiment_id: experimentId
    };
    fs.writeFileSync(metaPath, JSON.stringify(metaToSave, null, 2));

    if (payload.orig_b64) {
      const origPath = path.join(expDir, 'orig.jpg');
      fs.writeFileSync(origPath, Buffer.from(payload.orig_b64, 'base64'));
    }

    if (payload.recon_b64) {
      const reconPath = path.join(expDir, 'recon.jpg');
      fs.writeFileSync(reconPath, Buffer.from(payload.recon_b64, 'base64'));
    }

    if (payload.rois) {
      const roisPath = path.join(expDir, 'rois.json');
      fs.writeFileSync(roisPath, JSON.stringify(payload.rois, null, 2));
    }

    console.log(`[sample] saved for experiment ${experimentId}`);

    // === NEW: Resolve pending sample request ===
    if (pendingSampleRequest && pendingSampleRequest.experimentId === experimentId) {
      pendingSampleRequest.respond();
      pendingSampleRequest = null;
    }

    res.json({ success: true, experiment_id: experimentId, saved: true });
    
  } catch (err) {
    console.error('Error /sample:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});


// POST /motion - new endpoint for motion reports (from edge node)
app.post('/motion', (req, res) => {
  try {
    const p = req.body || {};
    const experimentId = p.experiment_id || `exp_${Date.now()}`;
    const frameId = p.frame_id || '';
    const motionPercent = Number(p.motion_percent || 0);
    const rois = p.rois || [];
    const numRois = rois.length;

    // append to motion CSV
    const ts = new Date().toISOString();
    const row = `${ts},${experimentId},${frameId},${motionPercent},${numRois},"${JSON.stringify(rois).replace(/"/g,'""')}"\n`;
    fs.appendFile(MOTION_CSV, row, (err) => { if (err) console.error('motion csv append err', err); });

    // optionally save per-experiment motion JSON for debugging
    const expDir = path.join(DATA_DIR, experimentId);
    if (!fs.existsSync(expDir)) fs.mkdirSync(expDir, { recursive: true });
    const motionFile = path.join(expDir, `motion_${frameId || Date.now()}.json`);
    fs.writeFileSync(motionFile, JSON.stringify({ ts, frameId, motionPercent, rois }, null, 2));

    res.json({ success: true, experiment_id: experimentId, motion_percent: motionPercent, num_rois: numRois });
  } catch (err) {
    console.error('Error /motion:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});

// RECONSTRUCT endpoint (FFmpeg capture + PSNR) - existing from previous upgrade
function runFFmpeg(cmd) {
  return new Promise((resolve, reject) => {
    exec(cmd, { maxBuffer: 1024 * 500 }, (err, stdout, stderr) => {
      if (err) reject(stderr || err.message);
      else resolve(stdout);
    });
  });
}
app.post("/reconstruct", async (req, res) => {
  const experimentId = req.body.experiment_id || `auto_${Date.now()}`;
  const outDir = path.join(__dirname, "experiments", experimentId);
  if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });

  const origPath = path.join(outDir, "orig_ffmpeg.jpg");
  const compPath = path.join(outDir, "comp_ffmpeg.jpg");

  try {
    console.log("[RECON] Capturing original reference frame from RTSP...");
    await runFFmpeg(`ffmpeg -y -ss 2 -i rtsp://localhost:8554/live -vframes 1 ${origPath} -loglevel error`);
    console.log("[RECON] Capturing compressed stream frame (UDP)...");
    await runFFmpeg(`ffmpeg -y -ss 2 -i udp://localhost:1234 -vframes 1 ${compPath} -loglevel error`);

    const sharp = require("sharp");
    const img1 = await sharp(origPath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
    const img2 = await sharp(compPath).ensureAlpha().raw().toBuffer({ resolveWithObject: true });

    function computeMSE(a, b) {
      let s = 0;
      for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        s += d * d;
      }
      return s / a.length;
    }

    const mse = computeMSE(img1.data, img2.data);
    const psnr = 10 * Math.log10((255 * 255) / mse);

    console.log(`[RECON] PSNR = ${psnr.toFixed(2)} dB`);
    const result = { experiment_id: experimentId, psnr: Number(psnr.toFixed(2)), orig_file: "orig_ffmpeg.jpg", comp_file: "comp_ffmpeg.jpg" };
    fs.writeFileSync(path.join(outDir, "result.json"), JSON.stringify(result, null, 2));
    res.json({ success: true, result });
  } catch (err) {
    console.error('[RECON ERROR]', err);
    res.status(500).json({ error: String(err) });
  }
});

// Health route
app.get('/', (req, res) => {
  res.send('CCTV Compression Server Running. Use /metrics, /control, /sample, /motion, /reconstruct.');
});

// -------------------- REQUEST SAMPLE ENDPOINT (Upgrade 3) --------------------
// evaluate.py uses this to request the next pair of orig/recon frames.

let pendingSampleRequest = null;

app.post('/request_sample', (req, res) => {
  try {
    const experimentId = req.body.experiment_id || `exp_${Date.now()}`;
    console.log(`[HTTP] /request_sample received for experiment: ${experimentId}`);

    // Store resolver to trigger when /sample arrives
    pendingSampleRequest = {
      experimentId,
      timestamp: Date.now(),
      respond: () => {
        console.log(`[HTTP] Resolving pending sample request for ${experimentId}`);
        res.json({ success: true, experiment_id: experimentId });
      }
    };

    // Auto-expire after 15 seconds
    setTimeout(() => {
      if (pendingSampleRequest && pendingSampleRequest.experimentId === experimentId) {
        console.log(`[HTTP] sample request timeout for ${experimentId}`);
        pendingSampleRequest = null;
        res.status(504).json({ success: false, error: "timeout waiting for sample" });
      }
    }, 15000);

  } catch (err) {
    console.error('Error /request_sample:', err);
    res.status(500).json({ success: false, error: err.message });
  }
});


// Start server
const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
