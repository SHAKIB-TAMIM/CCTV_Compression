// server.js
const express = require('express');
const http = require('http');
const socketio = require('socket.io');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(__dirname + '/public'));

const server = http.createServer(app);
const io = socketio(server, { cors: { origin: '*' } });

const STREAM_NS = '/stream'; // edge nodes connect here
const VIEW_NS = '/view';     // dashboards connect here

const streamNS = io.of(STREAM_NS);
const viewNS = io.of(VIEW_NS);

let bytesLastInterval = 0;
const BANDWIDTH_MONITOR_INTERVAL_MS = 3000; // 3s
const BANDWIDTH_THRESHOLD_KBPS = 128; // example threshold

streamNS.on('connection', (socket) => {
  console.log('[stream] connected', socket.id);

  socket.on('frame', (msg) => {
  try {
    const jsonStr = JSON.stringify(msg);
    const sizeBytes = Buffer.byteLength(jsonStr, 'utf8');
    bytesLastInterval += sizeBytes;

    // Convert frame to base64 string (if not already)
    // Assuming msg is a raw image buffer from edge node:
    //const base64Frame = Buffer.from(msg).toString('base64');

    // Broadcast to dashboard
    viewNS.emit('frame', msg);
  } catch (e) {
    console.error("Frame broadcast error:", e);
  }
});

  socket.on('disconnect', () => {
    console.log('[stream] disconnected', socket.id);
  });
});

viewNS.on('connection', (socket) => {
  console.log('[view] connected', socket.id);
  socket.on('disconnect', () => {
    console.log('[view] disconnected', socket.id);
  });
});

// simple bandwidth monitor + control messages
setInterval(() => {
  const kbps = (bytesLastInterval * 8) / (BANDWIDTH_MONITOR_INTERVAL_MS/1000) / 1000;
  console.log(`Incoming stream ~ ${kbps.toFixed(2)} kbps`);
  if (kbps > BANDWIDTH_THRESHOLD_KBPS) {
    const controlMsg = {
      bg_quality: 10,
      roi_quality: 80,
      bg_scale: 0.4,
      detect_every_n: 5
    };
    streamNS.emit('control', controlMsg);
  } else {
    const controlMsg = {
      bg_quality: 30,
      roi_quality: 90,
      bg_scale: 0.6,
      detect_every_n: 3
    };
    streamNS.emit('control', controlMsg);
  }
  bytesLastInterval = 0;
}, BANDWIDTH_MONITOR_INTERVAL_MS);

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server listening on port ${PORT}`));
