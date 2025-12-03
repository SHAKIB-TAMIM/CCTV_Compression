const { spawn, exec } = require("child_process");

// CONFIG
const CAMERA_DEVICE = "/dev/video0"; // your webcam device
const STREAM_URL = "udp://127.0.0.1:1234"; // local test stream
let currentBitrate = 2000; // start bitrate
let ffmpegProcess = null;

// Start FFmpeg process
function startFFmpeg() {
  console.log(` Starting FFmpeg with bitrate: ${currentBitrate} kbps`);

  ffmpegProcess = spawn("ffmpeg", [
    "-f", "v4l2",
    "-framerate", "30",
    "-video_size", "640x480",
    "-i", CAMERA_DEVICE,
    "-b:v", `${currentBitrate}k`,
    "-bufsize", `${currentBitrate}k`,
    "-f", "mpegts",
    STREAM_URL
  ]);

  ffmpegProcess.stderr.on("data", (data) => {
    console.log(`FFmpeg: ${data}`);
  });

  ffmpegProcess.on("close", (code) => {
    console.log(` FFmpeg stopped with code ${code}`);
  });
}

// Stop FFmpeg process
function stopFFmpeg() {
  if (ffmpegProcess) {
    ffmpegProcess.kill("SIGTERM");
    ffmpegProcess = null;
  }
}

// Bandwidth check (Linux only)
function checkBandwidth() {
  exec("ifstat -i wlan0 1 1 | awk 'NR==3 {print $1}'", (err, stdout) => {
    const bw = parseFloat(stdout.trim());
    if (!isNaN(bw)) {
      adaptBitrate(bw);
    } else {
      console.log(" Could not read bandwidth");
    }
  });
}

// Adapt bitrate based on bandwidth
function adaptBitrate(bw) {
  let newBitrate;
  if (bw < 500) newBitrate = 500;
  else if (bw < 1000) newBitrate = 1000;
  else if (bw < 3000) newBitrate = 2000;
  else newBitrate = 4000;

  if (newBitrate !== currentBitrate) {
    console.log(` Bandwidth: ${bw} KB/s → Adjusting bitrate to ${newBitrate} kbps`);
    currentBitrate = newBitrate;

    stopFFmpeg();
    startFFmpeg();
  } else {
    console.log(` Bandwidth: ${bw} KB/s → Bitrate unchanged (${currentBitrate} kbps)`);
  }
}

// Start streaming
console.log(" Adaptive Bitrate Live Streaming started...");
startFFmpeg();

// Check every 10 seconds
setInterval(checkBandwidth, 10000);
