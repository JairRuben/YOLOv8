(async () => {
  const video = document.getElementById('video');
  const canvas = document.createElement('canvas');
  const result = document.getElementById('result');
  const output = document.getElementById('output');
  const switchBtn = document.getElementById('switchCam');

  let stream = null;
  let facing = 'user';

  async function startStream() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: facing },
      audio: false
    });
    video.srcObject = stream;
  }

  switchBtn.addEventListener('click', () => {
    facing = facing === 'user' ? 'environment' : 'user';
    startStream();
  });

  await startStream();

  async function sendFrame() {
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);

    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg'));
    const form = new FormData();
    form.append('image', blob, 'frame.jpg');

    try {
      const res = await fetch('/detect', { method: 'POST', body: form });
      const imgBlob = await res.blob();
      const url = URL.createObjectURL(imgBlob);
      output.src = url;
    } catch (e) {
      console.error('Error en detección:', e);
    }
  }

  setInterval(sendFrame, 700);
})();
