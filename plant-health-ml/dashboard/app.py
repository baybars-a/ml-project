import io
import json
import numpy as np
import tensorflow as tf
import keras
from http.server import BaseHTTPRequestHandler, HTTPServer

print("Loading model...")
model = keras.models.load_model("results/model.keras")
print("Model loaded.")
print("")

PAGE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Plant Health Check</title>
<style>
body { font-family: system-ui, sans-serif; background: #f4f6f4; margin: 0; padding: 24px; color: #1a1a1a; }
.wrap { max-width: 560px; margin: 0 auto; }
h1 { font-size: 22px; margin: 0 0 4px; }
p.sub { color: #666; margin: 0 0 20px; font-size: 14px; }
#drop { border: 2px dashed #b5c4b5; border-radius: 12px; background: #fff; padding: 36px 16px; text-align: center; cursor: pointer; }
#drop.over { border-color: #4a8f4a; background: #f0f7f0; }
#drop p { margin: 0; color: #667; }
img#preview { max-width: 100%; border-radius: 10px; margin-top: 18px; display: none; }
#result { margin-top: 18px; padding: 18px; border-radius: 10px; display: none; background: #fff; }
.label { font-size: 26px; font-weight: 600; margin: 0 0 10px; }
.healthy { color: #2e7d32; }
.unhealthy { color: #c62828; }
.bar { height: 10px; background: #e6e6e6; border-radius: 6px; overflow: hidden; }
.fill { height: 100%; width: 0; transition: width .3s; }
.meta { font-size: 13px; color: #666; margin-top: 10px; font-family: ui-monospace, monospace; }
.note { margin-top: 22px; font-size: 12px; color: #888; line-height: 1.5; }
</style>
</head>
<body>
<div class="wrap">
  <h1>Plant Health Check</h1>
  <p class="sub">MobileNetV2 &middot; 99.55% test accuracy &middot; healthy or unhealthy</p>

  <div id="drop">
    <p>Click to choose an image, or drag one here</p>
    <input type="file" id="file" accept="image/*" style="display:none">
  </div>

  <img id="preview">

  <div id="result">
    <p class="label" id="label"></p>
    <div class="bar"><div class="fill" id="fill"></div></div>
    <p class="meta" id="meta"></p>
  </div>

  <p class="note">The model was trained on leaf photos taken against plain backgrounds.
  Busy backgrounds, whole plants or anything that is not a crop leaf will give unreliable
  results, and the model has no way to say "that is not a leaf".</p>
</div>

<script>
const drop = document.getElementById('drop');
const file = document.getElementById('file');
const preview = document.getElementById('preview');
const result = document.getElementById('result');

drop.onclick = () => file.click();
drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = e => {
  e.preventDefault();
  drop.classList.remove('over');
  if (e.dataTransfer.files.length) send(e.dataTransfer.files[0]);
};
file.onchange = () => { if (file.files.length) send(file.files[0]); };

function send(f) {
  preview.src = URL.createObjectURL(f);
  preview.style.display = 'block';
  document.getElementById('label').textContent = 'Checking...';
  document.getElementById('meta').textContent = '';
  document.getElementById('fill').style.width = '0';
  result.style.display = 'block';

  fetch('/predict', { method: 'POST', body: f })
    .then(r => r.json())
    .then(d => {
      if (d.error) {
        document.getElementById('label').textContent = 'Could not read that image';
        document.getElementById('meta').textContent = d.error;
        return;
      }
      const el = document.getElementById('label');
      el.textContent = d.label;
      el.className = 'label ' + d.label.toLowerCase();
      const fill = document.getElementById('fill');
      fill.style.width = (d.confidence * 100).toFixed(1) + '%';
      fill.style.background = d.label === 'Healthy' ? '#2e7d32' : '#c62828';
      document.getElementById('meta').textContent =
        'confidence ' + (d.confidence * 100).toFixed(1) + '%   raw output ' + d.raw.toFixed(6);
    })
    .catch(e => {
      document.getElementById('label').textContent = 'Request failed';
      document.getElementById('meta').textContent = e;
    });
}
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = self.rfile.read(n)

        try:
            img = tf.io.decode_image(data, channels=3, expand_animations=False)
        except Exception as e:
            out = json.dumps({"error": str(e)}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
            return

        img = tf.image.resize(img, [160, 160])
        x = tf.expand_dims(tf.cast(img, tf.float32), 0)
        raw = float(model.predict(x, verbose=0)[0][0])

        label = "Unhealthy" if raw > 0.5 else "Healthy"
        confidence = raw if raw > 0.5 else 1.0 - raw
        print("%s  %.1f%%  (raw %.6f)" % (label, confidence * 100, raw))

        out = json.dumps({"label": label, "confidence": confidence, "raw": raw}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, fmt, *args):
        return


print("Dashboard running at http://localhost:8000")
print("Press Ctrl+C to stop.")
print("")
HTTPServer(("0.0.0.0", 8000), Handler).serve_forever()
