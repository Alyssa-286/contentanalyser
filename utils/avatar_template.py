"""
Helper module to generate the HTML component code for the AI Voice Assistant avatar.
Renders a realistic-looking 3D-shaded face using the Canvas 2D API (no CDN needed).
Includes full lip-sync, blinking, idle float animation, and Web Speech API TTS.
"""

import json


def build_avatar_html(text: str) -> str:
    """
    Returns a fully self-contained HTML page (no external CDN dependencies)
    with a Canvas-2D drawn 3D-looking AI face that animates and reads aloud.

    Args:
        text: The summary text to read aloud.

    Returns:
        Complete HTML string ready for st.components.v1.html().
    """
    safe_text = json.dumps(text)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Avatar</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  *{{ margin:0; padding:0; box-sizing:border-box; }}

  body {{
    font-family:'Inter',sans-serif;
    background: linear-gradient(160deg,#0d0b1e 0%,#1a1040 60%,#0d0b1e 100%);
    display:flex; flex-direction:column; align-items:center;
    width:100%; overflow:hidden;
  }}

  /* ── Stage ── */
  #stage {{
    position:relative; width:100%; height:310px;
    display:flex; align-items:center; justify-content:center;
    overflow:hidden;
  }}
  #avatar-canvas {{ width:100%; height:310px; display:block; }}

  /* floating glow ring under face */
  .glow-ring {{
    position:absolute;
    bottom:18px;
    width:160px; height:28px;
    background:radial-gradient(ellipse,rgba(124,58,237,0.55) 0%,transparent 70%);
    border-radius:50%;
    animation:glow-pulse 2.5s ease-in-out infinite;
    pointer-events:none;
  }}
  @keyframes glow-pulse {{ 0%,100%{{opacity:0.6}} 50%{{opacity:1}} }}

  /* ── Panel ── */
  .panel {{
    width:100%;
    background:rgba(255,255,255,0.03);
    border-top:1px solid rgba(255,255,255,0.07);
    padding:10px 14px 14px;
    display:flex; flex-direction:column; gap:9px;
  }}

  /* Status row */
  .status-row{{ display:flex; align-items:center; gap:7px; }}
  .dot{{
    width:7px; height:7px; border-radius:50%;
    background:#4ade80; box-shadow:0 0 6px #4ade80;
    flex-shrink:0; transition:all 0.3s;
  }}
  .dot.speaking{{ background:#a78bfa; box-shadow:0 0 9px #a78bfa;
    animation:pulse-dot 0.7s ease-in-out infinite alternate; }}
  .dot.paused{{ background:#fbbf24; box-shadow:0 0 6px #fbbf24; animation:none; }}
  @keyframes pulse-dot{{ to{{transform:scale(1.7)}} }}
  #status-lbl{{
    font-size:11px; font-weight:600; color:rgba(255,255,255,0.5);
    text-transform:uppercase; letter-spacing:0.6px;
  }}

  /* Wave */
  .wave-row{{
    display:flex; align-items:flex-end; gap:2.5px; height:22px; padding:0 1px;
  }}
  .wb{{
    flex:1; height:3px; border-radius:2px;
    background:rgba(167,139,250,0.2);
    transition:height 0.07s ease,background 0.15s;
  }}

  /* Buttons */
  .btn-row{{ display:flex; gap:7px; }}
  .btn{{
    flex:1; border:none; border-radius:9px;
    padding:8px 6px; font-size:12px; font-weight:600;
    font-family:'Inter',sans-serif; cursor:pointer;
    display:flex; align-items:center; justify-content:center; gap:5px;
    transition:all 0.18s cubic-bezier(0.4,0,0.2,1);
  }}
  .btn-primary{{
    background:linear-gradient(135deg,#7c3aed,#4f46e5);
    color:#fff; box-shadow:0 3px 10px rgba(124,58,237,0.4);
  }}
  .btn-primary:hover{{transform:translateY(-1px);box-shadow:0 5px 16px rgba(124,58,237,0.55);}}
  .btn-sec{{
    background:rgba(255,255,255,0.06); color:rgba(255,255,255,0.75);
    border:1px solid rgba(255,255,255,0.09);
  }}
  .btn-sec:hover{{background:rgba(255,255,255,0.11);}}
  .btn:active{{transform:scale(0.97)!important;}}
  .btn:disabled{{opacity:0.4;cursor:not-allowed;}}

  /* Settings */
  .settings-grid{{
    display:grid; grid-template-columns:1fr 1fr 1fr; gap:8px;
  }}
  .sg{{ display:flex; flex-direction:column; gap:3px; }}
  .sg.full{{grid-column:span 3;}}
  .slabel{{
    font-size:9px; font-weight:700; text-transform:uppercase;
    letter-spacing:0.5px; color:rgba(255,255,255,0.32);
  }}
  select{{
    font-family:'Inter',sans-serif; font-size:10.5px;
    background:rgba(255,255,255,0.06);
    border:1px solid rgba(255,255,255,0.09);
    border-radius:7px; color:rgba(255,255,255,0.8);
    padding:5px 7px; outline:none; cursor:pointer; width:100%;
  }}
  input[type=range]{{
    -webkit-appearance:none; appearance:none;
    width:100%; height:4px; border-radius:2px; border:none; padding:0;
    background:rgba(167,139,250,0.3); cursor:pointer; margin-top:4px;
  }}
  input[type=range]::-webkit-slider-thumb{{
    -webkit-appearance:none; width:13px; height:13px;
    border-radius:50%; background:#a78bfa;
    box-shadow:0 0 5px rgba(167,139,250,0.6); cursor:pointer;
  }}

  .warn{{
    display:none; background:rgba(239,68,68,0.12);
    border:1px solid rgba(239,68,68,0.3); border-radius:8px;
    color:#fca5a5; font-size:11px; text-align:center;
    padding:8px; line-height:1.5;
  }}
</style>
</head>
<body>

<div id="stage">
  <canvas id="avatar-canvas"></canvas>
  <div class="glow-ring"></div>
</div>

<div class="panel">
  <div class="warn" id="warn">⚠️ <b>Speech API not available.</b> Use Chrome, Edge or Safari.</div>

  <div class="status-row">
    <div class="dot idle" id="dot"></div>
    <span id="status-lbl">Ready</span>
  </div>

  <div class="wave-row" id="wave-row">
    {"".join(f'<div class="wb" id="wb{i}"></div>' for i in range(28))}
  </div>

  <div class="btn-row">
    <button class="btn btn-primary" id="btn-play" onclick="playSpeech()">
      <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><polygon points="5 3 19 12 5 21"/></svg>
      Read Aloud
    </button>
    <button class="btn btn-sec" onclick="pauseSpeech()">
      <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/></svg>
      Pause
    </button>
    <button class="btn btn-sec" onclick="stopSpeech()">
      <svg width="11" height="11" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16"/></svg>
      Stop
    </button>
  </div>

  <div class="settings-grid">
    <div class="sg full">
      <div class="slabel">Voice</div>
      <select id="voice-sel" onchange="stopSpeech()">
        <option value="">⏳ Loading voices…</option>
      </select>
    </div>
    <div class="sg"><div class="slabel">Speed</div>
      <input type="range" id="rate-sl" min="0.6" max="1.8" step="0.1" value="1.0" onchange="stopSpeech()">
    </div>
    <div class="sg"><div class="slabel">Pitch</div>
      <input type="range" id="pitch-sl" min="0.7" max="1.5" step="0.1" value="1.0" onchange="stopSpeech()">
    </div>
    <div class="sg"><div class="slabel">Volume</div>
      <input type="range" id="vol-sl" min="0.1" max="1.0" step="0.05" value="0.9">
    </div>
  </div>
</div>

<script>
// ═══════════════════════════════════════════════════════════
// 1. CANVAS SETUP
// ═══════════════════════════════════════════════════════════
const canvas = document.getElementById('avatar-canvas');
const ctx    = canvas.getContext('2d');

function resizeCanvas() {{
  canvas.width  = canvas.offsetWidth  || window.innerWidth  || 460;
  canvas.height = canvas.offsetHeight || 310;
}}
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// ═══════════════════════════════════════════════════════════
// 2. ANIMATION STATE
// ═══════════════════════════════════════════════════════════
let mouthCurrent = 0, mouthTarget = 0;
let blinkState = 'open';   // open | closing | opening
let blinkAmount = 0;       // 0=fully open, 1=fully closed
let blinkTimer  = 200;
let floatOffset = 0;
let isSpeaking  = false;

// waveform bars
const waveBars = Array.from({{length:28}}, (_,i) => document.getElementById('wb'+i));
let waveInterval = null;

// ═══════════════════════════════════════════════════════════
// 3. CANVAS DRAW FUNCTIONS
// ═══════════════════════════════════════════════════════════
function drawScene() {{
  const W  = canvas.width, H = canvas.height;
  const cx = W / 2;
  const cy = H * 0.46 + floatOffset;
  const r  = Math.min(W * 0.23, H * 0.35, 108);  // head radius

  ctx.clearRect(0, 0, W, H);

  // ── Background gradient ──────────────────────────────────
  const bg = ctx.createLinearGradient(0, 0, 0, H);
  bg.addColorStop(0, '#100828');
  bg.addColorStop(1, '#0a0618');
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // ── Star particles ───────────────────────────────────────
  ctx.save();
  for (let i = 0; i < 55; i++) {{
    // deterministic positions from index
    const sx = ((i * 137.508 + 23) % W);
    const sy = ((i * 97.3   + 11) % H);
    const sr = (i % 3 === 0) ? 1.2 : 0.6;
    ctx.beginPath();
    ctx.arc(sx, sy, sr, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(200,190,255,${{0.2 + (i % 5) * 0.07}})`;
    ctx.fill();
  }}
  ctx.restore();

  // ── Body / shoulders ─────────────────────────────────────
  const bodyY = cy + r * 1.05;
  const bodyGrad = ctx.createLinearGradient(cx - r, bodyY, cx + r, bodyY + r * 1.6);
  bodyGrad.addColorStop(0, '#2d4a7a');
  bodyGrad.addColorStop(1, '#1a2d4a');
  ctx.beginPath();
  ctx.ellipse(cx, bodyY + r * 0.5, r * 1.35, r * 1.2, 0, 0, Math.PI * 2);
  ctx.fillStyle = bodyGrad;
  ctx.fill();

  // Shirt collar
  const collarGrad = ctx.createLinearGradient(cx, bodyY - 5, cx, bodyY + 22);
  collarGrad.addColorStop(0, '#f5f2ec');
  collarGrad.addColorStop(1, '#ddd8cc');
  ctx.beginPath();
  ctx.ellipse(cx, bodyY + 4, r * 0.32, 14, 0, 0, Math.PI * 2);
  ctx.fillStyle = collarGrad;
  ctx.fill();

  // ── Neck ─────────────────────────────────────────────────
  const neckGrad = ctx.createLinearGradient(cx - 18, cy + r * 0.8, cx + 18, cy + r * 0.8);
  neckGrad.addColorStop(0, '#c8916e');
  neckGrad.addColorStop(0.5, '#e8b894');
  neckGrad.addColorStop(1, '#c8916e');
  ctx.beginPath();
  ctx.roundRect(cx - 17, cy + r * 0.76, 34, r * 0.38, 8);
  ctx.fillStyle = neckGrad;
  ctx.fill();

  // ── Hair back layer ──────────────────────────────────────
  const hairColor = '#1a0e06';
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy - r * 0.08, r * 1.06, r * 1.12, 0, 0, Math.PI * 2);
  ctx.fillStyle = hairColor;
  ctx.fill();
  // Side hair panels
  ctx.beginPath();
  ctx.ellipse(cx - r * 0.9, cy + r * 0.1, r * 0.42, r * 0.75, -0.18, 0, Math.PI * 2);
  ctx.fillStyle = hairColor;
  ctx.fill();
  ctx.beginPath();
  ctx.ellipse(cx + r * 0.9, cy + r * 0.1, r * 0.42, r * 0.75,  0.18, 0, Math.PI * 2);
  ctx.fillStyle = hairColor;
  ctx.fill();
  ctx.restore();

  // ── Head (3D shaded sphere) ───────────────────────────────
  // Ambient shadow under head
  const shadowGrad = ctx.createRadialGradient(cx, cy + r * 0.82, 0, cx, cy + r * 0.82, r * 0.6);
  shadowGrad.addColorStop(0, 'rgba(0,0,0,0.35)');
  shadowGrad.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.beginPath();
  ctx.ellipse(cx, cy + r * 0.82, r * 0.6, r * 0.18, 0, 0, Math.PI * 2);
  ctx.fillStyle = shadowGrad;
  ctx.fill();

  // Face sphere with 3D radial gradient (light from top-left)
  const faceGrad = ctx.createRadialGradient(
    cx - r * 0.22, cy - r * 0.2, r * 0.04,   // light hotspot
    cx + r * 0.05, cy + r * 0.08, r * 1.05   // shadow edge
  );
  faceGrad.addColorStop(0.00, '#fde8c8');  // bright highlight
  faceGrad.addColorStop(0.28, '#f0c8a0');  // mid lit
  faceGrad.addColorStop(0.60, '#e0a878');  // skin tone
  faceGrad.addColorStop(0.85, '#c47850');  // shadow
  faceGrad.addColorStop(1.00, '#a05830');  // deep shadow rim
  ctx.beginPath();
  ctx.ellipse(cx, cy, r, r * 1.06, 0, 0, Math.PI * 2);
  ctx.fillStyle = faceGrad;
  ctx.fill();

  // ── Ears ─────────────────────────────────────────────────
  [[-1, 1]].forEach(() => {{
    for (const side of [-1, 1]) {{
      const ex = cx + side * r * 0.97;
      const earGrad = ctx.createRadialGradient(ex - side*4, cy - 3, 2, ex, cy, 14);
      earGrad.addColorStop(0, '#f0c8a0');
      earGrad.addColorStop(1, '#c47850');
      ctx.beginPath();
      ctx.ellipse(ex, cy + r * 0.05, 10, 16, side * 0.15, 0, Math.PI * 2);
      ctx.fillStyle = earGrad;
      ctx.fill();
      // Inner ear
      ctx.beginPath();
      ctx.ellipse(ex, cy + r * 0.05, 5, 9, side * 0.15, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(170,80,50,0.25)';
      ctx.fill();
    }}
  }});

  // ── Hair front cap ───────────────────────────────────────
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(cx, cy - r * 0.45, r * 1.02, r * 0.68, 0, Math.PI, Math.PI * 2);
  ctx.fillStyle = hairColor;
  ctx.fill();
  // Fringe
  ctx.beginPath();
  ctx.ellipse(cx, cy - r * 0.5, r * 0.82, r * 0.3, 0, 0, Math.PI);
  ctx.fillStyle = '#120a04';
  ctx.fill();
  ctx.restore();

  // ── Eyebrows ─────────────────────────────────────────────
  const browY = cy - r * 0.22;
  const browRaise = isSpeaking ? Math.sin(Date.now() / 320) * 2.5 : 0;
  ctx.save();
  ctx.strokeStyle = '#1a0e06';
  ctx.lineWidth   = 4.5;
  ctx.lineCap     = 'round';
  for (const side of [-1, 1]) {{
    const bx = cx + side * r * 0.34;
    ctx.beginPath();
    ctx.moveTo(bx - side * 22, browY - browRaise + (side === 1 ? 2 : 2));
    ctx.quadraticCurveTo(bx, browY - browRaise - 5, bx + side * 22, browY - browRaise + (side === 1 ? 4 : 4));
    ctx.stroke();
  }}
  ctx.restore();

  // ── Eyes ─────────────────────────────────────────────────
  const eyeY   = cy - r * 0.12;
  const eyeSpX = r * 0.34;
  const eyeRx  = r * 0.175, eyeRy = r * 0.135;

  for (const side of [-1, 1]) {{
    const ex = cx + side * eyeSpX;

    // Sclera (white)
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(ex, eyeY, eyeRx, eyeRy, 0, 0, Math.PI * 2);
    ctx.fillStyle = '#f8f4ef';
    ctx.shadowColor = 'rgba(0,0,0,0.3)';
    ctx.shadowBlur  = 4;
    ctx.fill();
    ctx.restore();

    // Iris
    const irisR = eyeRy * 0.72;
    const irisGrad = ctx.createRadialGradient(ex - irisR*0.3, eyeY - irisR*0.2, irisR*0.05, ex, eyeY, irisR);
    irisGrad.addColorStop(0.0, '#6688dd');
    irisGrad.addColorStop(0.5, '#2244aa');
    irisGrad.addColorStop(1.0, '#0d1a55');
    ctx.beginPath();
    ctx.arc(ex, eyeY, irisR, 0, Math.PI * 2);
    ctx.fillStyle = irisGrad;
    ctx.fill();

    // Pupil
    const pupilR = irisR * 0.46;
    ctx.beginPath();
    ctx.arc(ex, eyeY, pupilR, 0, Math.PI * 2);
    ctx.fillStyle = '#050510';
    ctx.fill();

    // Catchlight
    ctx.beginPath();
    ctx.arc(ex - irisR * 0.28, eyeY - irisR * 0.28, pupilR * 0.38, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.fill();

    // Eyelid blink (top lid comes down)
    if (blinkAmount > 0) {{
      ctx.save();
      ctx.beginPath();
      ctx.ellipse(ex, eyeY, eyeRx * 1.04, eyeRy * 1.04, 0, 0, Math.PI * 2);
      ctx.clip();
      // Top lid
      const lidH = eyeRy * 2.1 * blinkAmount;
      ctx.fillStyle = faceGrad; // match skin
      ctx.fillRect(ex - eyeRx * 1.1, eyeY - eyeRy * 1.1, eyeRx * 2.2, lidH + eyeRy * 0.1);
      ctx.restore();
    }}

    // Upper lash line
    ctx.save();
    ctx.beginPath();
    ctx.ellipse(ex, eyeY, eyeRx, eyeRy * 0.92, 0, Math.PI, Math.PI * 2);
    ctx.strokeStyle = '#111';
    ctx.lineWidth   = 2.5 - blinkAmount * 1.5;
    ctx.stroke();
    ctx.restore();

    // Clipping sclera border
    ctx.beginPath();
    ctx.ellipse(ex, eyeY, eyeRx, eyeRy, 0, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(150,100,60,0.25)';
    ctx.lineWidth   = 1;
    ctx.stroke();
  }}

  // ── Nose ─────────────────────────────────────────────────
  const noseX = cx, noseY = cy + r * 0.12;
  // Bridge shadow
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(noseX - 3, noseY - r * 0.22);
  ctx.quadraticCurveTo(noseX - 9, noseY, noseX - 11, noseY + r * 0.08);
  ctx.strokeStyle = 'rgba(140,75,40,0.25)';
  ctx.lineWidth   = 2.5;
  ctx.lineCap     = 'round';
  ctx.stroke();
  // Tip
  ctx.beginPath();
  ctx.arc(noseX, noseY + r * 0.06, r * 0.09, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(180,100,60,0.18)';
  ctx.fill();
  // Nostrils
  for (const s of [-1, 1]) {{
    ctx.beginPath();
    ctx.ellipse(noseX + s * r * 0.09, noseY + r * 0.1, 5, 3.5, s * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(120,60,30,0.3)';
    ctx.fill();
  }}
  ctx.restore();

  // ── Mouth ─────────────────────────────────────────────────
  const mouthY  = cy + r * 0.42;
  const mouthW  = r * 0.46;
  const openAmt = mouthCurrent;   // 0..1
  const lipH    = r * 0.065 + openAmt * r * 0.07;

  // Mouth cavity (dark opening)
  if (openAmt > 0.05) {{
    const cavH = openAmt * r * 0.22;
    const cavGrad = ctx.createRadialGradient(cx, mouthY, 0, cx, mouthY, mouthW * 0.8);
    cavGrad.addColorStop(0, '#1a0505');
    cavGrad.addColorStop(1, '#2a0808');
    ctx.beginPath();
    ctx.ellipse(cx, mouthY + lipH * 0.3, mouthW * 0.75, cavH, 0, 0, Math.PI * 2);
    ctx.fillStyle = cavGrad;
    ctx.fill();

    // Upper teeth
    if (openAmt > 0.2) {{
      ctx.beginPath();
      ctx.roundRect(cx - mouthW * 0.55, mouthY - lipH * 0.1, mouthW * 1.1, cavH * 0.38, 3);
      ctx.fillStyle = '#f2ede6';
      ctx.fill();
    }}
  }}

  // Upper lip
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(cx - mouthW, mouthY - openAmt * r * 0.035);
  // Cupid's bow
  ctx.bezierCurveTo(
    cx - mouthW * 0.5, mouthY - lipH * 1.1 - openAmt * r * 0.02,
    cx - mouthW * 0.18, mouthY - lipH * 0.7 - openAmt * r * 0.02,
    cx, mouthY - lipH * 0.9 - openAmt * r * 0.02
  );
  ctx.bezierCurveTo(
    cx + mouthW * 0.18, mouthY - lipH * 0.7 - openAmt * r * 0.02,
    cx + mouthW * 0.5, mouthY - lipH * 1.1 - openAmt * r * 0.02,
    cx + mouthW, mouthY - openAmt * r * 0.035
  );
  ctx.bezierCurveTo(
    cx + mouthW * 0.5, mouthY + lipH * 0.2,
    cx - mouthW * 0.5, mouthY + lipH * 0.2,
    cx - mouthW, mouthY - openAmt * r * 0.035
  );
  ctx.closePath();
  const lipGrad = ctx.createLinearGradient(cx, mouthY - lipH, cx, mouthY + lipH);
  lipGrad.addColorStop(0, '#d47070');
  lipGrad.addColorStop(0.5, '#c05858');
  lipGrad.addColorStop(1, '#a83838');
  ctx.fillStyle = lipGrad;
  ctx.fill();

  // Lower lip
  ctx.beginPath();
  ctx.moveTo(cx - mouthW, mouthY + openAmt * r * 0.09);
  ctx.bezierCurveTo(
    cx - mouthW * 0.4, mouthY + lipH * 1.5 + openAmt * r * 0.07,
    cx + mouthW * 0.4, mouthY + lipH * 1.5 + openAmt * r * 0.07,
    cx + mouthW, mouthY + openAmt * r * 0.09
  );
  ctx.bezierCurveTo(
    cx + mouthW * 0.35, mouthY + lipH * 0.7 + openAmt * r * 0.05,
    cx - mouthW * 0.35, mouthY + lipH * 0.7 + openAmt * r * 0.05,
    cx - mouthW, mouthY + openAmt * r * 0.09
  );
  ctx.closePath();
  const lowerLipGrad = ctx.createLinearGradient(cx, mouthY, cx, mouthY + lipH * 1.5);
  lowerLipGrad.addColorStop(0, '#d47070');
  lowerLipGrad.addColorStop(1, '#b04040');
  ctx.fillStyle = lowerLipGrad;
  ctx.fill();
  // Lower lip specular
  ctx.beginPath();
  ctx.ellipse(cx, mouthY + lipH * 1.0 + openAmt * r * 0.06, mouthW * 0.28, lipH * 0.3, 0, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(255,200,180,0.28)';
  ctx.fill();
  ctx.restore();

  // ── Specular highlight on forehead ────────────────────────
  const specGrad = ctx.createRadialGradient(
    cx - r * 0.24, cy - r * 0.34, 0,
    cx - r * 0.24, cy - r * 0.34, r * 0.42
  );
  specGrad.addColorStop(0, 'rgba(255,255,255,0.22)');
  specGrad.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.beginPath();
  ctx.ellipse(cx - r * 0.24, cy - r * 0.34, r * 0.42, r * 0.3, -0.3, 0, Math.PI * 2);
  ctx.fillStyle = specGrad;
  ctx.fill();

  // ── Name label ───────────────────────────────────────────
  ctx.save();
  ctx.font = `600 11px Inter,sans-serif`;
  ctx.textAlign = 'center';
  ctx.fillStyle = 'rgba(200,180,255,0.5)';
  ctx.fillText('AI NARRATOR', cx, H - 10);
  ctx.restore();
}}

// ═══════════════════════════════════════════════════════════
// 4. MAIN ANIMATION LOOP
// ═══════════════════════════════════════════════════════════
let lastTime = 0;

function loop(ts) {{
  const dt = (ts - lastTime) / 1000;  // seconds
  lastTime = ts;

  // Floating bob
  floatOffset = Math.sin(ts / 1800) * 6;

  // Smooth mouth
  mouthCurrent += (mouthTarget - mouthCurrent) * 0.22;

  // Blink
  blinkTimer -= dt;
  if (blinkTimer <= 0 && blinkState === 'open') {{
    blinkState = 'closing';
    blinkTimer = 0.06;
  }}
  if (blinkState === 'closing') {{
    blinkAmount = Math.min(1, blinkAmount + dt * 18);
    if (blinkAmount >= 1) {{ blinkState = 'opening'; }}
  }}
  if (blinkState === 'opening') {{
    blinkAmount = Math.max(0, blinkAmount - dt * 14);
    if (blinkAmount <= 0) {{ blinkState = 'open'; blinkTimer = 2.5 + Math.random() * 3.5; }}
  }}

  drawScene();
  requestAnimationFrame(loop);
}}
requestAnimationFrame(loop);

// ═══════════════════════════════════════════════════════════
// 5. VOICE LOADING (retry loop – works in iframes)
// ═══════════════════════════════════════════════════════════
let voices = [];
const voiceSel = document.getElementById('voice-sel');

function populateVoices() {{
  const all = window.speechSynthesis ? window.speechSynthesis.getVoices() : [];
  if (!all.length) return false;

  voices = all;
  const en = all.filter(v => v.lang.startsWith('en'));
  const list = en.length ? en : all;

  voiceSel.innerHTML = '';
  list.forEach((v, i) => {{
    const opt = document.createElement('option');
    opt.value = v.name;
    opt.textContent = v.name + ' (' + v.lang + ')';
    const preferred = ['Google US English','Samantha','Microsoft Zira','Microsoft David'];
    if (preferred.some(p => v.name.includes(p)) || i === 0) opt.selected = true;
    voiceSel.appendChild(opt);
  }});
  return true;
}}

if (!('speechSynthesis' in window)) {{
  document.getElementById('warn').style.display = 'block';
  document.getElementById('btn-play').disabled  = true;
}} else {{
  // Try immediately, then retry every 200 ms for up to 4 seconds
  if (!populateVoices()) {{
    window.speechSynthesis.onvoiceschanged = populateVoices;
    let retries = 20;
    const retryTimer = setInterval(() => {{
      if (populateVoices() || --retries <= 0) clearInterval(retryTimer);
    }}, 200);
  }}
}}

// ═══════════════════════════════════════════════════════════
// 6. VISEME → MOUTH SHAPE
// ═══════════════════════════════════════════════════════════
let visemeTimer = null;

function getViseme(ch) {{
  const c = ch.toLowerCase();
  if ('aeiou'.includes(c))   return {{ open: 0.85 }};
  if ('ou'.includes(c))      return {{ open: 0.65 }};
  if ('bmp'.includes(c))     return {{ open: 0.05 }};
  if ('fv'.includes(c))      return {{ open: 0.12 }};
  if ('szxc'.includes(c))    return {{ open: 0.30 }};
  if ('tdnlr'.includes(c))   return {{ open: 0.45 }};
  if ('ghky'.includes(c))    return {{ open: 0.38 }};
  if (c === ' ')             return {{ open: 0.0  }};
  return                            {{ open: 0.22 }};
}}

function animateWord(word) {{
  if (visemeTimer) clearInterval(visemeTimer);
  const chars = word.replace(/[^a-zA-Z ]/g, '').split('');
  if (!chars.length) return;
  const rate = parseFloat(document.getElementById('rate-sl').value) || 1.0;
  const interval = Math.max(42, 95 / rate);
  let i = 0;
  visemeTimer = setInterval(() => {{
    if (!isSpeaking || i >= chars.length) {{
      clearInterval(visemeTimer);
      mouthTarget = 0;
      return;
    }}
    mouthTarget = getViseme(chars[i]).open;
    i++;
  }}, interval);
}}

// ═══════════════════════════════════════════════════════════
// 7. WAVEFORM ANIMATION
// ═══════════════════════════════════════════════════════════
function animateWave(active) {{
  waveBars.forEach((b, i) => {{
    if (active) {{
      const h = 3 + Math.random() * 18;
      b.style.height     = h + 'px';
      b.style.background = `rgba(${{148 + (Math.random()*55|0)}},139,250,${{0.4 + Math.random()*0.55}})`;
    }} else {{
      b.style.height     = '3px';
      b.style.background = 'rgba(167,139,250,0.18)';
    }}
  }});
}}

// ═══════════════════════════════════════════════════════════
// 8. SPEECH PLAYBACK CONTROLS
// ═══════════════════════════════════════════════════════════
const textToSpeak = {safe_text};
const dotEl  = document.getElementById('dot');
const lblEl  = document.getElementById('status-lbl');
let utterance = null, paused = false;

function setStatus(s) {{
  dotEl.className = 'dot ' + s;
  lblEl.textContent = s === 'speaking' ? 'Speaking…' : s === 'paused' ? 'Paused' : 'Ready';
}}

function playSpeech() {{
  if (!('speechSynthesis' in window)) return;
  if (paused) {{
    window.speechSynthesis.resume();
    isSpeaking = true; paused = false;
    setStatus('speaking');
    waveInterval = setInterval(() => animateWave(true), 80);
    return;
  }}
  window.speechSynthesis.cancel();
  utterance = new SpeechSynthesisUtterance(textToSpeak);
  const sv = voices.find(v => v.name === voiceSel.value);
  if (sv) utterance.voice = sv;
  utterance.rate   = parseFloat(document.getElementById('rate-sl').value)  || 1.0;
  utterance.pitch  = parseFloat(document.getElementById('pitch-sl').value) || 1.0;
  utterance.volume = parseFloat(document.getElementById('vol-sl').value)   || 0.9;
  utterance.onstart = () => {{
    isSpeaking = true; paused = false;
    setStatus('speaking');
    waveInterval = setInterval(() => animateWave(true), 80);
  }};
  utterance.onend = utterance.onerror = () => {{
    isSpeaking = false; paused = false;
    mouthTarget = 0;
    setStatus('idle');
    clearInterval(waveInterval);
    animateWave(false);
  }};
  utterance.onboundary = e => {{
    if (e.name === 'word') {{
      const word = textToSpeak.substring(e.charIndex, e.charIndex + (e.charLength || 5));
      animateWord(word);
    }}
  }};
  window.speechSynthesis.speak(utterance);
}}

function pauseSpeech() {{
  if (isSpeaking && !paused) {{
    window.speechSynthesis.pause();
    isSpeaking = false; paused = true;
    mouthTarget = 0;
    setStatus('paused');
    clearInterval(waveInterval);
    animateWave(false);
  }}
}}

function stopSpeech() {{
  window.speechSynthesis.cancel();
  isSpeaking = false; paused = false;
  mouthTarget = 0;
  setStatus('idle');
  clearInterval(waveInterval);
  animateWave(false);
  if (visemeTimer) clearInterval(visemeTimer);
}}

window.addEventListener('beforeunload', stopSpeech);
</script>
</body>
</html>"""
