"""Lightweight avatar HTML helper for the Streamlit dashboard."""

from base64 import b64encode
from html import escape
from pathlib import Path
from typing import Optional


def _image_data_uri(image_path: Optional[str]) -> str:
    if not image_path:
        return ""

    path = Path(image_path)
    if not path.is_file():
        return ""

    mime_type = "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"
    encoded = b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_avatar_html(text: str, image_path: Optional[str] = None) -> str:
    """Build a small self-contained speech avatar with idle and speaking states."""
    safe_text = escape(text or "")
    avatar_src = _image_data_uri(image_path)

    image_markup = (
        f'<img id="avatar-image" class="avatar-image" src="{avatar_src}" alt="AI avatar" />'
        if avatar_src
        else (
            '<div id="avatar-image" class="avatar-image avatar-fallback" aria-label="AI avatar">'
            '<div class="face"></div><div class="eye eye-left"></div><div class="eye eye-right"></div>'
            '<div class="mouth"></div></div>'
        )
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Narrator</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
  .shell {{ padding: 14px; }}
  .stage {{ position: relative; min-height: 300px; border-radius: 24px; overflow: hidden; background: radial-gradient(circle at top, #1e293b 0%, #0f172a 60%, #020617 100%); }}
  .halo {{ position: absolute; inset: auto 50% 18px auto; transform: translateX(50%); width: 170px; height: 28px; border-radius: 50%; background: radial-gradient(ellipse, rgba(56,189,248,0.45) 0%, transparent 72%); filter: blur(2px); animation: glow 2.6s ease-in-out infinite; }}
  .avatar-wrap {{ position: absolute; inset: 0; display: grid; place-items: center; }}
  .avatar-image {{ width: min(78%, 240px); max-height: 220px; object-fit: contain; border-radius: 28px; transition: transform 180ms ease, filter 180ms ease, opacity 180ms ease; }}
  .avatar-image.speaking {{ transform: translateY(-3px) scale(1.03); filter: saturate(1.15) drop-shadow(0 0 24px rgba(56,189,248,0.38)); animation: bob 900ms ease-in-out infinite alternate; }}
  .avatar-fallback {{ position: relative; width: 220px; height: 220px; border-radius: 32px; background: linear-gradient(180deg, rgba(15,23,42,0.88), rgba(2,6,23,0.92)); border: 1px solid rgba(148,163,184,0.18); }}
  .face {{ position: absolute; inset: 30px; border-radius: 50%; background: radial-gradient(circle at 40% 35%, #f8fafc 0%, #cbd5e1 30%, #64748b 100%); }}
  .eye {{ position: absolute; top: 98px; width: 14px; height: 18px; border-radius: 50%; background: #0f172a; }}
  .eye-left {{ left: 82px; }}
  .eye-right {{ right: 82px; }}
  .mouth {{ position: absolute; left: 50%; top: 138px; width: 54px; height: 18px; transform: translateX(-50%); border-radius: 0 0 28px 28px; border-bottom: 5px solid #0f172a; opacity: 0.9; }}
  .panel {{ padding: 12px 14px 14px; background: rgba(15,23,42,0.86); border-top: 1px solid rgba(148,163,184,0.14); }}
  .status {{ display: flex; align-items: center; gap: 8px; margin-bottom: 10px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; color: rgba(226,232,240,0.7); }}
  .dot {{ width: 8px; height: 8px; border-radius: 50%; background: #22c55e; box-shadow: 0 0 10px rgba(34,197,94,0.55); }}
  .dot.speaking {{ background: #38bdf8; box-shadow: 0 0 14px rgba(56,189,248,0.7); }}
  .controls {{ display: grid; gap: 8px; }}
  .row {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; }}
  button, select, input[type=range] {{ width: 100%; border-radius: 10px; border: 1px solid rgba(148,163,184,0.18); background: rgba(15,23,42,0.85); color: #e2e8f0; }}
  button {{ padding: 10px 12px; cursor: pointer; }}
  button.primary {{ background: linear-gradient(135deg, #0f766e, #2563eb); border: none; font-weight: 700; }}
  .summary {{ margin-top: 10px; font-size: 13px; line-height: 1.5; color: rgba(226,232,240,0.9); max-height: 92px; overflow: auto; padding-right: 4px; }}
  .bars {{ position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%); display: flex; align-items: flex-end; gap: 4px; height: 32px; z-index: 10; opacity: 0; transition: opacity 0.3s; pointer-events: none; }}
  .bars.speaking {{ opacity: 1; }}
  .bar {{ width: 4px; height: 4px; border-radius: 999px; background: rgba(56,189,248,0.4); transition: height 90ms ease, background 90ms ease; }}
  .bar.active {{ background: rgba(56,189,248,1); box-shadow: 0 0 8px rgba(56,189,248,0.8); }}
  @keyframes bob {{ from {{ transform: translateY(-1px) scale(1.01); }} to {{ transform: translateY(-5px) scale(1.04); }} }}
  @keyframes glow {{ 0%,100% {{ opacity: 0.65; }} 50% {{ opacity: 1; }} }}
  .hidden {{ display: none; }}
</style>
</head>
<body>
  <div class="shell">
    <div class="stage">
      <div class="avatar-wrap">{image_markup}</div>
      <div class="halo"></div>
      <div class="bars" id="bars">{"".join('<div class="bar"></div>' for _ in range(16))}</div>
    </div>
    <div class="panel">
      <div class="status"><span id="status-dot" class="dot"></span><span id="status-text">Idle</span></div>
      <div class="controls">
        <select id="voice"></select>
        <div class="row">
          <input id="rate" type="range" min="0.75" max="1.4" step="0.05" value="1.0" />
          <input id="pitch" type="range" min="0.8" max="1.3" step="0.05" value="1.0" />
          <input id="volume" type="range" min="0.2" max="1.0" step="0.05" value="0.9" />
        </div>
        <div class="row">
          <button class="primary" id="play">Play Narration</button>
          <button id="pause">Pause</button>
          <button id="stop">Stop</button>
        </div>
      </div>
      <div class="summary">{safe_text}</div>
    </div>
  </div>
  <script>
    const summaryText = {text!r};
    const synth = window.speechSynthesis;
    const statusText = document.getElementById('status-text');
    const statusDot = document.getElementById('status-dot');
    const avatar = document.getElementById('avatar-image');
    const bars = Array.from(document.querySelectorAll('.bar'));
    const voiceSelect = document.getElementById('voice');
    const playBtn = document.getElementById('play');
    const pauseBtn = document.getElementById('pause');
    const stopBtn = document.getElementById('stop');
    const rate = document.getElementById('rate');
    const pitch = document.getElementById('pitch');
    const volume = document.getElementById('volume');

    let selectedVoice = null;
    let timer = null;

    function setSpeaking(isSpeaking) {{
      if (avatar) {{
        avatar.classList.toggle('speaking', isSpeaking);
      }}
      statusDot.classList.toggle('speaking', isSpeaking);
      statusText.textContent = isSpeaking ? 'Speaking' : 'Idle';
      
      const barsContainer = document.getElementById('bars');
      if (barsContainer) barsContainer.classList.toggle('speaking', isSpeaking);
      
      if (timer) clearInterval(timer);
      if (isSpeaking) {{
        timer = setInterval(() => {{
          bars.forEach((bar, index) => {{
            const height = 4 + Math.abs(Math.sin(Date.now() / 160 + index)) * 24;
            bar.style.height = `${{height}}px`;
            bar.classList.toggle('active', height > 14);
          }});
        }}, 90);
      }} else {{
        bars.forEach((bar) => {{ bar.style.height = '4px'; bar.classList.remove('active'); }});
      }}
    }}

    function loadVoices() {{
      const voices = synth.getVoices();
      voiceSelect.innerHTML = '';
      voices.forEach((voice, index) => {{
        const option = document.createElement('option');
        option.value = String(index);
        option.textContent = `${{voice.name}} (${{voice.lang}})`;
        voiceSelect.appendChild(option);
      }});
      selectedVoice = voices[0] || null;
    }}

    function speak() {{
      if (!summaryText || !synth) return;
      synth.cancel();
      const utterance = new SpeechSynthesisUtterance(summaryText);
      const voices = synth.getVoices();
      const voice = voices[Number(voiceSelect.value)] || selectedVoice || voices[0] || null;
      if (voice) utterance.voice = voice;
      utterance.rate = Number(rate.value);
      utterance.pitch = Number(pitch.value);
      utterance.volume = Number(volume.value);
      utterance.onstart = () => setSpeaking(true);
      utterance.onend = () => setSpeaking(false);
      utterance.onerror = () => setSpeaking(false);
      synth.speak(utterance);
    }}

    function pauseSpeech() {{
      if (synth.speaking && !synth.paused) {{
        synth.pause();
        statusText.textContent = 'Paused';
      }}
    }}

    function stopSpeech() {{
      synth.cancel();
      setSpeaking(false);
    }}

    playBtn.addEventListener('click', speak);
    pauseBtn.addEventListener('click', pauseSpeech);
    stopBtn.addEventListener('click', stopSpeech);
    
    // Dynamically adjust parameters if changed while speaking
    [rate, pitch, volume].forEach(slider => {{
      slider.addEventListener('change', () => {{
        if (synth.speaking && !synth.paused) {{
          speak();
        }}
      }});
    }});

    if ('onvoiceschanged' in synth) synth.onvoiceschanged = loadVoices;
    loadVoices();
    setSpeaking(false);
  </script>
</body>
</html>"""