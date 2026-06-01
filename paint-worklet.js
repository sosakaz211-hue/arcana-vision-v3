// CSS Houdini Paint Worklet — Arcana Noise Pattern
// Registers a custom painter that renders procedural noise in the ARCANA brand palette

class ArcanaNoisePainter {
  static get inputProperties() {
    return ['--arcana-noise-seed', '--arcana-noise-density', '--arcana-color-void', '--arcana-color-matrix'];
  }

  paint(ctx, size, props) {
    const seed = parseFloat(props.get('--arcana-noise-seed')) || 0;
    const density = parseFloat(props.get('--arcana-noise-density')) || 0.12;
    const w = size.width;
    const h = size.height;

    // Fill void
    ctx.fillStyle = '#0a0a0f';
    ctx.fillRect(0, 0, w, h);

    // Seeded pseudo-random
    function rng() {
      const x = Math.sin(seed * 12.9898 + (state *= 78.233)) * 43758.5453;
      return x - Math.floor(x);
    }
    let state = 1;

    const count = Math.floor(w * h * density * 0.01);
    for (let i = 0; i < count; i++) {
      const x = rng() * w;
      const y = rng() * h;
      const r = rng() * 1.2 + 0.3;
      const a = rng() * 0.25 + 0.05;
      const hue = rng() > 0.7 ? '#00ffa3' : rng() > 0.4 ? '#b28cff' : '#6f9cff';
      ctx.globalAlpha = a;
      ctx.fillStyle = hue;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Subtle grid lines
    ctx.globalAlpha = 0.03;
    ctx.strokeStyle = '#00ffa3';
    ctx.lineWidth = 0.5;
    const step = 32;
    for (let x = 0; x < w; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 0; y < h; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }
  }
}

registerPaint('arcana-noise', ArcanaNoisePainter);
