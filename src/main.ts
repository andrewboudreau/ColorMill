import { toCssRgb } from './color';
import { createInitialMillState, updateMill, type MillState } from './mill';
import './styles.css';

const app = document.querySelector<HTMLDivElement>('#app');

if (!app) {
  throw new Error('App root element was not found.');
}

app.innerHTML = `
  <main class="shell">
    <section class="hero">
      <p class="eyebrow">Two-roll color mill simulator</p>
      <h1>ColorMill</h1>
      <p>
        A tiny browser prototype for studying how colored material stretches, folds,
        cuts, and blends between mill rollers.
      </p>
    </section>
    <canvas id="mill-canvas" width="900" height="540" aria-label="ColorMill simulation canvas"></canvas>
    <section class="notes">
      <article>
        <h2>Current model</h2>
        <p>Gradient strips ride between counter-rotating rollers. This is the visual seed for later fold/cut mechanics.</p>
      </article>
      <article>
        <h2>Next mechanics</h2>
        <p>Add strip particles, roller compression, blade cuts, and fold-over operations.</p>
      </article>
    </section>
  </main>
`;

const canvas = document.querySelector<HTMLCanvasElement>('#mill-canvas');
const context = canvas?.getContext('2d');

if (!canvas || !context) {
  throw new Error('Canvas rendering context was not available.');
}

let state = createInitialMillState();
let previousTime = performance.now();

function addRoundedRectPath(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number): void {
  const r = Math.min(radius, width / 2, height / 2);
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + width - r, y);
  ctx.quadraticCurveTo(x + width, y, x + width, y + r);
  ctx.lineTo(x + width, y + height - r);
  ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
  ctx.lineTo(x + r, y + height);
  ctx.quadraticCurveTo(x, y + height, x, y + height - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
}

function drawRoller(ctx: CanvasRenderingContext2D, x: number, y: number, radius: number, angle: number, label: string): void {
  ctx.save();
  ctx.translate(x, y);

  const gradient = ctx.createRadialGradient(-radius * 0.35, -radius * 0.35, radius * 0.1, 0, 0, radius);
  gradient.addColorStop(0, '#f7fafc');
  gradient.addColorStop(0.45, '#a0aec0');
  gradient.addColorStop(1, '#2d3748');

  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(0, 0, radius, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = '#1a202c';
  ctx.lineWidth = 4;
  ctx.stroke();

  ctx.rotate(angle);
  ctx.strokeStyle = '#edf2f7';
  ctx.lineWidth = 3;
  for (let i = 0; i < 6; i++) {
    ctx.rotate(Math.PI / 3);
    ctx.beginPath();
    ctx.moveTo(-radius + 16, 0);
    ctx.lineTo(radius - 16, 0);
    ctx.stroke();
  }

  ctx.rotate(-angle);
  ctx.fillStyle = '#1a202c';
  ctx.font = '600 18px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText(label, 0, radius + 34);
  ctx.restore();
}

function drawMaterial(ctx: CanvasRenderingContext2D, millState: MillState): void {
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';

  for (const strip of millState.strips) {
    ctx.fillStyle = toCssRgb(strip.color);
    ctx.beginPath();
    addRoundedRectPath(ctx, strip.x, strip.y, strip.width, strip.height, 9);
    ctx.fill();
  }

  ctx.restore();
}

function draw(ctx: CanvasRenderingContext2D, millState: MillState): void {
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  ctx.fillStyle = '#e2e8f0';
  ctx.fillRect(86, 378, 728, 24);

  drawRoller(ctx, 330, 220, 112, millState.leftRollerAngle, 'Left roller');
  drawRoller(ctx, 570, 220, 112, millState.rightRollerAngle, 'Right roller');
  drawMaterial(ctx, millState);

  ctx.fillStyle = '#334155';
  ctx.font = '600 20px system-ui';
  ctx.textAlign = 'center';
  ctx.fillText('Nip gap / mixing zone', 450, 82);

  ctx.strokeStyle = '#64748b';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);
  ctx.beginPath();
  ctx.moveTo(450, 96);
  ctx.lineTo(450, 315);
  ctx.stroke();
  ctx.setLineDash([]);
}

function tick(now: number): void {
  const elapsedSeconds = (now - previousTime) / 1000;
  previousTime = now;
  state = updateMill(state, elapsedSeconds);
  draw(context, state);
  requestAnimationFrame(tick);
}

requestAnimationFrame(tick);
