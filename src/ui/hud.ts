/**
 * Top-left heads-up display: title, one status line (preset, particles, grid,
 * fps, ms/frame, sim speed), an error line, and a hint that fades out.
 */

export interface HudStatus {
  readonly preset: string;
  readonly particles: number;
  /** e.g. "97×81×97" */
  readonly grid: string;
  readonly fps: number;
  readonly msFrame: number;
  /** sim seconds per real second */
  readonly simSpeed: number;
  readonly paused: boolean;
  /** material on the mill, litres (at the assumed physical scale) */
  readonly litres: number;
  readonly kg: number;
}

/** 484096 -> "484k", 1234 -> "1.2k", 512 -> "512" */
export function formatCount(n: number): string {
  if (!Number.isFinite(n)) return '–';
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`;
  if (n >= 1e4) return `${Math.round(n / 1e3)}k`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(Math.round(n));
}

export class Hud {
  readonly el: HTMLElement;
  private readonly statusEl: HTMLElement;
  private readonly errorEl: HTMLElement;
  private readonly hintEl: HTMLElement;
  private hintTimer: number | undefined;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'cm-hud';
    this.el.innerHTML =
      '<h1 class="cm-hud-title">ColorMill</h1>' +
      '<div class="cm-hud-status" aria-live="off">starting…</div>' +
      '<div class="cm-hud-error" role="alert" hidden></div>' +
      '<div class="cm-hud-hint" hidden></div>';
    this.statusEl = this.el.querySelector('.cm-hud-status') as HTMLElement;
    this.errorEl = this.el.querySelector('.cm-hud-error') as HTMLElement;
    this.hintEl = this.el.querySelector('.cm-hud-hint') as HTMLElement;
    parent.appendChild(this.el);
  }

  /** Free-form status text (used while loading / rebuilding). */
  setText(text: string): void {
    this.statusEl.textContent = text;
  }

  setStatus(s: HudStatus): void {
    const fps = s.fps > 0 ? s.fps.toFixed(0) : '–';
    const ms = s.msFrame > 0 ? s.msFrame.toFixed(1) : '–';
    const speed = s.paused ? 'paused' : `${s.simSpeed.toFixed(2)}× real time`;
    this.statusEl.textContent =
      `${s.preset} · ${s.litres.toFixed(2)} L (${s.kg.toFixed(2)} kg) in ${formatCount(s.particles)} particles · ${s.grid} · ${fps} fps · ${ms} ms · ${speed}`;
  }

  /** Show (or clear with null) a persistent error line. */
  setError(message: string | null): void {
    if (message) {
      this.errorEl.textContent = message;
      this.errorEl.hidden = false;
    } else {
      this.errorEl.textContent = '';
      this.errorEl.hidden = true;
    }
  }

  /** Show a one-line hint that fades out after `ms` (0 = stays). */
  showHint(text: string, ms = 7000): void {
    if (this.hintTimer !== undefined) window.clearTimeout(this.hintTimer);
    this.hintEl.textContent = text;
    this.hintEl.hidden = false;
    this.hintEl.classList.remove('is-fading');
    if (ms > 0) {
      this.hintTimer = window.setTimeout(() => {
        this.hintEl.classList.add('is-fading');
        this.hintTimer = window.setTimeout(() => { this.hintEl.hidden = true; }, 1200);
      }, ms);
    }
  }

  destroy(): void {
    if (this.hintTimer !== undefined) window.clearTimeout(this.hintTimer);
    this.el.remove();
  }
}
