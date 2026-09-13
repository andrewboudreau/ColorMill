/**
 * Full-screen overlays: the loading screen (with progress text) and the
 * "WebGPU unavailable" / fatal error screen with a link to the CPU version.
 */

const LEGACY_URL = 'legacy/index.html';

export class Overlay {
  readonly el: HTMLElement;
  private readonly box: HTMLElement;

  constructor(parent: HTMLElement) {
    this.el = document.createElement('div');
    this.el.className = 'cm-overlay';
    this.el.hidden = true;
    this.box = document.createElement('div');
    this.box.className = 'cm-overlay-box';
    this.el.appendChild(this.box);
    parent.appendChild(this.el);
  }

  showLoading(progress = 'Starting…'): void {
    this.el.hidden = false;
    this.el.dataset.mode = 'loading';
    this.box.innerHTML =
      '<div class="cm-spinner" aria-hidden="true"></div>' +
      '<h1>ColorMill</h1>' +
      '<p class="cm-overlay-sub">WebGPU two-roll mill</p>' +
      '<p class="cm-overlay-progress" aria-live="polite"></p>';
    this.setProgress(progress);
  }

  setProgress(text: string): void {
    const p = this.box.querySelector('.cm-overlay-progress');
    if (p) p.textContent = text;
  }

  /** WebGPU is missing: explain and offer the CPU build. */
  showUnavailable(message: string): void {
    this.showError('WebGPU is not available', message, true);
  }

  /** A fatal startup error (sim/renderer construction failed). */
  showError(title: string, message: string, offerLegacy = true): void {
    this.el.hidden = false;
    this.el.dataset.mode = 'error';
    this.box.innerHTML = '';
    const h = document.createElement('h1');
    h.textContent = title;
    const p = document.createElement('p');
    p.className = 'cm-overlay-message';
    p.textContent = message;
    this.box.append(h, p);
    if (offerLegacy) {
      const a = document.createElement('a');
      a.className = 'cm-btn cm-btn-primary';
      a.href = LEGACY_URL;
      a.textContent = 'Open the CPU version';
      const note = document.createElement('p');
      note.className = 'cm-overlay-sub';
      note.textContent = 'The v1 build runs on a CPU grid and works in any browser.';
      this.box.append(a, note);
    }
  }

  hide(): void {
    this.el.hidden = true;
  }

  get visible(): boolean {
    return !this.el.hidden;
  }

  destroy(): void {
    this.el.remove();
  }
}
