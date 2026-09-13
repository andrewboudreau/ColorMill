/**
 * Bottom bar: large round pigment swatches (tap = inject), a custom colour
 * swatch backed by <input type="color">, and the action buttons
 * (Cut & fold, Clear pigment, Reset, Pause/Play).
 */
import { PALETTE_ORDER, PIGMENTS } from '../color/pigments';

export interface PaletteCallbacks {
  onPigment(key: string): void;
  /** the picker closed on a new colour ('#rrggbb') */
  onCustom(hex: string): void;
  onCutFold(): void;
  onClear(): void;
  onReset(): void;
  onTogglePause(): void;
}

export class Palette {
  readonly el: HTMLElement;
  readonly customInput: HTMLInputElement;
  private readonly pauseButton: HTMLButtonElement;
  private readonly foldButton: HTMLButtonElement;
  private readonly swatches = new Map<string, HTMLButtonElement>();

  constructor(parent: HTMLElement, private readonly cb: PaletteCallbacks, customHex = '#ff8a00') {
    this.el = document.createElement('div');
    this.el.className = 'cm-palette';
    this.el.setAttribute('role', 'toolbar');
    this.el.setAttribute('aria-label', 'Pigments and actions');

    const swatches = document.createElement('div');
    swatches.className = 'cm-swatches';
    PALETTE_ORDER.forEach((key, i) => {
      const p = PIGMENTS[key];
      if (!p) return;
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'cm-swatch';
      b.style.setProperty('--c', p.hex);
      b.dataset.pigment = key;
      b.title = i < 8 ? `${p.name} (${i + 1})` : p.name;
      b.setAttribute('aria-label', `Add ${p.name}`);
      if (i < 8) {
        const k = document.createElement('span');
        k.className = 'cm-swatch-key';
        k.textContent = String(i + 1);
        b.appendChild(k);
      }
      b.addEventListener('click', () => {
        this.flash(b);
        this.cb.onPigment(key);
      });
      swatches.appendChild(b);
      this.swatches.set(key, b);
    });

    // custom colour: the native picker sits invisibly over a round swatch
    const custom = document.createElement('label');
    custom.className = 'cm-swatch cm-swatch-custom';
    custom.title = 'Custom colour';
    custom.style.setProperty('--c', customHex);
    this.customInput = document.createElement('input');
    this.customInput.type = 'color';
    this.customInput.value = customHex;
    this.customInput.setAttribute('aria-label', 'Custom pigment colour');
    this.customInput.addEventListener('input', () => custom.style.setProperty('--c', this.customInput.value));
    this.customInput.addEventListener('change', () => {
      custom.style.setProperty('--c', this.customInput.value);
      this.flash(custom);
      this.cb.onCustom(this.customInput.value);
    });
    const plus = document.createElement('span');
    plus.className = 'cm-swatch-plus';
    plus.setAttribute('aria-hidden', 'true');
    plus.textContent = '+';
    custom.append(this.customInput, plus);
    swatches.appendChild(custom);

    const actions = document.createElement('div');
    actions.className = 'cm-actions';
    this.foldButton = this.button(actions, 'Cut & fold', 'F', () => this.cb.onCutFold());
    this.button(actions, 'Clear pigment', undefined, () => this.cb.onClear());
    this.button(actions, 'Reset', 'R', () => this.cb.onReset());
    this.pauseButton = this.button(actions, 'Pause', 'Space', () => this.cb.onTogglePause());
    this.pauseButton.classList.add('cm-btn-primary');
    this.pauseButton.setAttribute('aria-pressed', 'false');

    this.el.append(swatches, actions);
    parent.appendChild(this.el);
  }

  private button(parent: HTMLElement, label: string, key: string | undefined, onClick: () => void): HTMLButtonElement {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'cm-btn';
    b.textContent = label;
    if (key) b.title = `${label} (${key})`;
    b.addEventListener('click', onClick);
    parent.appendChild(b);
    return b;
  }

  /** Short press animation, also used when a keyboard shortcut fires. */
  flash(el: HTMLElement): void {
    el.classList.remove('is-flash');
    // restart the animation
    void el.offsetWidth;
    el.classList.add('is-flash');
  }

  flashPigment(key: string): void {
    const b = this.swatches.get(key);
    if (b) this.flash(b);
  }

  setPaused(paused: boolean): void {
    this.pauseButton.textContent = paused ? 'Play' : 'Pause';
    this.pauseButton.setAttribute('aria-pressed', String(paused));
    this.pauseButton.title = `${paused ? 'Play' : 'Pause'} (Space)`;
  }

  setFoldBusy(busy: boolean): void {
    this.foldButton.disabled = busy;
  }

  /** Pigment keys in on-screen order (for the 1–8 shortcuts). */
  get order(): readonly string[] {
    return PALETTE_ORDER;
  }

  destroy(): void {
    this.el.remove();
  }
}
