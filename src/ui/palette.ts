/**
 * Bottom bar: the drop-slot strip (where along the roll the next tap lands),
 * large round pigment swatches (tap = inject), a custom colour swatch backed
 * by <input type="color">, and the action buttons (Cut & roll, Clear pigment,
 * Reset, Pause/Play).
 */
import { PALETTE_ORDER, PIGMENTS } from '../color/pigments';
import { DROP_SLOTS } from '../config/mill';

export interface PaletteCallbacks {
  onPigment(key: string): void;
  /** the operator picked a drop slot (0-based, left to right along the roll) */
  onSlot(slot: number): void;
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
  private readonly slotButtons: HTMLButtonElement[] = [];
  private slot = 0;

  constructor(parent: HTMLElement, private readonly cb: PaletteCallbacks, customHex = '#ff8a00', slot = 0) {
    this.el = document.createElement('div');
    this.el.className = 'cm-palette';
    this.el.setAttribute('role', 'toolbar');
    this.el.setAttribute('aria-label', 'Pigments and actions');

    // drop slots: a strip of positions along the roll, left to right as seen
    // from the front; the chosen one is where the next tap sets its chunk down
    const slots = document.createElement('div');
    slots.className = 'cm-slots';
    slots.setAttribute('role', 'radiogroup');
    slots.setAttribute('aria-label', 'Pigment drop position along the roll');
    slots.title = 'Where the next pigment lands, left to right along the roll ([ and ])';
    for (let i = 0; i < DROP_SLOTS; i++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'cm-slot';
      b.dataset.slot = String(i);
      b.setAttribute('role', 'radio');
      b.setAttribute('aria-label', `Drop slot ${i + 1} of ${DROP_SLOTS}`);
      b.addEventListener('click', () => { this.setSlot(i); this.cb.onSlot(i); b.blur(); });
      slots.appendChild(b);
      this.slotButtons.push(b);
    }
    this.setSlot(slot);

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
        b.blur(); // keep Space/Enter for the viewport shortcuts, not a re-click
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
    this.foldButton = this.button(actions, 'Cut & roll', 'F', () => this.cb.onCutFold());
    this.button(actions, 'Clear pigment', undefined, () => this.cb.onClear());
    this.button(actions, 'Reset', 'R', () => this.cb.onReset());
    this.pauseButton = this.button(actions, 'Pause', 'Space', () => this.cb.onTogglePause());
    this.pauseButton.classList.add('cm-btn-primary');
    this.pauseButton.setAttribute('aria-pressed', 'false');

    this.el.append(slots, swatches, actions);
    parent.appendChild(this.el);
  }

  /** Highlight `slot` as the one taps land on (does not fire the callback). */
  setSlot(slot: number): void {
    this.slot = Math.min(DROP_SLOTS - 1, Math.max(0, Math.round(slot)));
    this.slotButtons.forEach((b, i) => {
      const on = i === this.slot;
      b.classList.toggle('is-on', on);
      b.setAttribute('aria-checked', String(on));
    });
  }

  /** the drop slot taps currently land on (0-based) */
  get dropSlot(): number {
    return this.slot;
  }

  /** Press animation on the active slot (a pigment just landed there). */
  flashSlot(): void {
    const b = this.slotButtons[this.slot];
    if (b) this.flash(b);
  }

  private button(parent: HTMLElement, label: string, key: string | undefined, onClick: () => void): HTMLButtonElement {
    const b = document.createElement('button');
    b.type = 'button';
    b.className = 'cm-btn';
    b.textContent = label;
    if (key) b.title = `${label} (${key})`;
    b.addEventListener('click', () => { onClick(); b.blur(); });
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
