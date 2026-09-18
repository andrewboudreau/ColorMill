/**
 * Keyboard shortcuts (design §9): Space pause, R reset, F cut & roll, C cut & fold,
 * 1–8 pigments, arrows speed (up/down) and nip gap (left/right), [ and ]
 * move the pigment drop slot, - and = step the chunk size, P toggles the settings drawer. Keys are ignored while a form control has focus.
 */

export interface KeyActions {
  togglePause(): void;
  reset(): void;
  cutFold(): void;
  /** cut & fold: flop one half of the top of the mill over onto the other */
  cutFlop(): void;
  /** zero-based palette index */
  pigment(index: number): void;
  /** +1 / -1 notches */
  speed(direction: 1 | -1): void;
  gap(direction: 1 | -1): void;
  /** move the pigment drop slot one step left (-1) or right (+1) */
  slot(direction: 1 | -1): void;
  /** step the pigment chunk size down (-1) or up (+1) */
  chunkSize(direction: 1 | -1): void;
  togglePanel(): void;
}

function isEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || target.isContentEditable;
}

/** Install the shortcuts on `target`; returns a function that removes them. */
export function installKeyboard(target: Window | HTMLElement, a: KeyActions): () => void {
  const onKey = (ev: Event): void => {
    const e = ev as KeyboardEvent;
    if (e.defaultPrevented || e.metaKey || e.ctrlKey || e.altKey) return;
    if (isEditable(e.target) && e.key !== 'Escape') return;
    let handled = true;
    switch (e.key) {
      case ' ': a.togglePause(); break;
      case 'r': case 'R': a.reset(); break;
      case 'f': case 'F': a.cutFold(); break;
      case 'c': case 'C': a.cutFlop(); break;
      case 'p': case 'P': a.togglePanel(); break;
      case 'ArrowUp': a.speed(1); break;
      case 'ArrowDown': a.speed(-1); break;
      case 'ArrowRight': a.gap(1); break;
      case 'ArrowLeft': a.gap(-1); break;
      case '[': a.slot(-1); break;
      case ']': a.slot(1); break;
      case '-': case '_': a.chunkSize(-1); break;
      case '=': case '+': a.chunkSize(1); break;
      default:
        if (e.key >= '1' && e.key <= '8' && e.key.length === 1) a.pigment(e.key.charCodeAt(0) - '1'.charCodeAt(0));
        else handled = false;
    }
    if (handled) e.preventDefault();
  };
  target.addEventListener('keydown', onKey);
  return () => target.removeEventListener('keydown', onKey);
}
