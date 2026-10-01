/**
 * The knife: a pointer gesture that draws a cut across the sheet on the front roll.
 *
 * While the knife is armed (the Knife button / K) or Shift is held, a drag that starts on the
 * canvas draws a stroke instead of orbiting the camera; on release the stroke, projected onto the
 * sheet (`toSheet`), is handed to `onCut`. Only ONE pointer is claimed: the first that goes down
 * while armed. Any other pointer is left alone and reaches the camera controls (or, later, a grab
 * tool), so "one hand cuts, the other holds" can be added without changing this module.
 *
 * The listeners sit on `root` in the capture phase, so they see the pointer before the canvas's
 * orbit controls and can stop it from reaching them.
 */
import type { SheetPoint } from '../sim/cut';

export interface KnifeOptions {
  /** true when this pointer-down should start a cut (knife armed, or Shift held) */
  armed(e: PointerEvent): boolean;
  /** project a point on screen (client pixels) onto the sheet; null where it misses the roll */
  toSheet(clientX: number, clientY: number): SheetPoint | null;
  /** the finished stroke in sheet coordinates (at least two points) */
  onCut(points: SheetPoint[]): void;
}

export interface Knife {
  /** true while a stroke is being drawn */
  readonly drawing: boolean;
  destroy(): void;
}

export function installKnife(root: HTMLElement, canvas: HTMLCanvasElement, opts: KnifeOptions): Knife {
  const svgNs = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNs, 'svg');
  svg.setAttribute('class', 'cm-knife-stroke');
  svg.setAttribute('aria-hidden', 'true');
  const line = document.createElementNS(svgNs, 'polyline');
  svg.appendChild(line);
  root.appendChild(svg);

  let pointerId: number | null = null;
  let screen: [number, number][] = [];
  let sheet: SheetPoint[] = [];
  let fadeTimer = 0;

  const local = (e: PointerEvent): [number, number] => {
    const r = canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const paint = (): void => {
    line.setAttribute('points', screen.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' '));
  };
  const add = (e: PointerEvent): void => {
    screen.push(local(e));
    const p = opts.toSheet(e.clientX, e.clientY);
    if (p) sheet.push(p);
    paint();
  };

  const onDown = (e: PointerEvent): void => {
    if (e.target !== canvas || pointerId !== null) return;
    if (e.pointerType === 'mouse' && e.button !== 0) return;
    if (!opts.armed(e)) return;
    pointerId = e.pointerId;
    window.clearTimeout(fadeTimer);
    line.style.opacity = '1';
    screen = [];
    sheet = [];
    add(e);
    try { canvas.setPointerCapture(e.pointerId); } catch { /* not supported everywhere */ }
    e.stopPropagation();
    e.preventDefault();
  };
  const onMove = (e: PointerEvent): void => {
    if (e.pointerId !== pointerId) return;
    add(e);
    e.stopPropagation();
    e.preventDefault();
  };
  const onUp = (e: PointerEvent): void => {
    if (e.pointerId !== pointerId) return;
    pointerId = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch { /* ignore */ }
    e.stopPropagation();
    e.preventDefault();
    const cancelled = e.type === 'pointercancel';
    if (!cancelled) add(e);
    if (!cancelled && sheet.length >= 2) opts.onCut(sheet);
    // let the stroke linger a moment, then fade it out
    fadeTimer = window.setTimeout(() => { line.style.opacity = '0'; }, 250);
  };

  root.addEventListener('pointerdown', onDown, { capture: true });
  root.addEventListener('pointermove', onMove, { capture: true });
  root.addEventListener('pointerup', onUp, { capture: true });
  root.addEventListener('pointercancel', onUp, { capture: true });

  return {
    get drawing() { return pointerId !== null; },
    destroy() {
      window.clearTimeout(fadeTimer);
      root.removeEventListener('pointerdown', onDown, { capture: true });
      root.removeEventListener('pointermove', onMove, { capture: true });
      root.removeEventListener('pointerup', onUp, { capture: true });
      root.removeEventListener('pointercancel', onUp, { capture: true });
      svg.remove();
    }
  };
}
