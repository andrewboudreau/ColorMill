/**
 * Right-hand drawer: parameter sliders bound to MillParams, quality select,
 * auto-orbit toggle and a stats block. Collapsible; a round toggle button
 * sits in the top-right corner. Plain DOM, no framework.
 */
import {
  PARAM_LIMITS, QUALITY_PRESETS, estimateParticleCount, omegaToRpm,
  type MillParams, type QualityPreset
} from '../config/mill';
import { formatCount } from './hud';

export interface PanelCallbacks {
  onParam(key: keyof MillParams, value: number): void;
  onQuality(preset: QualityPreset): void;
  onAutoOrbit(on: boolean): void;
}

export interface PanelStats {
  readonly particles: number;
  readonly grid: string;
  readonly fps: number;
  readonly msFrame: number;
  readonly simSpeed: number;
  /** NaN when timestamp queries are unavailable */
  readonly gpuMs: number;
  readonly simTime: number;
  readonly adapter: string;
}

export interface PanelInitial {
  readonly params: MillParams;
  readonly preset: QualityPreset;
  readonly autoOrbit: boolean;
  readonly open: boolean;
}

interface SliderSpec {
  readonly key: keyof MillParams;
  readonly label: string;
  readonly format: (v: number) => string;
}

const SLIDERS: readonly SliderSpec[] = [
  { key: 'omega', label: 'Roller speed', format: (v) => `${omegaToRpm(v).toFixed(1)} rpm` },
  { key: 'frictionRatio', label: 'Friction ratio', format: (v) => `${v.toFixed(2)}×` },
  { key: 'gap', label: 'Nip gap', format: (v) => v.toFixed(3) },
  { key: 'dispersion', label: 'Dispersion', format: (v) => v.toFixed(2) },
  { key: 'gravity', label: 'Gravity', format: (v) => v.toFixed(1) },
  { key: 'backFriction', label: 'Back-roll friction', format: (v) => v.toFixed(2) }
];

const PRESETS: readonly QualityPreset[] = ['low', 'medium', 'high', 'ultra'];

export class Panel {
  readonly el: HTMLElement;
  readonly toggleButton: HTMLButtonElement;
  private readonly sliders = new Map<keyof MillParams, { input: HTMLInputElement; value: HTMLElement; spec: SliderSpec }>();
  private readonly quality: HTMLSelectElement;
  private readonly orbit: HTMLInputElement;
  private readonly stats: Record<string, HTMLElement> = {};
  private open: boolean;

  private readonly parent: HTMLElement;

  constructor(parent: HTMLElement, private readonly cb: PanelCallbacks, initial: PanelInitial) {
    this.open = initial.open;
    this.parent = parent;

    this.toggleButton = document.createElement('button');
    this.toggleButton.type = 'button';
    this.toggleButton.className = 'cm-drawer-toggle';
    this.toggleButton.setAttribute('aria-label', 'Toggle settings');
    this.toggleButton.setAttribute('aria-controls', 'cm-drawer');
    this.toggleButton.innerHTML = '<span class="cm-drawer-toggle-icon" aria-hidden="true"></span>';
    this.toggleButton.addEventListener('click', () => this.toggle());
    parent.appendChild(this.toggleButton);

    this.el = document.createElement('aside');
    this.el.id = 'cm-drawer';
    this.el.className = 'cm-drawer';
    parent.appendChild(this.el);

    // --- parameters ---------------------------------------------------------
    const params = this.section('Mill');
    for (const spec of SLIDERS) {
      const lim = PARAM_LIMITS[spec.key];
      const field = document.createElement('label');
      field.className = 'cm-field';
      const row = document.createElement('span');
      row.className = 'cm-field-row';
      const name = document.createElement('span');
      name.className = 'cm-field-name';
      name.textContent = spec.label;
      const value = document.createElement('span');
      value.className = 'cm-field-value';
      row.append(name, value);
      const input = document.createElement('input');
      input.type = 'range';
      input.min = String(lim.min);
      input.max = String(lim.max);
      input.step = String(lim.step);
      input.value = String(initial.params[spec.key]);
      input.name = spec.key;
      input.addEventListener('input', () => {
        const v = parseFloat(input.value);
        value.textContent = spec.format(v);
        this.cb.onParam(spec.key, v);
      });
      value.textContent = spec.format(initial.params[spec.key]);
      field.append(row, input);
      params.appendChild(field);
      this.sliders.set(spec.key, { input, value, spec });
    }

    // --- quality --------------------------------------------------------------
    const qsec = this.section('Quality');
    const qfield = document.createElement('label');
    qfield.className = 'cm-field';
    const qname = document.createElement('span');
    qname.className = 'cm-field-name';
    qname.textContent = 'Preset (rebuilds the bank)';
    this.quality = document.createElement('select');
    this.quality.className = 'cm-select';
    this.quality.name = 'quality';
    for (const p of PRESETS) {
      const q = QUALITY_PRESETS[p];
      const o = document.createElement('option');
      o.value = p;
      o.textContent = `${p} · ~${formatCount(estimateParticleCount(q))} particles · ${q.cellsPerUnit}/unit`;
      this.quality.appendChild(o);
    }
    this.quality.value = initial.preset;
    this.quality.addEventListener('change', () => this.cb.onQuality(this.quality.value as QualityPreset));
    qfield.append(qname, this.quality);
    qsec.appendChild(qfield);

    // --- camera ---------------------------------------------------------------
    const csec = this.section('Camera');
    const ofield = document.createElement('label');
    ofield.className = 'cm-field cm-field-check';
    this.orbit = document.createElement('input');
    this.orbit.type = 'checkbox';
    this.orbit.name = 'autoOrbit';
    this.orbit.checked = initial.autoOrbit;
    this.orbit.addEventListener('change', () => this.cb.onAutoOrbit(this.orbit.checked));
    const otext = document.createElement('span');
    otext.textContent = 'Auto-orbit';
    ofield.append(this.orbit, otext);
    csec.appendChild(ofield);
    const chint = document.createElement('p');
    chint.className = 'cm-muted';
    chint.textContent = 'Drag to orbit · wheel or pinch to zoom · double-tap to reset the view';
    csec.appendChild(chint);

    // --- stats ----------------------------------------------------------------
    const ssec = this.section('Stats');
    const dl = document.createElement('dl');
    dl.className = 'cm-stats';
    const rows: Array<[string, string]> = [
      ['particles', 'Particles'], ['grid', 'Grid nodes'], ['fps', 'Frame rate'], ['ms', 'Frame time'],
      ['gpu', 'GPU step'], ['speed', 'Sim speed'], ['simTime', 'Sim time'], ['adapter', 'Adapter']
    ];
    for (const [k, label] of rows) {
      const dt = document.createElement('dt');
      dt.textContent = label;
      const dd = document.createElement('dd');
      dd.textContent = '–';
      dl.append(dt, dd);
      this.stats[k] = dd;
    }
    ssec.appendChild(dl);

    this.applyOpen();
  }

  private section(title: string): HTMLElement {
    const s = document.createElement('section');
    s.className = 'cm-section';
    const h = document.createElement('h2');
    h.textContent = title;
    s.appendChild(h);
    this.el.appendChild(s);
    return s;
  }

  /** Reflect params in the sliders without firing callbacks. */
  setParams(p: MillParams): void {
    for (const [key, s] of this.sliders) {
      const v = p[key];
      if (parseFloat(s.input.value) !== v) s.input.value = String(v);
      s.value.textContent = s.spec.format(v);
    }
  }

  setQuality(preset: QualityPreset): void {
    if (this.quality.value !== preset) this.quality.value = preset;
  }

  setQualityEnabled(enabled: boolean): void {
    this.quality.disabled = !enabled;
  }

  setAutoOrbit(on: boolean): void {
    this.orbit.checked = on;
  }

  setStats(s: PanelStats): void {
    this.stats.particles.textContent = s.particles.toLocaleString();
    this.stats.grid.textContent = s.grid;
    this.stats.fps.textContent = s.fps > 0 ? `${s.fps.toFixed(0)} fps` : '–';
    this.stats.ms.textContent = s.msFrame > 0 ? `${s.msFrame.toFixed(1)} ms` : '–';
    this.stats.gpu.textContent = Number.isFinite(s.gpuMs) ? `${s.gpuMs.toFixed(1)} ms` : 'n/a';
    this.stats.speed.textContent = s.simSpeed > 0 ? `${s.simSpeed.toFixed(2)}× real time` : '–';
    this.stats.simTime.textContent = `${s.simTime.toFixed(1)} s`;
    this.stats.adapter.textContent = s.adapter;
  }

  get isOpen(): boolean {
    return this.open;
  }

  setOpen(open: boolean): void {
    this.open = open;
    this.applyOpen();
  }

  toggle(): void {
    this.setOpen(!this.open);
  }

  private applyOpen(): void {
    this.el.classList.toggle('is-open', this.open);
    this.toggleButton.classList.toggle('is-open', this.open);
    this.toggleButton.setAttribute('aria-expanded', String(this.open));
    // lets the bottom bar make room for the drawer on wide screens (styles.css)
    this.parent.classList.toggle('is-drawer-open', this.open);
  }

  destroy(): void {
    this.parent.classList.remove('is-drawer-open');
    this.el.remove();
    this.toggleButton.remove();
  }
}
