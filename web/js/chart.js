// Dependency-free SVG line chart, built for live-appending training metrics.
//
// Deliberate choices:
//  * One y-axis only. Loss and learning rate live on separate charts rather
//    than sharing a dual axis, which misleads by making unrelated scales look
//    comparable. Two series appear together only when they measure the same
//    thing in the same unit — training loss and held-out loss.
//  * A single series carries no legend box; the title names it. Two or more
//    always get both a legend and an end-of-line label, so identity never
//    depends on colour alone.
//  * Text uses the theme's ink tokens, never the series color.
//
// The two series colours are validated, not chosen by eye: OKLab ΔE under
// simulated protanopia and deuteranopia, lightness band, chroma floor and
// WCAG contrast against each theme's surface. See --series-1/--series-2 in
// styles.css.

const NS = "http://www.w3.org/2000/svg";

function el(name, attrs = {}) {
  const n = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  return n;
}

function niceTicks(min, max, count = 5) {
  if (!isFinite(min) || !isFinite(max)) return [0, 1];
  if (min === max) { min -= 0.5; max += 0.5; }
  const span = max - min;
  const raw = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1) * mag;
  const start = Math.floor(min / step) * step;
  const out = [];
  for (let v = start; v <= max + step * 0.5; v += step) out.push(+v.toFixed(10));
  return out;
}

const fmtTick = (v) => {
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a < 0.001 || a >= 1e5) return v.toExponential(0);
  if (a < 1) return v.toFixed(a < 0.01 ? 4 : 3);
  if (a < 100) return v.toFixed(2).replace(/\.?0+$/, "");
  return String(Math.round(v));
};

export class LineChart {
  /**
   * @param {HTMLElement} mount
   * @param {{title?:string, yLabel?:string, xLabel?:string, color?:string,
   *          height?:number, format?:(n:number)=>string,
   *          series?:{key:string,label:string,color?:string,dashed?:boolean}[]}} opts
   */
  constructor(mount, opts = {}) {
    this.mount = mount;
    this.o = {
      title: "", yLabel: "", xLabel: "Step", height: 240,
      color: "var(--series-1)", format: fmtTick, ...opts,
    };
    // A single unnamed series is the common case and stays the default, so
    // every existing call site keeps working unchanged.
    this.series = (this.o.series || [{ key: "main", label: this.o.title }])
      .map((s, i) => ({
        dashed: false,
        color: s.color || (i === 0 ? this.o.color : `var(--series-${i + 1})`),
        ...s,
        points: [],
      }));
    this._build();
    this._ro = new ResizeObserver(() => this.render());
    this._ro.observe(mount);
  }

  _find(key) {
    return this.series.find((s) => s.key === key) || this.series[0];
  }

  _build() {
    this.mount.innerHTML = "";
    this.mount.classList.add("chart-wrap");
    if (this.o.title) {
      const h = document.createElement("div");
      h.className = "row-between chart-head";
      h.innerHTML =
        `<span style="font-weight:600;font-size:13px">${this.o.title}</span>` +
        `<span class="tiny mono" data-readout style="color:var(--text-2)"></span>`;
      this.mount.appendChild(h);
      this.readout = h.querySelector("[data-readout]");
    }
    if (this.series.length > 1) {
      const leg = document.createElement("div");
      leg.className = "chart-legend";
      leg.innerHTML = this.series.map((s) =>
        `<span class="chart-legend-item"><i style="background:${s.color}"></i>${
          s.label}</span>`).join("");
      this.mount.appendChild(leg);
    }
    this.svg = el("svg", { width: "100%", height: this.o.height,
                           role: "img", "aria-label": this.o.title || "chart" });
    this.svg.style.display = "block";
    this.svg.style.overflow = "visible";
    this.mount.appendChild(this.svg);

    // Crosshair tooltip: a line chart in a browser should be interactive.
    this.tip = document.createElement("div");
    Object.assign(this.tip.style, {
      position: "absolute", pointerEvents: "none", opacity: "0",
      background: "var(--surface)", border: "1px solid var(--border)",
      borderRadius: "6px", padding: "5px 9px", fontSize: "12px",
      boxShadow: "var(--shadow)", whiteSpace: "nowrap", zIndex: "5",
      transition: "opacity .1s",
    });
    this.mount.appendChild(this.tip);

    this.svg.addEventListener("mousemove", (e) => this._hover(e));
    this.svg.addEventListener("mouseleave", () => {
      this.tip.style.opacity = "0";
      if (this._cross) this._cross.setAttribute("opacity", "0");
      (this._dots || []).forEach((d) => d.setAttribute("opacity", "0"));
    });
  }

  setData(points) { this.setSeries(this.series[0].key, points); }
  push(point) { this.pushSeries(this.series[0].key, point); }

  setSeries(key, points) {
    this._find(key).points = points.filter((p) => isFinite(p.y));
    this.render();
  }

  pushSeries(key, point) {
    if (!isFinite(point.y)) return;
    this._find(key).points.push(point);
    this.render();
  }

  render() {
    const W = this.mount.clientWidth || 600;
    // Charts shrink on phones so the plot plus its stats still fit one screen.
    const narrow = W < 520;
    const H = narrow ? Math.round(this.o.height * 0.75) : this.o.height;
    this.svg.setAttribute("height", H);
    const pad = { t: 8, r: 12, b: 28, l: 40 };
    const ih = Math.max(10, H - pad.t - pad.b);
    this.svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    this.svg.innerHTML = "";

    const live = this.series.filter((s) => s.points.length);
    if (!live.length) {
      const t = el("text", { x: W / 2, y: H / 2, "text-anchor": "middle",
                             fill: "var(--text-3)", "font-size": 12.5 });
      t.textContent = "Waiting for the first measurement…";
      this.svg.appendChild(t);
      return;
    }

    // One scale across every series: they share a unit, which is the only
    // reason they are allowed to share a chart at all.
    const all = live.flatMap((s) => s.points);
    const xs = all.map((p) => p.x);
    const ys = all.map((p) => p.y);
    const x0 = Math.min(...xs), x1 = Math.max(...xs);
    let y0 = Math.min(...ys), y1 = Math.max(...ys);
    const padY = (y1 - y0) * 0.12 || Math.abs(y1 || 1) * 0.12;
    y0 -= padY; y1 += padY;
    if (ys.every((v) => v >= 0) && y0 < 0) y0 = 0;

    const yTicks = niceTicks(y0, y1, 4).filter((t) => t >= y0 && t <= y1);
    // Gutter is measured from the widest tick label, not fixed: an exponential
    // format like "0.0e+0" is far wider than "3.01" and would be clipped.
    const widest = Math.max(0, ...yTicks.map((t) => this.o.format(t).length));
    pad.l = Math.min(Math.round(W * 0.3), Math.max(30, widest * 7 + 12));
    // Room on the right for end-of-line labels, which carry identity when
    // more than one series shares the plot.
    const labelled = live.length > 1 && !narrow;
    if (labelled) pad.r = 46;
    const iw = Math.max(10, W - pad.l - pad.r);
    this._geom = { W, H, pad, iw, ih };

    const sx = (v) => pad.l + (x1 === x0 ? iw / 2 : ((v - x0) / (x1 - x0)) * iw);
    const sy = (v) => pad.t + ih - (y1 === y0 ? ih / 2 : ((v - y0) / (y1 - y0)) * ih);
    this._scale = { sx, sy, x0, x1 };

    // Recessive grid + axis labels in ink tokens, never the series color.
    for (const t of yTicks) {
      const y = sy(t);
      this.svg.appendChild(el("line", {
        x1: pad.l, x2: W - pad.r, y1: y, y2: y,
        stroke: "var(--border)", "stroke-width": 1, opacity: .7 }));
      const lab = el("text", { x: pad.l - 8, y: y + 3.5, "text-anchor": "end",
                               fill: "var(--text-3)", "font-size": 11 });
      lab.textContent = this.o.format(t);
      this.svg.appendChild(lab);
    }
    for (const t of niceTicks(x0, x1, 5)) {
      if (t < x0 || t > x1) continue;
      const lab = el("text", { x: sx(t), y: H - 8, "text-anchor": "middle",
                               fill: "var(--text-3)", "font-size": 11 });
      lab.textContent = String(Math.round(t));
      this.svg.appendChild(lab);
    }

    // Only the first series gets a fill. Two translucent areas over each other
    // make a third colour that means nothing.
    const primary = live[0];
    if (live.length === 1) {
      const d0 = primary.points.map((p, i) =>
        `${i ? "L" : "M"}${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`).join("");
      const gid = "g" + Math.random().toString(36).slice(2, 8);
      const grad = el("linearGradient", { id: gid, x1: 0, y1: 0, x2: 0, y2: 1 });
      grad.appendChild(el("stop", { offset: "0%", "stop-color": primary.color,
                                    "stop-opacity": .22 }));
      grad.appendChild(el("stop", { offset: "100%", "stop-color": primary.color,
                                    "stop-opacity": 0 }));
      const defs = el("defs"); defs.appendChild(grad); this.svg.appendChild(defs);
      const px0 = sx(primary.points[0].x);
      const px1 = sx(primary.points[primary.points.length - 1].x);
      this.svg.appendChild(el("path", {
        d: `${d0}L${px1.toFixed(2)},${(pad.t + ih).toFixed(2)}` +
           `L${px0.toFixed(2)},${(pad.t + ih).toFixed(2)}Z`,
        fill: `url(#${gid})`, stroke: "none" }));
    }

    for (const s of live) {
      const d = s.points.map((p, i) =>
        `${i ? "L" : "M"}${sx(p.x).toFixed(2)},${sy(p.y).toFixed(2)}`).join("");
      this.svg.appendChild(el("path", {
        d, fill: "none", stroke: s.color, "stroke-width": 2,
        "stroke-dasharray": s.dashed ? "5 4" : null,
        "stroke-linejoin": "round", "stroke-linecap": "round" }));

      // Last point gets a marker with a surface ring so it stays legible
      // wherever it lands, including on top of another line.
      const last = s.points[s.points.length - 1];
      this.svg.appendChild(el("circle", {
        cx: sx(last.x), cy: sy(last.y), r: 4, fill: s.color,
        stroke: "var(--surface)", "stroke-width": 2 }));

      if (labelled) {
        const t = el("text", { x: sx(last.x) + 9, y: sy(last.y) + 3.5,
                               fill: "var(--text-2)", "font-size": 10.5,
                               "font-weight": 600 });
        t.textContent = s.label;
        this.svg.appendChild(t);
      }
    }

    this._cross = el("line", { y1: pad.t, y2: pad.t + ih, stroke: "var(--text-3)",
                               "stroke-width": 1, "stroke-dasharray": "3 3", opacity: 0 });
    this.svg.appendChild(this._cross);
    this._dots = live.map((s) => {
      const c = el("circle", { r: 4.5, fill: s.color, stroke: "var(--surface)",
                               "stroke-width": 2, opacity: 0 });
      this.svg.appendChild(c);
      return c;
    });
    this._live = live;

    if (this.readout) this.readout.textContent =
      this.o.format(primary.points[primary.points.length - 1].y);
  }

  _hover(e) {
    if (!this._scale || !this._live?.length) return;
    const rect = this.svg.getBoundingClientRect();
    const mx = e.clientX - rect.left;

    // Nearest point per series, independently. Held-out loss is measured every
    // few dozen steps rather than every step, so snapping both series to one
    // shared index would silently show the wrong number for the sparse one.
    let px = null;
    const parts = [];
    this._live.forEach((s, i) => {
      let best = s.points[0], bd = Infinity;
      for (const p of s.points) {
        const d = Math.abs(this._scale.sx(p.x) - mx);
        if (d < bd) { bd = d; best = p; }
      }
      const x = this._scale.sx(best.x), y = this._scale.sy(best.y);
      if (i === 0) px = x;
      this._dots[i].setAttribute("cx", x);
      this._dots[i].setAttribute("cy", y);
      this._dots[i].setAttribute("opacity", 1);
      parts.push(this._live.length > 1
        ? `<span style="color:var(--text-3)">${s.label}</span> <strong>${
            this.o.format(best.y)}</strong>`
        : `<strong>${this.o.format(best.y)}</strong>` +
          `<span style="color:var(--text-3)"> · step ${best.x}</span>`);
    });

    this._cross.setAttribute("x1", px);
    this._cross.setAttribute("x2", px);
    this._cross.setAttribute("opacity", 1);
    this.tip.innerHTML = parts.join(
      '<span style="color:var(--border-2)"> │ </span>');
    this.tip.style.opacity = "1";
    const tw = this.tip.offsetWidth;
    const left = Math.min(Math.max(px - tw / 2, 0), (this.mount.clientWidth || 600) - tw);
    this.tip.style.left = `${left}px`;
    this.tip.style.top = `${this._scale.sy(this._live[0].points.at(-1).y) * 0 +
      (this.o.title ? 22 : 0) + 4}px`;
  }

  destroy() { this._ro.disconnect(); }
}
