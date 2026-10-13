/* snapshot.js: draw a "snapshots" trace (docsgen/traces.py), one state per presenter click.
 *
 *   const fig = new SnapshotStepper(containerEl, trace, {readouts: el, caption: el});
 *   fig.render(k);                       // draws steps[k]; pure, no hidden state
 *   fig.bindReveal(Reveal, sectionEl);   // step fragments drive render(k) forward and back
 *
 * Every step carries the complete visual state, so the renderer computes nothing.
 */
(function (global) {
  "use strict";
  const NS = "http://www.w3.org/2000/svg";
  const S = (x) => String(x);
  const key = (a, b) => (+a < +b ? `${a}_${b}` : `${b}_${a}`);

  function el(name, attrs, parent) {
    const e = document.createElementNS(NS, name);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }

  class SnapshotStepper {
    constructor(container, trace, opts = {}) {
      this.trace = trace;
      this.pos = trace.nodes;
      this.ids = Object.keys(trace.nodes);
      this.opts = Object.assign({ radius: 20, margin: 16, readouts: null, caption: null }, opts);
      const xs = this.ids.map((v) => this.pos[v][0]), ys = this.ids.map((v) => this.pos[v][1]);
      const m = this.opts.margin + this.opts.radius;
      const x0 = Math.min(...xs) - m, y0 = Math.min(...ys) - m;
      const w = Math.max(...xs) - Math.min(...xs) + 2 * m, h = Math.max(...ys) - Math.min(...ys) + 2 * m;
      this.svg = el("svg", { class: "ssv", viewBox: `${x0} ${y0} ${w} ${h}`, role: "img",
                             "aria-label": (trace.meta && trace.meta.title) || trace.id });
      container.appendChild(this.svg);
      this.render(0);
    }

    get length() { return this.trace.steps.length - 1; }  // clicks after the initial state

    render(k) {
      k = Math.max(0, Math.min(this.length, k | 0));
      const step = this.trace.steps[k], P = this.pos, svg = this.svg, r = this.opts.radius;
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      const gT = el("g", {}, svg), gE = el("g", {}, svg), gX = el("g", {}, svg), gN = el("g", {}, svg);
      // triangles: the six (i, i+1, 6+i) of the kagome star, drawn as background
      for (let i = 0; i < 6; i++) {
        const pts = [S(i), S((i + 1) % 6), S(6 + i)].filter((v) => v in P);
        if (pts.length === 3) el("polygon", { class: "ssv-tri", points: pts.map((v) => P[v].join(",")).join(" ") }, gT);
      }
      const classes = step.edges || {};
      this.trace.edges.forEach(([a, b]) => {
        const cls = classes[key(a, b)];
        el("line", { x1: P[a][0], y1: P[a][1], x2: P[b][0], y2: P[b][1], class: "ssv-edge" + (cls ? " " + cls : "") },
           cls ? gX : gE);
      });
      (step.extra || []).forEach(([a, b, cls]) => {
        const [x1, y1] = P[S(a)], [x2, y2] = P[S(b)];
        // bow non-bonds outwards so they never hide under a lattice edge
        const mx = (x1 + x2) / 2, my = (y1 + y2) / 2, len = Math.hypot(mx, my) || 1;
        const cx = mx + (mx / len) * 40, cy = my + (my / len) * 40;
        el("path", { d: `M${x1},${y1} Q${cx},${cy} ${x2},${y2}`, class: "ssv-extra " + cls }, gX);
      });
      const labels = step.labels || {};
      this.ids.forEach((v) => {
        const g = el("g", { class: "ssv-node " + ((step.nodes || {})[v] || "") }, gN);
        el("circle", { cx: P[v][0], cy: P[v][1], r }, g);
        el("text", { x: P[v][0], y: P[v][1] }, g).textContent = labels[v] ?? v;
      });
      const root = this.opts.readouts;
      if (root) root.querySelectorAll("[data-readout]").forEach((e) => {
        e.textContent = (step.readouts || {})[e.dataset.readout] ?? "–";
      });
      if (this.opts.caption) this.opts.caption.textContent = step.caption || "";
      return this;
    }

    bindReveal(Reveal, section) {
      const mine = (e) => e.fragment && section.contains(e.fragment) && e.fragment.dataset.step;
      Reveal.on("fragmentshown", (e) => { if (mine(e)) this.render(+e.fragment.dataset.step); });
      Reveal.on("fragmenthidden", (e) => { if (mine(e)) this.render(+e.fragment.dataset.step - 1); });
      const sync = (e) => {
        if (e.currentSlide === section) this.render(section.querySelectorAll(".fragment.visible[data-step]").length);
      };
      Reveal.on("slidechanged", sync);
      Reveal.on("ready", sync);
      return this;
    }
  }

  global.SnapshotStepper = SnapshotStepper;
})(typeof window !== "undefined" ? window : globalThis);
