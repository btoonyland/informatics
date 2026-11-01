/* Draws report.html from the JSON embedded in the page. Plain SVG, no libraries.
   Text from the data is only ever inserted with textContent, never as HTML. */
(function () {
  "use strict";

  var DATA = JSON.parse(document.getElementById("report-data").textContent);
  var SVGNS = "http://www.w3.org/2000/svg";
  var main = document.getElementById("main");
  var tooltip = document.getElementById("tooltip");
  var redraws = [];

  // Helpers ------------------------------------------------------------------------------------
  function h(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (key) {
      var value = attrs[key];
      if (value == null) return;
      if (key === "text") node.textContent = value;
      else if (key === "class") node.className = value;
      else node.setAttribute(key, value);
    });
    (children || []).forEach(function (child) {
      if (child == null) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  function s(tag, attrs, parent) {
    var node = document.createElementNS(SVGNS, tag);
    Object.keys(attrs || {}).forEach(function (key) {
      if (attrs[key] == null) return;
      if (key === "text") node.textContent = attrs[key];
      else node.setAttribute(key, attrs[key]);
    });
    if (parent) parent.appendChild(node);
    return node;
  }

  function pct(value, digits) {
    if (value == null) return "–";
    return (100 * value).toFixed(digits == null ? 1 : digits) + "%";
  }
  function count(value) {
    return value == null ? "–" : Number(value).toLocaleString("en-US");
  }
  function fixed(value, digits) {
    return value == null ? "–" : Number(value).toFixed(digits);
  }
  function compact(value) {
    var size = Math.abs(value);
    if (size >= 1e6) return (value / 1e6).toFixed(1).replace(/\.0$/, "") + "M";
    if (size >= 1e4) return Math.round(value / 1e3) + "K";
    return Number(value).toLocaleString("en-US");
  }
  function niceStep(span, n) {
    var raw = span / Math.max(n, 1);
    var magnitude = Math.pow(10, Math.floor(Math.log10(raw)));
    var f = raw / magnitude;
    return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 2.5 ? 2.5 : f <= 5 ? 5 : 10) * magnitude;
  }
  // Round tick values from the step at or below ``min`` to the step at or above ``max``, so an
  // axis always covers its data.
  function ticks(min, max, n, integer) {
    if (!(max > min)) max = min + 1;
    var step = niceStep(max - min, n);
    if (integer) step = Math.max(1, Math.round(step));
    var first = Math.floor(min / step + 1e-9) * step;
    var last = Math.ceil(max / step - 1e-9) * step;
    var out = [];
    for (var v = first; v <= last + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  function linear(d0, d1, r0, r1) {
    return function (v) { return r0 + ((v - d0) / ((d1 - d0) || 1)) * (r1 - r0); };
  }
  function safeUrl(url) {
    return typeof url === "string" && url.indexOf("https://") === 0 ? url : null;
  }
  // Marks: 4px rounded data end, square at the baseline.
  function columnPath(x, y, w, height) {
    var r = Math.min(4, w / 2, height);
    return "M" + x + "," + (y + height) + "V" + (y + r) + "A" + r + "," + r + " 0 0 1 " +
      (x + r) + "," + y + "H" + (x + w - r) + "A" + r + "," + r + " 0 0 1 " + (x + w) + "," +
      (y + r) + "V" + (y + height) + "Z";
  }
  function barPath(x0, y, length, thickness) {  // horizontal; negative length grows left
    var r = Math.min(4, thickness / 2, Math.abs(length));
    var end = x0 + length;
    var dir = length >= 0 ? 1 : -1;
    var sweep = length >= 0 ? 1 : 0;
    return "M" + x0 + "," + y + "H" + (end - dir * r) + "A" + r + "," + r + " 0 0 " + sweep + " " +
      end + "," + (y + r) + "V" + (y + thickness - r) + "A" + r + "," + r + " 0 0 " + sweep + " " +
      (end - dir * r) + "," + (y + thickness) + "H" + x0 + "Z";
  }
  function color(slot) { return "var(--series-" + slot + ")"; }

  // Tooltip: values lead, labels follow; series keyed by a short line --------------------------
  function fillTip(title, rows) {
    while (tooltip.firstChild) tooltip.removeChild(tooltip.firstChild);
    if (title) tooltip.appendChild(h("div", { class: "title", text: title }));
    rows.forEach(function (row) {
      var line = h("div", { class: "row" });
      if (row.color) {
        var key = h("span", { class: "swatch-line" });
        key.style.background = row.color;
        line.appendChild(key);
      }
      line.appendChild(h("strong", { text: row.value }));
      if (row.label) line.appendChild(h("span", { text: row.label }));
      tooltip.appendChild(line);
    });
    tooltip.hidden = false;
  }
  function placeTip(x, y) {
    var box = tooltip.getBoundingClientRect();
    var left = x + 14, top = y + 14;
    if (left + box.width > window.innerWidth - 8) left = x - box.width - 14;
    if (top + box.height > window.innerHeight - 8) top = y - box.height - 14;
    tooltip.style.left = Math.max(8, left) + "px";
    tooltip.style.top = Math.max(8, top) + "px";
  }
  function showTip(event, title, rows) {
    fillTip(title, rows);
    placeTip(event.clientX, event.clientY);
  }
  function showTipAt(node, title, rows) {
    var r = node.getBoundingClientRect();
    fillTip(title, rows);
    placeTip(r.left + r.width / 2, r.top);
  }
  function hideTip() { tooltip.hidden = true; }
  // A mark (or its larger hit area) shows the same tooltip on hover and on keyboard focus.
  function hover(target, mark, content) {
    target.setAttribute("tabindex", "0");
    target.addEventListener("pointermove", function (event) {
      var c = content();
      if (mark) mark.classList.add("active");
      showTip(event, c.title, c.rows);
    });
    target.addEventListener("pointerleave", function () {
      if (mark) mark.classList.remove("active");
      hideTip();
    });
    target.addEventListener("focus", function () {
      var c = content();
      if (mark) mark.classList.add("active");
      showTipAt(mark || target, c.title, c.rows);
    });
    target.addEventListener("blur", function () {
      if (mark) mark.classList.remove("active");
      hideTip();
    });
  }

  // Layout pieces --------------------------------------------------------------------------------
  function card(id, title, subtitle) {
    var section = h("section", { class: "card", id: id }, [
      h("h2", { text: title }),
      subtitle ? h("p", { class: "sub", text: subtitle }) : null,
    ]);
    main.appendChild(section);
    return section;
  }
  function chart(parent, height, label, draw) {
    var box = h("div", { class: "chart" });
    parent.appendChild(box);
    function render() {
      var width = Math.max(box.clientWidth, 240);
      var tall = typeof height === "function" ? height(width) : height;
      while (box.firstChild) box.removeChild(box.firstChild);
      var svg = s("svg", {
        width: width, height: tall, viewBox: "0 0 " + width + " " + tall,
        role: "group", "aria-label": label,
      });
      box.appendChild(svg);
      draw(svg, width, tall);
    }
    redraws.push(render);
    render();
    return box;
  }
  function legend(parent, items, kind) {
    var row = h("div", { class: "legend" });
    items.forEach(function (item) {
      var key = h("span", { class: kind === "dot" ? "swatch-dot" : "swatch-line" });
      key.style.background = item.color;
      row.appendChild(h("span", { class: "key" }, [key, item.label]));
    });
    parent.appendChild(row);
  }
  function tableView(parent, headers, rows, numeric) {
    var head = h("tr", {}, headers.map(function (text, i) {
      return h("th", { text: text, class: numeric && numeric[i] ? "num" : null, scope: "col" });
    }));
    var body = h("tbody", {}, rows.map(function (row) {
      return h("tr", {}, row.map(function (cell, i) {
        return h("td", { text: cell == null ? "" : String(cell), class: numeric && numeric[i] ? "num" : null });
      }));
    }));
    parent.appendChild(h("details", { class: "table-view" }, [
      h("summary", { text: "Show as table" }),
      h("div", { class: "table-wrap" }, [h("table", {}, [h("thead", {}, [head]), body])]),
    ]));
  }
  function yAxis(svg, y, values, left, right, format) {
    values.forEach(function (v, i) {
      s("line", { x1: left, x2: right, y1: y(v), y2: y(v), class: i === 0 ? "axis-line" : "grid-line" }, svg);
      s("text", { x: left - 6, y: y(v) + 4, "text-anchor": "end", text: format(v) }, svg);
    });
  }
  function xYears(svg, x, years, bottom, width) {
    var shown = ticks(years[0], years[years.length - 1], Math.max(2, Math.floor(width / 70)), true);
    shown.forEach(function (year) {
      if (year < years[0] || year > years[years.length - 1]) return;
      s("text", { x: x(year), y: bottom + 18, "text-anchor": "middle", text: String(year) }, svg);
    });
  }

  // Column chart (one series) --------------------------------------------------------------------
  function columns(parent, opts) {
    chart(parent, 240, opts.label, function (svg, W, H) {
      var m = { top: 12, right: 8, bottom: 28, left: 48 };
      var max = Math.max.apply(null, opts.values.concat([1]));
      var yt = ticks(0, max, 4, true);
      var y = linear(0, yt[yt.length - 1], H - m.bottom, m.top);
      yAxis(svg, y, yt, m.left, W - m.right, compact);
      var years = opts.labels;
      var first = years[0], span = years[years.length - 1] - first + 1;
      var band = (W - m.left - m.right) / Math.max(span, 1);
      var width = Math.max(2, Math.min(24, band - 2));  // at most 24px, with a gap between bars
      var x = function (year) { return m.left + band * (year - first + 0.5); };
      opts.values.forEach(function (value, i) {
        var cx = x(years[i]);
        var top = y(value), base = y(0);
        var bar = s("path", {
          d: columnPath(cx - width / 2, top, width, Math.max(base - top, 0)),
          class: "mark", style: "fill: " + color(1),
        }, svg);
        var hit = s("rect", {
          x: cx - band / 2, y: m.top, width: band, height: base - m.top,
          fill: "transparent", "aria-label": opts.tip(i).title,
        }, svg);
        hover(hit, bar, function () { return opts.tip(i); });
      });
      xYears(svg, x, years, H - m.bottom, W - m.left - m.right);
    });
  }

  // Line chart with optional interval bands and a crosshair --------------------------------------
  function lines(parent, opts) {
    chart(parent, opts.height || 240, opts.label, function (svg, W, H) {
      var labelled = opts.directLabels && opts.series.length > 1 && opts.series.length <= 4;
      var m = { top: 12, right: labelled ? 84 : 12, bottom: 28, left: 48 };
      var xs = opts.x;
      var max = 0;
      opts.series.forEach(function (series) {
        (series.high || series.values).forEach(function (v) { if (v != null && v > max) max = v; });
      });
      var yt = ticks(0, max || 1, opts.yTicks || 4);
      var y = linear(0, yt[yt.length - 1], H - m.bottom, m.top);
      var x0 = opts.domain ? opts.domain[0] : xs[0];
      var x1 = opts.domain ? opts.domain[1] : xs[xs.length - 1];
      var x = linear(x0, x1 === x0 ? x0 + 1 : x1, m.left, W - m.right);
      yAxis(svg, y, yt, m.left, W - m.right, opts.yFormat);
      xYears(svg, x, [x0, x1], H - m.bottom, W - m.left - m.right);

      opts.series.forEach(function (series) {
        if (series.low && series.high) {
          var top = [], bottom = [];
          xs.forEach(function (xv, i) {
            if (series.low[i] == null || series.high[i] == null) return;
            top.push(x(xv) + "," + y(series.high[i]));
            bottom.unshift(x(xv) + "," + y(series.low[i]));
          });
          if (top.length > 1) {
            s("path", { d: "M" + top.join("L") + "L" + bottom.join("L") + "Z", class: "band",
                        style: "fill: " + series.color }, svg);
          }
        }
        var d = "", pen = false;
        series.values.forEach(function (v, i) {
          if (v == null) { pen = false; return; }
          d += (pen ? "L" : "M") + x(xs[i]) + "," + y(v);
          pen = true;
        });
        s("path", { d: d, class: "line", style: "stroke: " + series.color }, svg);
        var last = series.values.length - 1;
        while (last >= 0 && series.values[last] == null) last--;
        if (last >= 0) {
          s("circle", { cx: x(xs[last]), cy: y(series.values[last]), r: 4, class: "dot",
                        style: "fill: " + series.color }, svg);
          if (labelled) {
            s("text", { x: x(xs[last]) + 8, y: y(series.values[last]) + 4, class: "label halo",
                        text: series.name }, svg);
          }
        }
      });

      // Crosshair: snaps to the nearest x and lists every series there.
      var cross = s("line", { y1: m.top, y2: H - m.bottom, class: "crosshair", visibility: "hidden" }, svg);
      var dots = opts.series.map(function (series) {
        return s("circle", { r: 4, class: "dot", visibility: "hidden", style: "fill: " + series.color }, svg);
      });
      var overlay = s("rect", { x: m.left, y: m.top, width: W - m.left - m.right,
                                height: H - m.top - m.bottom, fill: "transparent",
                                tabindex: 0, "aria-label": opts.label + ": use the arrow keys" }, svg);
      var current = xs.length - 1;
      function show(i, event) {
        current = i;
        cross.setAttribute("x1", x(xs[i]));
        cross.setAttribute("x2", x(xs[i]));
        cross.setAttribute("visibility", "visible");
        opts.series.forEach(function (series, k) {
          var v = series.values[i];
          dots[k].setAttribute("visibility", v == null ? "hidden" : "visible");
          if (v != null) { dots[k].setAttribute("cx", x(xs[i])); dots[k].setAttribute("cy", y(v)); }
        });
        var c = opts.tip(i);
        if (event) showTip(event, c.title, c.rows);
        else showTipAt(dots[0].getAttribute("visibility") === "visible" ? dots[0] : cross, c.title, c.rows);
      }
      function hide() {
        cross.setAttribute("visibility", "hidden");
        dots.forEach(function (dot) { dot.setAttribute("visibility", "hidden"); });
        hideTip();
      }
      function nearest(px) {
        var best = 0, gap = Infinity;
        xs.forEach(function (xv, i) {
          var g = Math.abs(x(xv) - px);
          if (g < gap) { gap = g; best = i; }
        });
        return best;
      }
      overlay.addEventListener("pointermove", function (event) {
        var r = svg.getBoundingClientRect();
        show(nearest(event.clientX - r.left), event);
      });
      overlay.addEventListener("pointerleave", hide);
      overlay.addEventListener("focus", function () { show(current); });
      overlay.addEventListener("blur", hide);
      overlay.addEventListener("keydown", function (event) {
        if (event.key === "ArrowLeft" && current > 0) { show(current - 1); event.preventDefault(); }
        if (event.key === "ArrowRight" && current < xs.length - 1) { show(current + 1); event.preventDefault(); }
      });
    });
  }

  // Forest plot: odds ratio per year with 95% intervals, log scale -------------------------------
  function forest(parent, rows, series) {
    var rowHeight = 30;
    chart(parent, function () { return rows.length * rowHeight + 44; }, "Odds ratio per year", function (svg, W, H) {
      var longest = Math.max.apply(null, rows.map(function (r) { return r.label.length; }));
      var m = { top: 8, right: 16, bottom: 36, left: Math.min(180, 16 + longest * 7) };
      var low = 1, high = 1;
      rows.forEach(function (row) {
        series.forEach(function (sr) {
          if (row[sr.low] != null) low = Math.min(low, row[sr.low]);
          if (row[sr.high] != null) high = Math.max(high, row[sr.high]);
        });
      });
      var l0 = Math.log(low) - 0.05, l1 = Math.log(high) + 0.05;
      var x = function (v) { return linear(l0, l1, m.left, W - m.right)(Math.log(v)); };
      var candidates = [0.25, 0.33, 0.5, 0.67, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 2, 3, 4];
      var shown = candidates.filter(function (v) { return Math.log(v) >= l0 && Math.log(v) <= l1; });
      var plotBottom = H - m.bottom;
      shown.forEach(function (v) {
        s("line", { x1: x(v), x2: x(v), y1: m.top, y2: plotBottom, class: v === 1 ? "axis-line" : "grid-line" }, svg);
        s("text", { x: x(v), y: plotBottom + 16, "text-anchor": "middle", text: String(v) }, svg);
      });
      s("text", { x: (m.left + W - m.right) / 2, y: plotBottom + 32, "text-anchor": "middle",
                  class: "label", text: "← falling   odds ratio per year   rising →" }, svg);
      rows.forEach(function (row, i) {
        var cy = m.top + i * rowHeight + rowHeight / 2;
        s("text", { x: m.left - 10, y: cy + 4, "text-anchor": "end", class: "label", text: row.label }, svg);
        var marks = s("g", { class: "mark" }, svg);
        series.forEach(function (sr, k) {
          var offset = (k - (series.length - 1) / 2) * 8;
          if (row[sr.value] == null) return;
          s("line", { x1: x(row[sr.low]), x2: x(row[sr.high]), y1: cy + offset, y2: cy + offset,
                      class: "whisker", style: "stroke: " + sr.color }, marks);
          s("circle", { cx: x(row[sr.value]), cy: cy + offset, r: 4, class: "dot",
                        style: "fill: " + sr.color }, marks);
        });
        var hit = s("rect", { x: 0, y: cy - rowHeight / 2, width: W, height: rowHeight,
                              fill: "transparent", "aria-label": row.label }, svg);
        hover(hit, marks, function () { return row.tip; });
      });
    });
  }

  // Ranked horizontal bars ------------------------------------------------------------------------
  function hbars(parent, rows, opts) {
    var rowHeight = 28;
    chart(parent, function () { return rows.length * rowHeight + 8; }, opts.label, function (svg, W) {
      var longest = Math.max.apply(null, rows.map(function (r) { return r.label.length; }));
      var m = { left: Math.min(170, 12 + longest * 7), right: 56 };
      var max = Math.max.apply(null, rows.map(function (r) { return r.value; }).concat([1e-9]));
      var x = linear(0, max, m.left, W - m.right);
      var thickness = 16;
      s("line", { x1: m.left, x2: m.left, y1: 0, y2: rows.length * rowHeight, class: "axis-line" }, svg);
      rows.forEach(function (row, i) {
        var top = i * rowHeight + (rowHeight - thickness) / 2;
        s("text", { x: m.left - 8, y: top + thickness / 2 + 4, "text-anchor": "end", class: "label", text: row.label }, svg);
        var bar = s("path", { d: barPath(m.left, top, Math.max(x(row.value) - m.left, 0), thickness),
                              class: "mark", style: "fill: " + color(1) }, svg);
        s("text", { x: x(row.value) + 6, y: top + thickness / 2 + 4, class: "value", text: opts.format(row.value) }, svg);
        var hit = s("rect", { x: 0, y: i * rowHeight, width: W, height: rowHeight, fill: "transparent",
                              "aria-label": row.label }, svg);
        hover(hit, bar, function () { return row.tip; });
      });
    });
  }

  // Diverging bars (phrases: recent against earlier) ---------------------------------------------
  function diverging(parent, rows, opts) {
    var rowHeight = 24;
    chart(parent, function () { return rows.length * rowHeight + 8; }, opts.label, function (svg, W) {
      var longest = Math.max.apply(null, rows.map(function (r) { return r.label.length; }));
      var m = { left: Math.min(240, 12 + longest * 7), right: 40 };
      var max = Math.max.apply(null, rows.map(function (r) { return Math.abs(r.value); }).concat([1e-9]));
      var mid = m.left + 32 + (W - m.left - m.right - 64) / 2;
      var half = (W - m.left - m.right - 64) / 2;
      var thickness = 14;
      s("line", { x1: mid, x2: mid, y1: 0, y2: rows.length * rowHeight, class: "axis-line" }, svg);
      rows.forEach(function (row, i) {
        var top = i * rowHeight + (rowHeight - thickness) / 2;
        var length = (row.value / max) * half;
        s("text", { x: m.left - 8, y: top + thickness / 2 + 4, "text-anchor": "end", class: "label", text: row.label }, svg);
        var bar = s("path", { d: barPath(mid, top, length, thickness), class: "mark",
                              style: "fill: " + (row.value >= 0 ? "var(--diverging-up)" : "var(--diverging-down)") }, svg);
        s("text", { x: mid + length + (row.value >= 0 ? 6 : -6), y: top + thickness / 2 + 4,
                    "text-anchor": row.value >= 0 ? "start" : "end", class: "value", text: fixed(row.value, 1) }, svg);
        var hit = s("rect", { x: 0, y: i * rowHeight, width: W, height: rowHeight, fill: "transparent",
                              "aria-label": row.label }, svg);
        hover(hit, bar, function () { return row.tip; });
      });
    });
  }

  // Topic map: one cluster highlighted, the rest grey (emphasis) ----------------------------------
  function topicMap(parent, map, clusters, onPick) {
    var selected = clusters.length ? clusters[0].cluster : 0;
    var select = h("select", { "aria-label": "Highlighted cluster" });
    clusters.forEach(function (c) {
      select.appendChild(h("option", { value: String(c.cluster),
        text: "Cluster " + c.cluster + ": " + c.terms.slice(0, 3).join(", ") + " (" + count(c.papers) + ")" }));
    });
    parent.appendChild(h("div", { class: "controls" }, [h("label", {}, ["Highlight", select])]));
    var redraw;
    select.addEventListener("change", function () { selected = Number(select.value); redraw(); });
    var box = chart(parent, function (width) { return Math.min(560, Math.max(320, width * 0.62)); },
      "Map of the papers, coloured by topic cluster", function (svg, W, H) {
        var m = 12;
        var xMin = Math.min.apply(null, map.x), xMax = Math.max.apply(null, map.x);
        var yMin = Math.min.apply(null, map.y), yMax = Math.max.apply(null, map.y);
        var x = linear(xMin, xMax, m, W - m), y = linear(yMin, yMax, H - m, m);
        var order = map.x.map(function (_, i) { return i; });
        order.sort(function (a, b) { return (map.cluster[a] === selected) - (map.cluster[b] === selected); });
        var px = [], py = [];
        order.forEach(function (i) {
          var on = map.cluster[i] === selected;
          px[i] = x(map.x[i]); py[i] = y(map.y[i]);
          s("circle", { cx: px[i], cy: py[i], r: on ? 4 : 3, class: on ? "dot" : null,
                        style: "fill: " + (on ? color(1) : "var(--deemphasis)") }, svg);
        });
        clusters.forEach(function (c) {
          var xs = [], ys = [];
          map.cluster.forEach(function (k, i) { if (k === c.cluster) { xs.push(px[i]); ys.push(py[i]); } });
          if (!xs.length) return;
          xs.sort(function (a, b) { return a - b; });
          ys.sort(function (a, b) { return a - b; });
          var mx = xs[Math.floor(xs.length / 2)], my = ys[Math.floor(ys.length / 2)];
          s("text", { x: mx, y: my, "text-anchor": "middle", class: "label halo",
                      "font-weight": c.cluster === selected ? "700" : null, text: String(c.cluster) }, svg);
        });
        var ring = s("circle", { r: 7, fill: "none", stroke: "var(--text-primary)", "stroke-width": 1.5,
                                 visibility: "hidden" }, svg);
        var overlay = s("rect", { x: 0, y: 0, width: W, height: H, fill: "transparent" }, svg);
        function nearest(event) {
          var r = svg.getBoundingClientRect(), qx = event.clientX - r.left, qy = event.clientY - r.top;
          var best = -1, gap = 16 * 16;
          for (var i = 0; i < px.length; i++) {
            var d = (px[i] - qx) * (px[i] - qx) + (py[i] - qy) * (py[i] - qy);
            if (d < gap) { gap = d; best = i; }
          }
          return best;
        }
        overlay.addEventListener("pointermove", function (event) {
          var i = nearest(event);
          if (i < 0) { ring.setAttribute("visibility", "hidden"); hideTip(); overlay.style.cursor = ""; return; }
          ring.setAttribute("cx", px[i]); ring.setAttribute("cy", py[i]);
          ring.setAttribute("visibility", "visible");
          overlay.style.cursor = "pointer";
          var paper = map.row[i] == null ? null : DATA.papers[map.row[i]];
          showTip(event, paper ? paper[1] : "", [
            { value: "Cluster " + map.cluster[i], label: paper && paper[2] ? String(paper[2]) : "" },
          ]);
        });
        overlay.addEventListener("pointerleave", function () { ring.setAttribute("visibility", "hidden"); hideTip(); });
        overlay.addEventListener("click", function (event) {
          var i = nearest(event);
          if (i < 0) return;
          selected = map.cluster[i];
          select.value = String(selected);
          redraw();
          if (onPick) onPick(selected);
        });
      });
    redraw = redraws[redraws.length - 1];
    return box;
  }

  // Sections ---------------------------------------------------------------------------------------
  function header() {
    document.getElementById("title").textContent = DATA.title;
    document.getElementById("meta").textContent =
      "Generated " + DATA.generated + " by informatics " + DATA.version + " (commit " + DATA.commit +
      "). Tables with every number are in the CSV files next to this page.";
    var kpis = h("section", { class: "kpis", "aria-label": "Key numbers" });
    [
      ["Papers analysed", compact(DATA.kpis.papers)],
      ["Candidates (wide keyword net)", compact(DATA.kpis.candidates)],
      ["Topic clusters", String(DATA.kpis.clusters)],
      ["AI-only comparison papers", compact(DATA.kpis.background)],
    ].forEach(function (k) {
      kpis.appendChild(h("div", { class: "kpi" }, [h("div", { class: "label", text: k[0] }), h("div", { class: "value", text: k[1] })]));
    });
    main.appendChild(kpis);
    var notes = h("ul", { class: "notes" }, DATA.selection.map(function (sel) {
      return h("li", { text: sel.set + ": " + count(sel.selected) + " of " + count(sel.candidates) +
        " candidates selected by " + sel.description + "." });
    }));
    notes.appendChild(h("li", { text: "Themes are tagged by " + (DATA.themesSource === "llm"
      ? "a language model reading each title and abstract." : "keyword patterns with exclusions.") }));
    main.appendChild(notes);
    var toc = h("nav", { class: "toc", "aria-label": "Sections" });
    [["years", "Papers per year"], ["themes", "Themes"], ["trends", "Trends"], ["concepts", "Concepts"],
     ["phrases", "Phrases"], ["map", "Topic map"], ["papers", "Papers"]].forEach(function (t) {
      toc.appendChild(h("a", { href: "#" + t[0], text: t[1] }));
    });
    main.appendChild(toc);
  }

  function yearsSection() {
    var y = DATA.years;
    var section = card("years", "Papers per year", "Selected papers by year of first submission (arXiv) or publication (other sources).");
    columns(section, {
      label: "Selected papers per year", labels: y.year, values: y.selected,
      tip: function (i) { return { title: String(y.year[i]), rows: [{ value: count(y.selected[i]), label: "papers" }] }; },
    });
    tableView(section, ["Year", "Papers"], y.year.map(function (year, i) { return [year, count(y.selected[i])]; }), [false, true]);
    var sets = Object.keys(y.share);
    if (!sets.length) return;
    section.appendChild(h("h3", { text: "Share of all papers in the set that were selected", style: "margin-top: 16px" }));
    var series = sets.slice(0, 3).map(function (name, k) { return { name: name, values: y.share[name], color: color(k + 1) }; });
    if (series.length > 1) legend(section, series.map(function (sr) { return { label: sr.name, color: sr.color }; }), "line");
    lines(section, {
      label: "Share of papers selected", x: y.year, series: series, yFormat: function (v) { return pct(v, 1); },
      directLabels: true,
      tip: function (i) {
        return { title: String(y.year[i]), rows: series.map(function (sr) {
          return { value: pct(sr.values[i], 2), label: sr.name, color: sr.color };
        }) };
      },
    });
    tableView(section, ["Year"].concat(sets), y.year.map(function (year, i) {
      return [year].concat(sets.map(function (name) { return pct(y.share[name][i], 2); }));
    }), [false].concat(sets.map(function () { return true; })));
  }

  function themesSection() {
    var section = card("themes", "Themes by year",
      "Share of papers that discuss each theme, with 95% intervals (years with enough papers). Each panel has its own scale.");
    if (!DATA.themes.length) { section.appendChild(h("p", { class: "empty", text: "No theme was found." })); return; }
    var all = [];
    DATA.themes.forEach(function (t) { all = all.concat(t.years); });
    var domain = [Math.min.apply(null, all), Math.max.apply(null, all)];
    var grid = h("div", { class: "grid" });
    section.appendChild(grid);
    DATA.themes.forEach(function (t) {
      var panel = h("div", {}, [h("h3", { text: t.tag })]);
      grid.appendChild(panel);
      if (!t.years.length) return;
      lines(panel, {
        label: t.tag + " by year", x: t.years, domain: domain, height: 150, yTicks: 3,
        series: [{ name: t.tag, values: t.share, low: t.low, high: t.high, color: color(1) }],
        yFormat: function (v) { return pct(v, 0); },
        tip: function (i) {
          return { title: String(t.years[i]), rows: [
            { value: pct(t.share[i]), label: count(t.papers[i]) + " of " + count(t.n[i]) + " papers" },
            { value: pct(t.low[i]) + "–" + pct(t.high[i]), label: "95% interval" },
          ] };
        },
      });
    });
    var rows = [];
    DATA.themes.forEach(function (t) {
      t.years.forEach(function (year, i) {
        rows.push([t.tag, year, count(t.papers[i]), count(t.n[i]), pct(t.share[i]), pct(t.low[i]) + "–" + pct(t.high[i])]);
      });
    });
    tableView(section, ["Theme", "Year", "Papers", "Of", "Share", "95% interval"], rows, [false, false, true, true, true, true]);
  }

  function trendsSection() {
    var section = card("trends", "Trends: odds ratio per year",
      "Logistic regression of whether a paper has the theme on its date. Above 1: the theme grows each year. " +
      "'Within topics' adjusts for the " + DATA.adjustedFor + ": when it is closer to 1 than 'All papers', " +
      "the change comes from some topics growing faster than others.");
    if (!DATA.trends.length) {
      section.appendChild(h("p", { class: "empty", text: "Too few papers per year for trend estimates." }));
      return;
    }
    var series = [
      { label: "All papers", value: "or", low: "low", high: "high", color: color(1) },
      { label: "Within topics", value: "adj", low: "adj_low", high: "adj_high", color: color(2) },
    ];
    legend(section, series, "dot");
    function ci(row, sr) { return fixed(row[sr.value], 2) + " (" + fixed(row[sr.low], 2) + "–" + fixed(row[sr.high], 2) + ")"; }
    var rows = DATA.trends.slice().reverse().map(function (row) {
      return { label: row.tag, or: row.or, low: row.low, high: row.high, adj: row.adj,
        adj_low: row.adj_low, adj_high: row.adj_high,
        tip: { title: row.tag + " (" + count(row.papers) + " papers)", rows: [
          { value: ci(row, series[0]), label: "all papers", color: series[0].color },
          { value: ci(row, series[1]), label: "within topics", color: series[1].color },
          { value: (row.pp >= 0 ? "+" : "") + fixed(row.pp, 2) + " points", label: "per year, within topics" },
        ] } };
    });
    forest(section, rows, series);
    tableView(section, ["Theme", "Papers", "All papers", "Within topics", "Points per year", "q"],
      DATA.trends.map(function (row) {
        return [row.tag, count(row.papers), ci(row, series[0]), ci(row, series[1]), fixed(row.pp, 2), fixed(row.q, 3)];
      }), [false, true, true, true, true, true]);
  }

  function conceptsSection() {
    var section = card("concepts", "Concepts", "What the papers are about: the most common concepts in each dimension.");
    var grid = h("div", { class: "grid wide" });
    section.appendChild(grid);
    var table = [];
    DATA.concepts.forEach(function (panel) {
      var box = h("div", {}, [h("h3", { text: panel.dimension })]);
      grid.appendChild(box);
      if (!panel.rows.length) { box.appendChild(h("p", { class: "empty", text: "None found." })); return; }
      hbars(box, panel.rows.map(function (r) {
        table.push([panel.dimension, r.tag, count(r.papers), pct(r.share), pct(r.low) + "–" + pct(r.high)]);
        return { label: r.tag, value: r.share, tip: { title: r.tag, rows: [
          { value: pct(r.share), label: count(r.papers) + " papers" },
          { value: pct(r.low) + "–" + pct(r.high), label: "95% interval" },
        ] } };
      }), { label: panel.dimension, format: function (v) { return pct(v, 0); } });
    });
    tableView(section, ["Dimension", "Concept", "Papers", "Share", "95% interval"], table, [false, false, true, true, true]);
  }

  function phrasesSection() {
    var p = DATA.phrases;
    if (!p) return;
    var recent = p.recent[0] + "–" + p.recent[1], earlier = p.earlier[0] + "–" + p.earlier[1];
    var section = card("phrases", "Rising and falling phrases",
      "Phrases more common in " + recent + " than in " + earlier + " (and the reverse), ranked by the z-score of a log-odds ratio.");
    legend(section, [
      { label: "More common in " + recent, color: "var(--diverging-up)" },
      { label: "More common in " + earlier, color: "var(--diverging-down)" },
    ], "line");
    diverging(section, p.rows.map(function (r) {
      return { label: r.phrase, value: r.z, tip: { title: r.phrase, rows: [
        { value: pct(r.recent), label: "of papers in " + recent },
        { value: pct(r.earlier), label: "of papers in " + earlier },
        { value: fixed(r.z, 1), label: "z" },
      ] } };
    }), { label: "Rising and falling phrases" });
    tableView(section, ["Phrase", recent, earlier, "z"], p.rows.map(function (r) {
      return [r.phrase, pct(r.recent), pct(r.earlier), fixed(r.z, 1)];
    }), [false, true, true, true]);
  }

  function mapSection(filterTable) {
    var section = card("map", "Topic map",
      "Each dot is a paper, placed by t-SNE so that papers with similar text sit together; numbers mark the clusters. " +
      "Distances between distant groups mean little. Click a dot to highlight its cluster and list its papers below.");
    if (DATA.map) topicMap(section, DATA.map, DATA.clusters, filterTable);
    tableView(section, ["Cluster", "Papers", "Share", "Distinctive terms"], DATA.clusters.map(function (c) {
      return [c.cluster, count(c.papers), pct(c.share), c.terms.join(", ")];
    }), [false, true, true, false]);
  }

  function papersSection() {
    var section = card("papers", "Papers", DATA.papersTotal > DATA.papers.length
      ? "A random sample of " + count(DATA.papers.length) + " of the " + count(DATA.papersTotal) + " papers; papers.csv lists all."
      : "Every analysed paper. Search the titles or narrow the list by year, cluster and theme.");
    var rows = DATA.papers;
    var years = rows.map(function (r) { return r[2]; }).filter(function (v) { return v != null; });
    var yearOptions = [];
    if (years.length) {
      for (var yv = Math.min.apply(null, years); yv <= Math.max.apply(null, years); yv++) yearOptions.push(yv);
    }
    function options(select, values, labeller, allText) {
      select.appendChild(h("option", { value: "", text: allText }));
      values.forEach(function (v) { select.appendChild(h("option", { value: String(v), text: labeller(v) })); });
      return select;
    }
    var search = h("input", { type: "search", placeholder: "Search titles or IDs", "aria-label": "Search titles or IDs" });
    var from = options(h("select", { "aria-label": "From year" }), yearOptions, String, "from any year");
    var to = options(h("select", { "aria-label": "To year" }), yearOptions, String, "to any year");
    var cluster = options(h("select", { "aria-label": "Cluster" }), DATA.clusters.map(function (c) { return c.cluster; }),
      function (c) { return "Cluster " + c; }, "all clusters");
    var theme = options(h("select", { "aria-label": "Theme" }), DATA.themeNames.map(function (_, i) { return i; }),
      function (i) { return DATA.themeNames[i]; }, "all themes");
    var shown = h("span", { class: "sub" });
    section.appendChild(h("div", { class: "controls" }, [search, from, to, cluster, theme, shown]));
    var multiSet = DATA.sets.length > 1;
    var headers = ["Year", "Title", "Cluster", "Themes", "Publication"].concat(multiSet ? ["Sets"] : []);
    var tbody = h("tbody");
    var table = h("table", { class: "papers" }, [
      h("thead", {}, [h("tr", {}, headers.map(function (t, i) { return h("th", { text: t, scope: "col", class: i === 0 ? "num" : null }); }))]),
      tbody,
    ]);
    section.appendChild(h("div", { class: "table-wrap", style: "max-height: none" }, [table]));
    var prev = h("button", { class: "button", type: "button", text: "Previous" });
    var next = h("button", { class: "button", type: "button", text: "Next" });
    var page = h("span");
    section.appendChild(h("div", { class: "pager" }, [prev, page, next]));
    var perPage = 50, start = 0, matches = rows;

    function render() {
      while (tbody.firstChild) tbody.removeChild(tbody.firstChild);
      matches.slice(start, start + perPage).forEach(function (r) {
        var title = h("td", { class: "title" });
        var url = safeUrl(r[6]);
        if (url) title.appendChild(h("a", { href: url, target: "_blank", rel: "noopener noreferrer", text: r[1] || r[0] }));
        else title.appendChild(document.createTextNode(r[1] || r[0]));
        title.appendChild(h("div", { class: "sub", text: r[0], style: "margin: 0" }));
        tbody.appendChild(h("tr", {}, [
          h("td", { class: "num", text: r[2] == null ? "" : String(r[2]) }),
          title,
          h("td", { text: String(r[3]) }),
          h("td", { text: r[4].map(function (i) { return DATA.themeNames[i]; }).join(", ") }),
          h("td", { text: r[5] || "" }),
        ].concat(multiSet ? [h("td", { text: r[7] })] : [])));
      });
      var end = Math.min(start + perPage, matches.length);
      shown.textContent = count(matches.length) + " of " + count(rows.length) + " papers";
      page.textContent = matches.length ? (start + 1) + "–" + end + " of " + count(matches.length) : "No papers match.";
      prev.disabled = start === 0;
      next.disabled = end >= matches.length;
    }
    function apply() {
      var q = search.value.trim().toLowerCase();
      var y0 = from.value ? Number(from.value) : -Infinity, y1 = to.value ? Number(to.value) : Infinity;
      var c = cluster.value === "" ? null : Number(cluster.value);
      var t = theme.value === "" ? null : Number(theme.value);
      matches = rows.filter(function (r) {
        if (q && (r[1] || "").toLowerCase().indexOf(q) < 0 && r[0].toLowerCase().indexOf(q) < 0) return false;
        if (r[2] != null && (r[2] < y0 || r[2] > y1)) return false;
        if (r[2] == null && (from.value || to.value)) return false;
        if (c != null && r[3] !== c) return false;
        if (t != null && r[4].indexOf(t) < 0) return false;
        return true;
      });
      start = 0;
      render();
    }
    var timer;
    search.addEventListener("input", function () { clearTimeout(timer); timer = setTimeout(apply, 150); });
    [from, to, cluster, theme].forEach(function (control) { control.addEventListener("change", apply); });
    prev.addEventListener("click", function () { start = Math.max(0, start - perPage); render(); });
    next.addEventListener("click", function () { start += perPage; render(); });
    render();
    return function (clusterId) { cluster.value = String(clusterId); apply(); };
  }

  // Theme toggle -----------------------------------------------------------------------------------
  function themeToggle() {
    var button = document.getElementById("theme-toggle");
    function isDark() {
      var set = document.documentElement.getAttribute("data-theme");
      if (set) return set === "dark";
      return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    function label() {
      button.textContent = isDark() ? "Light theme" : "Dark theme";
      button.setAttribute("aria-pressed", String(isDark()));
    }
    button.addEventListener("click", function () {
      document.documentElement.setAttribute("data-theme", isDark() ? "light" : "dark");
      label();
    });
    label();
  }

  header();
  yearsSection();
  themesSection();
  trendsSection();
  conceptsSection();
  phrasesSection();
  var filterTable = null;
  mapSection(function (clusterId) { if (filterTable) filterTable(clusterId); });
  filterTable = papersSection();
  themeToggle();

  var pending;
  window.addEventListener("resize", function () {
    clearTimeout(pending);
    pending = setTimeout(function () { redraws.forEach(function (draw) { draw(); }); }, 150);
  });
})();
