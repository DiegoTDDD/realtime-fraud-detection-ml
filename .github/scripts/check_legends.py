"""Open the dashboard in a real browser (dark and light mode) and measure the contrast of every chart legend.

Fails when a legend's text has a contrast ratio below 4.5:1 against what is actually drawn behind it
(the legend box composited over the chart background), which is the WCAG AA minimum for normal text.
Reasons are printed as ::error:: annotations, which are visible on the public run page.
"""
import re
import sys
import time

from playwright.sync_api import sync_playwright

URL = "http://127.0.0.1:8501"
MIN_RATIO = 4.5

JS = """
() => {
  const out = [];
  document.querySelectorAll('.js-plotly-plot').forEach((plot, i) => {
    const legend = plot.querySelector('g.legend');
    if (!legend) return;
    const cs = (el) => (el ? getComputedStyle(el) : null);
    const bg = legend.querySelector('rect.bg');
    const paper = plot.querySelector('svg.main-svg rect.bg');
    const txt = legend.querySelector('text.legendtext');
    out.push({
      chart: i,
      names: [...legend.querySelectorAll('text.legendtext')].map((t) => t.textContent),
      legendBg: bg ? cs(bg).fill : null,
      legendBgOpacity: bg ? cs(bg).fillOpacity : null,
      paperBg: paper ? cs(paper).fill : null,
      paperBgOpacity: paper ? cs(paper).fillOpacity : null,
      text: txt ? cs(txt).fill : null,
      textOpacity: txt ? cs(txt).opacity : null,
    });
  });
  return out;
}
"""


def parse(color):
    m = re.search(r"rgba?\(([^)]+)\)", color or "")
    if not m:
        return None
    p = [float(x) for x in m.group(1).split(",")]
    return (p[0], p[1], p[2], p[3] if len(p) > 3 else 1.0)


def over(fg, alpha, bg):
    return tuple(fg[i] * alpha + bg[i] * (1 - alpha) for i in range(3))


def lin(c):
    c /= 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def lum(rgb):
    return 0.2126 * lin(rgb[0]) + 0.7152 * lin(rgb[1]) + 0.0722 * lin(rgb[2])


def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def measure(item):
    """Contrast of the legend text against the legend box drawn over the chart background."""
    paper = parse(item["paperBg"]) or (255, 255, 255, 1.0)
    paper_alpha = float(item["paperBgOpacity"] or 1) * paper[3]
    base = over(paper[:3], paper_alpha, (255, 255, 255)) if paper_alpha < 1 else paper[:3]
    box = parse(item["legendBg"]) or (0, 0, 0, 0.0)
    box_alpha = float(item["legendBgOpacity"] or 1) * box[3]
    behind = over(box[:3], box_alpha, base)
    txt = parse(item["text"]) or (0, 0, 0, 1.0)
    txt_alpha = float(item["textOpacity"] or 1) * txt[3]
    fg = over(txt[:3], txt_alpha, behind)
    return ratio(fg, behind), behind, fg


def selftest():
    dark = {"paperBg": "rgb(22, 27, 34)", "paperBgOpacity": "1", "legendBg": "rgb(0, 0, 0)",
            "legendBgOpacity": "0", "text": "rgb(201, 209, 217)", "textOpacity": "1"}
    bad = dict(dark, legendBg="rgb(255, 255, 255)", legendBgOpacity="0.5")
    assert measure(dark)[0] > 10, measure(dark)
    assert measure(bad)[0] < MIN_RATIO, measure(bad)


selftest()
failures = []
found = 0
with sync_playwright() as p:
    browser = p.chromium.launch()
    for scheme in ("dark", "light"):
        ctx = browser.new_context(color_scheme=scheme, viewport={"width": 1400, "height": 1000})
        page = ctx.new_page()
        page.goto(URL, wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector(".js-plotly-plot", timeout=120000)
        time.sleep(8)  # let every chart finish drawing
        items = page.evaluate(JS)
        if not items:
            failures.append(f"{scheme} mode: no chart legend found on the page")
        for it in items:
            found += 1
            r, behind, fg = measure(it)
            line = (f"{scheme} mode, chart {it['chart']}, legend {it['names']}: contrast {r:.1f}:1 "
                    f"(text rgb{tuple(round(x) for x in fg)} on rgb{tuple(round(x) for x in behind)})")
            print(line)
            if r < MIN_RATIO:
                failures.append(line + f" is below {MIN_RATIO}:1")
        ctx.close()
    browser.close()

for f in failures:
    print(f"::error::{f}")
if failures:
    sys.exit(1)
print(f"OK - {found} legends checked in dark and light mode, all readable")
