#!/usr/bin/env python
"""Load the live demo page in headless Chromium and check what it displays.

    python scripts/check_site.py                 build, serve, check (make site-check)
    python scripts/check_site.py --build _site   only assemble the site (pages.yml)

tests/test_demo_data.py proves the committed per-transaction export agrees
with the committed metrics and business case. It does not run the page's
JavaScript, which re-implements the lane rule by hand (lanesFor). This script
does: it assembles the site from the committed files exactly as pages.yml
deploys it, serves it on localhost, and drives the real page.

  - The single knob: at every committed threshold (0.5 to 1.0) the review
    count, the confusion table and the precision tile must equal the committed
    coverage_precision_curve row, and at the business case's two thresholds the
    false alerts and fraud Amount missed must equal policies A and C.
  - The split-cut mode: with block = t and approve = 1 - t the counts must
    equal the same committed row (ADR 008's generalization property), and the
    two sliders' ranges must make overlapping cuts impossible.
  - The headline tiles against business_case_ulb_creditcard.json, the analyst
    hours box, and the single-transaction viewer against the export.
  - No JavaScript error, no load failure, and no horizontal page scroll at a
    390 px phone width or at 1280 px.

This checks consistency with the committed files. CI has no ULB data, so it
cannot check that the scores came from the committed code; `make reproduce`
does that locally.

Needs `pip install playwright` and `python -m playwright install chromium`.
Uses only the standard library otherwise.
"""
from __future__ import annotations

import argparse
import functools
import json
import re
import shutil
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
ART = REPO / "artifacts"
DATA_FILES = ["demo_scores_ulb_creditcard.json", "business_case_ulb_creditcard.json"]


def build_site(dest: Path) -> Path:
    """What pages.yml publishes: the page plus the committed data files."""
    (dest / "data").mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "site" / "index.html", dest / "index.html")
    for name in DATA_FILES:
        shutil.copy(ART / name, dest / "data" / name)
    return dest


def nums(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]


def num(text: str) -> float:
    found = nums(text)
    if not found:
        raise ValueError(f"no number in {text!r}")
    return found[0]


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.checks = 0

    def eq(self, what: str, got: Any, want: Any) -> None:
        self.checks += 1
        if got != want:
            self.failures.append(f"{what}: page shows {got!r}, expected {want!r}")

    def near(self, what: str, got: float, want: float, tol: float) -> None:
        self.checks += 1
        if abs(got - want) > tol + 1e-9:
            self.failures.append(f"{what}: page shows {got}, expected {want} (+/- {tol})")


def set_input(page, sel: str, value: float) -> None:
    page.eval_on_selector(
        sel, "(el, v) => { el.value = v; el.dispatchEvent(new Event('input')); }", repr(value)
    )


def tx(page, i: str) -> str:
    return page.text_content(f"#{i}") or ""


def check_counts(page, ck: Checker, where: str, row: dict[str, Any]) -> None:
    ff, fr, fc = (int(num(tx(page, i))) for i in ("c-ff", "c-fr", "c-fc"))
    lf, lr, lc = (int(num(tx(page, i))) for i in ("c-lf", "c-lr", "c-lc"))
    ck.eq(f"{where} blocked fraud", ff, row["fraud_caught_auto"])
    ck.eq(f"{where} fraud in review", fr, row["fraud_in_review_queue"])
    ck.eq(f"{where} fraud approved", fc, row["fraud_missed"])
    ck.eq(f"{where} blocked total", ff + lf, row["n_flagged_fraud"])
    ck.eq(f"{where} sent to review", fr + lr, row["n_sent_to_review"])
    ck.eq(f"{where} review lane label", int(num(tx(page, "n-review"))), row["n_sent_to_review"])
    ck.eq(f"{where} missed tile", nums(tx(page, "m-miss"))[0], row["fraud_missed"])
    prec = tx(page, "m-prec")
    if row["precision_on_flagged"] is None:
        ck.eq(f"{where} precision tile", prec, "nothing blocked")
    else:
        ck.near(f"{where} precision tile", num(prec), 100 * row["precision_on_flagged"], 0.05)
    ck.eq(f"{where} lanes add up", ff + fr + fc + lf + lr + lc, 56961)


def check_knob(page, ck: Checker, metrics_curve, business) -> None:
    for row in metrics_curve:
        t = row["review_threshold"]
        set_input(page, "#t", t)
        ck.eq(f"slider reaches {t}", float(page.input_value("#t")), t)
        check_counts(page, ck, f"knob {t}", row)
        pol = None
        if t == 0.5:
            pol = business["policies"]["A_single_threshold_0.5"]
        elif t == business["operating_threshold"]:
            pol = business["policies"][f"C_review_band_{t}"]
        if pol:
            ck.eq(f"knob {t} false alerts vs business case",
                  int(num(tx(page, "c-lf"))), pol["counts"]["false_alerts"])
            ck.near(f"knob {t} false alerts per 10k", num(tx(page, "m-fa")),
                    pol["per_10k_transactions"]["false_alerts"], 0.051)
            ck.near(f"knob {t} fraud amount missed", num(tx(page, "m-amt-miss")),
                    pol["amount"]["fraud_amount_missed"], 0.005)
            ck.near(f"knob {t} legit amount blocked", num(tx(page, "m-legit-amt")),
                    pol["amount"]["legit_amount_auto_flagged"], 0.005)
        if t == business["operating_threshold"]:
            set_input(page, "#mins", 3)
            review_per_10k = row["n_sent_to_review"] * 10000 / 56961
            ck.near("analyst hours at 3 min", num(tx(page, "hours")),
                    review_per_10k * 3 / 60, 0.051)


def check_split(page, ck: Checker, metrics_curve) -> None:
    fmin = float(page.get_attribute("#flag", "min") or "nan")
    cmax = float(page.get_attribute("#clear", "max") or "nan")
    ck.eq("approve cut cannot reach the block cut", cmax < fmin, True)
    page.check("#split")
    ck.eq("raw-score caveat visible", page.is_visible("#split-note"), True)
    for row in metrics_curve:
        t = row["review_threshold"]
        if not 0.5 < t < 1.0:
            continue
        set_input(page, "#flag", t)
        set_input(page, "#clear", round(1 - t, 3))
        check_counts(page, ck, f"split block {t} / approve {round(1 - t, 3)}", row)
    page.uncheck("#split")


def check_headline(page, ck: Checker, b) -> None:
    t = b["operating_threshold"]
    pol = b["policies"]
    a = pol["A_single_threshold_0.5"]
    bb, c = pol[f"B_single_threshold_{t}"], pol[f"C_review_band_{t}"]
    got = nums(tx(page, "h-fa"))
    ck.eq("headline false alerts A -> B", got,
          [round(a["per_10k_transactions"]["false_alerts"], 1),
           round(bb["per_10k_transactions"]["false_alerts"], 1)])
    ck.near("headline C false alerts", nums(tx(page, "h-fa-ci"))[-1],
            c["per_10k_transactions"]["false_alerts"], 0.051)
    ck.eq("headline silent misses A -> B -> C", nums(tx(page, "h-miss")),
          [a["counts"]["fraud_missed"], bb["counts"]["fraud_missed"], c["counts"]["fraud_missed"]])
    mci = b["bootstrap"]["intervals_95"]["paired_A_minus_C_frauds_cleared_without_review"]
    ck.eq("headline paired CI on misses", nums(tx(page, "h-miss-ci"))[-2:],
          [mci["ci_lower"], mci["ci_upper"]])
    ck.near("headline review queue", num(tx(page, "h-cost")),
            c["per_10k_transactions"]["review_queue"], 0.5)
    cost = nums(tx(page, "h-cost-ci"))
    # "... 16 frauds in 4,039 reviews, about 1 in 252."
    ck.eq("headline queue hit rate", cost[-4:],
          [c["counts"]["fraud_in_review"], c["counts"]["sent_to_review"], 1,
           round(c["counts"]["sent_to_review"] / c["counts"]["fraud_in_review"])])


def check_viewer(page, ck: Checker, demo) -> None:
    fraud = set(demo["fraud_rows"])
    verdicts = ["approved automatically", "sent to a human", "blocked automatically"]
    for pick in ["fraud", "any"] * 4:
        page.click(f"[data-pick={pick}]")
        text = tx(page, "txn")
        m = re.search(r"row (\d+) of .*?fraud score ([\d.]+)", text)
        if not m:
            ck.eq("viewer text", text, "row ... fraud score ...")
            continue
        i = int(m.group(1))
        p = demo["p_fraud"][i]
        ck.near(f"viewer row {i} score", float(m.group(2)), p, 0.00005)
        conf = max(p, 1 - p)
        lane = 1 if conf < 0.95 else (2 if p >= 0.5 else 0)
        ck.eq(f"viewer row {i} verdict", verdicts[lane] in text, True)
        ck.eq(f"viewer row {i} truth", ("really fraud" in text), i in fraud)
        if pick == "fraud":
            ck.eq(f"viewer fraud pick {i} is a fraud", i in fraud, True)


def serve(root: Path) -> tuple[ThreadingHTTPServer, str]:
    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *args: Any) -> None:
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}/"


def run(root: Path, screenshots: Path | None) -> int:
    from playwright.sync_api import sync_playwright

    demo, business = (json.loads((ART / n).read_text()) for n in DATA_FILES)
    curve = json.loads((ART / "metrics_ulb_creditcard.json").read_text())[
        "coverage_precision_curve"]
    ck = Checker()
    httpd, url = serve(root)
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            for width, height in [(390, 844), (1280, 900)]:
                page = browser.new_page(viewport={"width": width, "height": height})
                errors: list[str] = []
                page.on("pageerror", lambda e, errors=errors: errors.append(str(e)))
                page.on("console", lambda m, errors=errors: m.type == "error"
                        and errors.append(m.text))
                page.goto(url)
                page.wait_for_function("document.getElementById('n-review').textContent !== ''")
                ck.eq(f"{width}px load error banner", page.locator(".err").count(), 0)
                if width == 390:
                    check_headline(page, ck, business)
                    check_knob(page, ck, curve, business)
                    set_input(page, "#t", business["operating_threshold"])
                    check_viewer(page, ck, demo)
                    check_split(page, ck, curve)
                over = page.evaluate(
                    "document.documentElement.scrollWidth - document.documentElement.clientWidth"
                )
                ck.eq(f"{width}px horizontal overflow (px)", over, 0)
                if screenshots:
                    screenshots.mkdir(parents=True, exist_ok=True)
                    page.screenshot(path=str(screenshots / f"ledgersentry_{width}.png"),
                                    full_page=True)
                ck.eq(f"{width}px JavaScript errors", errors, [])
                page.close()
            browser.close()
    finally:
        httpd.shutdown()
    for f in ck.failures:
        print(f"FAIL: {f}")
    if ck.failures:
        print(f"{len(ck.failures)} of {ck.checks} page checks failed")
        return 1
    print(f"PASS: {ck.checks} checks of what the page displays, against the committed artifacts")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Headless check of the live demo page.")
    ap.add_argument("--build", type=Path, help="only assemble the site into this directory")
    ap.add_argument("--screenshots", type=Path, help="save full-page screenshots here")
    args = ap.parse_args(argv)
    if args.build:
        build_site(args.build)
        print(f"[site ] assembled {args.build}")
        return 0
    with tempfile.TemporaryDirectory() as tmp:
        return run(build_site(Path(tmp)), args.screenshots)


if __name__ == "__main__":
    sys.exit(main())
