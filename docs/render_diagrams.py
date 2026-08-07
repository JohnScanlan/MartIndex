#!/usr/bin/env python3
"""
Re-render docs/*.png from docs/diagram-source.html.

Edit the HTML (plain SVG, literal colours), then:
    .venv/bin/python docs/render_diagrams.py

Uses the Chromium that ships with playwright, so the PNGs match the browser
exactly. Update the row counts in the HTML when the data grows.
"""
from pathlib import Path
from playwright.sync_api import sync_playwright

DOCS = Path(__file__).parent
SHEETS = [("sheet-in", "pipeline-data-in.png"), ("sheet-out", "pipeline-data-out.png")]

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 1300, "height": 900}, device_scale_factor=2)
    page.goto(f"file://{DOCS / 'diagram-source.html'}")
    page.wait_for_timeout(1200)          # let fonts and layout settle
    for element_id, name in SHEETS:
        page.locator(f"#{element_id}").screenshot(path=str(DOCS / name))
        print(f"wrote docs/{name}")
    browser.close()
