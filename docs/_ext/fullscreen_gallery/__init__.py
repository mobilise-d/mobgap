"""
A Sphinx extension that adds fullscreen buttons to gallery outputs.
"""

from pathlib import Path

from bs4 import BeautifulSoup
from sphinx.util import logging
from sphinx.util.fileutil import copy_asset

logger = logging.getLogger(__name__)


def setup(app):
    """Setup the extension"""
    # Add static files
    app.add_css_file("gallery_fullscreen.css")
    app.add_js_file("gallery_fullscreen.js")

    # Add our custom post-processing function
    app.connect("build-finished", add_fullscreen_buttons)

    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}


def add_fullscreen_buttons(app, exception):
    """Add fullscreen buttons to gallery outputs after build"""
    if exception is not None:
        return

    if app.builder.name != "html":
        return

    # Copy our static files
    static_dir = Path(__file__).parent / "static"
    copy_asset(str(static_dir / "gallery_fullscreen.css"), str(Path(app.outdir) / "_static"))
    copy_asset(str(static_dir / "gallery_fullscreen.js"), str(Path(app.outdir) / "_static"))

    # Process HTML files
    for html_file in Path(app.outdir).rglob("*.html"):
        process_html_file(html_file)


def process_html_file(html_file):
    """Add fullscreen buttons to a single HTML file"""
    content = html_file.read_text()

    # Don't process files that don't contain gallery outputs
    if "output_html rendered_html" not in content and "sphx-glr-single-img" not in content:
        return

    # Add fullscreen buttons to images and tables
    modified = add_buttons_to_content(content)

    html_file.write_text(modified)


def add_buttons_to_content(content):
    """Add fullscreen buttons to gallery outputs in HTML content"""
    soup = BeautifulSoup(content, "html.parser")

    # Add buttons to tables
    for div in soup.find_all("div", class_="output_subarea output_html rendered_html output_result"):
        div["class"].append("gallery-fullscreen-wrapper")
        button = soup.new_tag("button", attrs={"class": "gallery-fullscreen-btn", "title": "View fullscreen"})
        button.string = "⤢"
        div.insert(0, button)

    # Add buttons to images
    for img in soup.find_all("img", class_="sphx-glr-single-img"):
        wrapper = soup.new_tag("div", attrs={"class": "gallery-fullscreen-wrapper"})
        button = soup.new_tag("button", attrs={"class": "gallery-fullscreen-btn", "title": "View fullscreen"})
        button.string = "⤢"
        img.wrap(wrapper)
        wrapper.insert(0, button)

    return str(soup)
