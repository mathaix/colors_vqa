import os
import xml.etree.ElementTree as ET


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGO_PATH = os.path.join(REPO_ROOT, "logo.svg")
README_PATH = os.path.join(REPO_ROOT, "README.md")
SVG_NS = "{http://www.w3.org/2000/svg}"


def load_logo():
    return ET.parse(LOGO_PATH).getroot()


def test_logo_is_valid_svg_with_viewbox():
    root = load_logo()
    assert root.tag == SVG_NS + "svg"
    assert root.get("viewBox") == "0 0 200 200"


def test_logo_has_four_distinct_colour_swatches():
    circles = [
        c for c in load_logo().iter(SVG_NS + "circle")
        if c.get("r") == "10"
    ]
    fills = [c.get("fill") for c in circles]
    assert len(fills) == 4
    assert len(set(fills)) == 4


def test_readme_displays_logo():
    with open(README_PATH, encoding="utf-8") as f:
        readme = f.read()
    assert '<img src="logo.svg"' in readme
    assert os.path.exists(LOGO_PATH)
