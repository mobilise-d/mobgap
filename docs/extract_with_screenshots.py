#!/usr/bin/env python3
# /// script
# dependencies = [
#     "beautifulsoup4>=4.9.0",
#     "playwright>=1.40.0",
#     "lxml>=4.6.0",
#     "css_inline>=0.10.0",
# ]
# ///
"""
Enhanced script to extract all images and HTML tables from the revalidation documentation
and generate screenshots of the tables.

This script will:
1. Find all images referenced in the HTML files
2. Copy them to an organized output directory
3. Extract HTML tables and save them as separate HTML files
4. Generate screenshots of all tables using Playwright
"""

import argparse
import asyncio
import re
import shutil
from pathlib import Path

from bs4 import BeautifulSoup
from css_inline import inline
from playwright.async_api import async_playwright


def setup_directories(output_dir: Path, categories: list[str]):
    """Create output directory structure organized by algorithm groups."""
    for category in categories:
        (output_dir / category / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / category / "tables").mkdir(parents=True, exist_ok=True)
        (output_dir / category / "screenshots").mkdir(parents=True, exist_ok=True)


def find_html_files(docs_dir: Path) -> list[Path]:
    """Find all HTML files in the revalidation directory."""
    revalidation_dir = docs_dir / "_build/html/auto_revalidation"
    html_files = []

    if revalidation_dir.exists():
        html_files = list(revalidation_dir.rglob("*.html"))

    return html_files


def extract_images_from_html(html_file: Path, docs_dir: Path, output_dir: Path):
    """Extract and copy images referenced in an HTML file."""
    with open(html_file, encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    # Find all img tags
    images = soup.find_all("img")
    copied_images = []

    for img in images:
        src = img.get("src")
        if not src:
            continue

        # Handle relative paths
        if src.startswith("../"):
            # Convert relative path to absolute path
            img_path = docs_dir / "_build/html" / src.replace("../", "")
        elif src.startswith("_images/"):
            img_path = docs_dir / "_build/html" / src
        else:
            # Try to find the image
            img_path = docs_dir / "_build/html/_images" / Path(src).name

        if img_path.exists():
            # Create organized subdirectory based on source HTML file structure
            rel_path = html_file.relative_to(docs_dir / "_build/html/auto_revalidation")
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "general"

            output_subdir = output_dir / category / "images"
            output_subdir.mkdir(parents=True, exist_ok=True)

            dest_path = output_subdir / img_path.name
            shutil.copy2(img_path, dest_path)
            copied_images.append((img_path.name, str(dest_path)))

    return copied_images


def extract_table_styles(soup):
    """Extract CSS styles related to tables from the HTML."""
    styles = []

    # Find style tags and extract table-related CSS
    style_tags = soup.find_all("style")
    for style_tag in style_tags:
        if style_tag.string:
            # Look for table-related CSS rules
            css_content = style_tag.string
            # Extract table-specific styles using regex
            table_patterns = [
                r"[^}]*\.dataframe[^}]*{[^}]*}",
                r"[^}]*table[^}]*{[^}]*}",
                r"[^}]*thead[^}]*{[^}]*}",
                r"[^}]*tbody[^}]*{[^}]*}",
                r"[^}]*th[^}]*{[^}]*}",
                r"[^}]*td[^}]*{[^}]*}",
                r"[^}]*#T_[^}]*{[^}]*}",  # Specific table IDs
            ]

            for pattern in table_patterns:
                matches = re.findall(pattern, css_content, re.IGNORECASE | re.MULTILINE)
                styles.extend(matches)

    return "\n".join(styles)


def extract_tables_from_html(html_file: Path, output_dir: Path):
    """Extract HTML tables from an HTML file."""
    with open(html_file, encoding="utf-8") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")

    # Find all tables
    tables = soup.find_all("table")
    extracted_tables = []

    if not tables:
        return extracted_tables

    # Create organized subdirectory based on source HTML file
    rel_path = html_file.relative_to(Path("_build/html/auto_revalidation"))
    category = rel_path.parts[0] if len(rel_path.parts) > 1 else "general"

    output_subdir = output_dir / category / "tables"
    output_subdir.mkdir(parents=True, exist_ok=True)

    for i, table in enumerate(tables, 1):
        # Get context around the table (previous heading, etc.)
        context_elements = []

        # Look for preceding headings
        current = table.find_previous(["h1", "h2", "h3", "h4", "h5", "h6"])
        if current:
            context_elements.append(str(current))

        # Look for preceding paragraphs (description)
        prev_p = table.find_previous("p")
        if prev_p and current and prev_p.sourcepos > current.sourcepos if hasattr(prev_p, "sourcepos") else True:
            context_elements.append(str(prev_p))

        # Create standalone HTML for the table
        table_html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Table from {html_file.name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: white; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background: white; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        .context {{ margin-bottom: 20px; color: #666; }}
        /* Copy existing table styles */
        {extract_table_styles(soup)}
    </style>
</head>
<body>
    <div class="context">
        <p><strong>Source:</strong> {html_file.name}</p>
        {"".join(context_elements)}
    </div>
    {table!s}
</body>
</html>
"""

        # Inline all CSS styles
        inlined_html = inline(table_html)

        # Save table HTML
        table_filename = f"{html_file.stem}_table_{i}.html"
        table_path = output_subdir / table_filename

        with open(table_path, "w", encoding="utf-8") as f:
            f.write(inlined_html)

        extracted_tables.append((table_filename, str(table_path)))

    return extracted_tables


async def generate_table_screenshots(output_dir: Path):
    """Generate screenshots of HTML tables using Playwright."""
    try:
        async with async_playwright() as p:
            # Launch browser
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Set viewport for consistent screenshots
            await page.set_viewport_size({"width": 1200, "height": 800})

            # Iterate through algorithm groups
            for category_dir in output_dir.iterdir():
                if category_dir.is_dir():
                    tables_dir = category_dir / "tables"
                    screenshots_dir = category_dir / "screenshots"

                    if tables_dir.exists():
                        for table_file in tables_dir.glob("*.html"):
                            try:
                                # Load the HTML file
                                file_url = f"file://{table_file.absolute()}"
                                await page.goto(file_url)

                                # Wait for the page to load
                                await page.wait_for_timeout(1000)

                                # Find the table element and take a screenshot
                                table_element = await page.query_selector("table")
                                if table_element:
                                    screenshot_path = screenshots_dir / f"{table_file.stem}.png"
                                    await table_element.screenshot(path=str(screenshot_path))
                                    print(f"Screenshot saved: {screenshot_path}")
                                else:
                                    # Fallback: screenshot the whole page
                                    screenshot_path = screenshots_dir / f"{table_file.stem}.png"
                                    await page.screenshot(path=str(screenshot_path))
                                    print(f"Full page screenshot saved: {screenshot_path}")

                            except Exception as e:
                                print(f"Error generating screenshot for {table_file}: {e}")

            await browser.close()

    except Exception as e:
        print(f"Error setting up screenshot generation: {e}")
        print("Make sure to install playwright browsers with: playwright install")


def create_summary_report(output_dir: Path, extracted_data: dict):
    """Create a summary report of extracted assets."""
    report_content = f"""# Revalidation Assets Extraction Report

## Summary
- **Total HTML files processed:** {extracted_data["total_files"]}
- **Total images extracted:** {extracted_data["total_images"]}
- **Total tables extracted:** {extracted_data["total_tables"]}

## Directory Structure
```
{output_dir.name}/
├── cadence/                    # Cadence analysis assets
│   ├── images/                 # Plots and figures
│   ├── tables/                 # HTML tables  
│   └── screenshots/            # Table screenshots
├── stride_length/              # Stride length analysis assets
│   ├── images/
│   ├── tables/
│   └── screenshots/
├── laterality/                 # Laterality analysis assets
│   ├── images/
│   ├── tables/
│   └── screenshots/
├── (other algorithm groups...)
└── extraction_report.md        # This file
```

## Assets by Algorithm Group
"""

    # Add assets by category
    for category in sorted(extracted_data["images_by_category"].keys()):
        report_content += f"\n### {category}\n"

        if category in extracted_data["images_by_category"]:
            images = extracted_data["images_by_category"][category]
            report_content += f"**Images ({len(images)}):**\n"
            for img_name, img_path in images:
                report_content += f"- {img_name}\n"

        if category in extracted_data["tables_by_category"]:
            tables = extracted_data["tables_by_category"][category]
            report_content += f"\n**Tables ({len(tables)}):**\n"
            for table_name, table_path in tables:
                report_content += f"- {table_name}\n"

    # Add usage instructions
    report_content += """

## Usage Instructions

### Viewing Tables
- Navigate to the algorithm group folder (e.g., `cadence/`, `stride_length/`)
- Open HTML files in the `tables/` subdirectory with any web browser
- Each table includes context (heading and description) from the original document
- Tables can be directly copied and pasted into Word documents with formatting preserved

### Screenshots
- PNG screenshots of all tables are available in `screenshots/` subdirectories
- Navigate to the algorithm group folder, then to `screenshots/`
- Use these for presentations or reports where you need image format

### Images
- All plot images and figures are available in `images/` subdirectories
- Navigate to the algorithm group folder, then to `images/`
- These include all matplotlib plots and diagrams from the revalidation

## Integration with Reports
You can easily integrate these assets into your reports:
1. Navigate to the relevant algorithm group folder (cadence, stride_length, etc.)
2. Use images from the `images/` folder for plots and figures
3. Use table screenshots from the `screenshots/` folder for table visuals
4. Copy and paste HTML tables from the `tables/` folder directly into Word documents
   - All formatting and colors will be preserved
   - Tables are self-contained with inlined styles

This organization makes it easy to gather all assets for a specific analysis area.
"""

    # Save report
    with open(output_dir / "extraction_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)


async def main():
    parser = argparse.ArgumentParser(description="Extract images and tables from revalidation docs with screenshots")
    parser.add_argument(
        "--docs-dir", type=Path, default=Path(), help="Path to docs directory (default: current directory)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("revalidation_assets"),
        help="Output directory for extracted assets (default: revalidation_assets)",
    )
    parser.add_argument("--no-screenshots", action="store_true", help="Skip generating screenshots of tables")

    args = parser.parse_args()

    # Find HTML files first to determine categories
    html_files = find_html_files(args.docs_dir)

    if not html_files:
        print(f"No HTML files found in {args.docs_dir}/_build/html/auto_revalidation")
        return

    # Determine all categories from HTML files
    categories = set()
    for html_file in html_files:
        rel_path = html_file.relative_to(args.docs_dir / "_build/html/auto_revalidation")
        category = rel_path.parts[0] if len(rel_path.parts) > 1 else "general"
        categories.add(category)

    # Setup directories with discovered categories
    setup_directories(args.output_dir, list(categories))

    print(f"Found {len(html_files)} HTML files to process")
    print(f"Algorithm groups: {', '.join(sorted(categories))}")

    # Track extracted data
    extracted_data = {
        "total_files": len(html_files),
        "total_images": 0,
        "total_tables": 0,
        "images_by_category": {},
        "tables_by_category": {},
    }

    # Process each HTML file
    for html_file in html_files:
        print(f"Processing: {html_file.relative_to(args.docs_dir)}")

        # Extract images
        images = extract_images_from_html(html_file, args.docs_dir, args.output_dir)
        if images:
            rel_path = html_file.relative_to(args.docs_dir / "_build/html/auto_revalidation")
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "general"

            if category not in extracted_data["images_by_category"]:
                extracted_data["images_by_category"][category] = []
            extracted_data["images_by_category"][category].extend(images)
            extracted_data["total_images"] += len(images)

        # Extract tables
        tables = extract_tables_from_html(html_file, args.output_dir)
        if tables:
            rel_path = html_file.relative_to(args.docs_dir / "_build/html/auto_revalidation")
            category = rel_path.parts[0] if len(rel_path.parts) > 1 else "general"

            if category not in extracted_data["tables_by_category"]:
                extracted_data["tables_by_category"][category] = []
            extracted_data["tables_by_category"][category].extend(tables)
            extracted_data["total_tables"] += len(tables)

    # Generate screenshots
    if not args.no_screenshots:
        print("Generating table screenshots...")
        await generate_table_screenshots(args.output_dir)

    # Create summary report
    create_summary_report(args.output_dir, extracted_data)

    print("\nExtraction complete!")
    print(f"Assets saved to: {args.output_dir}")
    print(f"Images: {extracted_data['total_images']}")
    print(f"Tables: {extracted_data['total_tables']}")
    if not args.no_screenshots:
        print("Screenshots: Generated for all tables")


if __name__ == "__main__":
    asyncio.run(main())
