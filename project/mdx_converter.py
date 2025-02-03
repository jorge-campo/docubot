#!/usr/bin/env python3
"""
MDX-to-plaintext transformation script with:
 - Enhanced indentation
 - Removal of <ExternalIcon />
 - Stripping '}>...' from ContextTag labels
 - Recursive search for .mdx in docs/mdx/* subfolders

Directory structure:

project/
├── docs/
│   ├── mdx/             # Input MDX files (including subfolders)
│   └── processed/       # Output directory
└── mdx_converter.py     # This script

Transformation Pipeline:
1. Convert MDX features:
   - Admonitions => [Admonition: Xyz]
   - Tabs => separate blocks
   - ContextTag => "Some text", removing leading '}>' if present
   - <ExternalIcon /> => removed entirely
   - <br> => two trailing spaces
   - <Table> => [Table]\n|...| with indentation

2. Post-process indentation:
   - Flatten top-level headings/steps
   - Indent admonitions
   - Indent entire table blocks (including multi-line cells)

The script recursively searches docs/mdx for .mdx files and writes .txt outputs
to docs/processed in a flattened fashion.

Usage:
    python mdx_converter.py
"""

import re
import sys
from pathlib import Path

########################################
# Regex Patterns
########################################

RE_ADMONITION = re.compile(r'<Admonition\s+type="([^"\']+)"[^>]*>(.*?)</Admonition>', re.DOTALL)
RE_CONTEXTTAG = re.compile(r'<ContextTag(?:[^>]*)>(.*?)</ContextTag>', re.DOTALL)
RE_TABS_BLOCK = re.compile(r'<Tabs[^>]*>(.*?)</Tabs>', re.DOTALL)
RE_TABS_TRIGGER = re.compile(r'<TabsTrigger\s+value="([^"\']+)"([^>]*)>')
RE_TABS_CONTENT = re.compile(r'<TabsContent\s+value="([^"\']+)"[^>]*>(.*?)</TabsContent>', re.DOTALL)
RE_TABLE_BLOCK = re.compile(r'<Table[^>]*>(.*?)</Table>', re.DOTALL)
RE_TABLEHEAD = re.compile(r'<TableHead>(.*?)</TableHead>', re.DOTALL)
RE_TABLEHEADER = re.compile(r'<TableHeader[^>]*>(.*?)</TableHeader>', re.DOTALL)
RE_TABLECONTENT = re.compile(r'<TableContent>(.*?)</TableContent>', re.DOTALL)
RE_TABLEROW = re.compile(r'<TableRow[^>]*>(.*?)</TableRow>', re.DOTALL)
RE_TABLECELL = re.compile(r'<TableCell[^>]*>(.*?)</TableCell>', re.DOTALL)
RE_BR = re.compile(r'<br\s*/?>')

# To remove <ExternalIcon />
RE_EXTERNAL_ICON = re.compile(r'<ExternalIcon\s*/>')

########################################
# Conversion Functions
########################################

def remove_external_icons(text: str) -> str:
    """Remove any <ExternalIcon /> tags entirely."""
    return RE_EXTERNAL_ICON.sub('', text)

def convert_admonitions(text: str) -> str:
    """
    <Admonition type="XYZ">...</Admonition> =>
    [Admonition: Xyz]
    <content>
    """
    def _admon_replacer(match: re.Match) -> str:
        admon_type = match.group(1).strip()
        content = match.group(2).strip()
        return f"[Admonition: {admon_type.capitalize()}]\n{content}"
    return RE_ADMONITION.sub(_admon_replacer, text)

def convert_context_tags(text: str) -> str:
    """
    <ContextTag>...</ContextTag> => "..."
    - Remove any leading '}>...' from the content if present.
    """
    def _ctx_replacer(m: re.Match) -> str:
        content = m.group(1).strip()
        # e.g. '}>Scan Keycard' => 'Scan Keycard'
        content = re.sub(r'^}>\s*', '', content)
        return f"\"{content}\""
    return RE_CONTEXTTAG.sub(_ctx_replacer, text)

def convert_br_tags(text: str) -> str:
    """<br> or <br/> => two trailing spaces."""
    return RE_BR.sub("  ", text)

def convert_table_block(table_html: str) -> str:
    """
    <Table> => [Table]\n|...|
    """
    head_match = RE_TABLEHEAD.search(table_html)
    if head_match:
        head_content = head_match.group(1)
        headers = RE_TABLEHEADER.findall(head_content)
        headers = [h.strip() for h in headers]
    else:
        headers = []

    content_match = RE_TABLECONTENT.search(table_html)
    if content_match:
        content_html = content_match.group(1)
    else:
        content_html = table_html

    rows = RE_TABLEROW.findall(content_html)
    table_lines = []

    if headers:
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "|" + ("---|" * len(headers))
        table_lines.append(header_row)
        table_lines.append(separator_row)

    for row_html in rows:
        cells = RE_TABLECELL.findall(row_html)
        cells = [c.strip() for c in cells]
        line = "| " + " | ".join(cells) + " |"
        table_lines.append(line)

    if not table_lines:
        return "[Table]\n(No recognizable rows/cells)"

    return "[Table]\n" + "\n".join(table_lines)

def convert_tables(text: str) -> str:
    """Replace <Table> blocks with Markdown tables."""
    def _replacer(m: re.Match) -> str:
        return convert_table_block(m.group(1))
    return RE_TABLE_BLOCK.sub(_replacer, text)

def convert_tabs_block(tabs_html: str) -> str:
    """
    <Tabs> => multiple instructions blocks
    e.g. [Mobile Instructions], [Desktop Instructions]
    """
    triggers = RE_TABS_TRIGGER.findall(tabs_html)
    content_blocks = RE_TABS_CONTENT.findall(tabs_html)

    enabled_map = {}
    for (val, attrs) in triggers:
        disabled = ("disabled" in attrs)
        enabled_map[val] = not disabled

    out_lines = []
    for (val, block_html) in content_blocks:
        if val in enabled_map and enabled_map[val]:
            out_lines.append(f"[{val} Instructions]")
            out_lines.append(block_html.strip())

    return "\n".join(out_lines)

def convert_tabs(text: str) -> str:
    """Replace <Tabs>... with the output of convert_tabs_block."""
    def _tabs_replacer(m: re.Match) -> str:
        return convert_tabs_block(m.group(1))
    return RE_TABS_BLOCK.sub(_tabs_replacer, text)

def convert_mdx_to_text(mdx_text: str) -> str:
    """
    Main transformation pipeline:
    1) remove_external_icons
    2) convert_admonitions
    3) convert_tabs
    4) convert_context_tags
    5) convert_br_tags
    6) convert_tables
    """
    output = remove_external_icons(mdx_text)
    output = convert_admonitions(output)
    output = convert_tabs(output)
    output = convert_context_tags(output)
    output = convert_br_tags(output)
    output = convert_tables(output)
    return output

########################################
# Indentation Post-Processing
########################################

STEP_OR_HEADING_REGEX = [
    re.compile(r'^\s*\d+\.\s+'),           # e.g. "1. Step"
    re.compile(r'^\s*#{1,6}\s'),           # e.g. "#", "##"
    re.compile(r'^\s*\[.*Instructions\]'), # e.g. "[Desktop Instructions]"
]

RE_ADMONITION_LABEL = re.compile(r'^\s*\[Admonition:\s*.*\]')
RE_TABLE_LABEL = re.compile(r'^\s*\[Table\]')

def fix_indentation(final_text: str) -> str:
    """
    1) Flatten top-level steps/headings/instructions
    2) Indent admonition blocks by 4 spaces
    3) Indent entire table blocks by 4 spaces, ignoring step lines so multi-line cells remain in table
    """
    lines = final_text.splitlines()
    corrected = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # (A) Flatten headings/steps if not in a table or admonition
        if any(r.match(line) for r in STEP_OR_HEADING_REGEX):
            line = line.lstrip()
            corrected.append(line)
            i += 1
            continue

        # (B) Admonition block
        if RE_ADMONITION_LABEL.match(line):
            line = line.lstrip()
            line = "    " + line
            corrected.append(line)

            i += 1
            while i < len(lines):
                next_line = lines[i]
                if (not next_line.strip() or
                    RE_ADMONITION_LABEL.match(next_line) or
                    RE_TABLE_LABEL.match(next_line)):
                    if corrected and corrected[-1].strip():
                        corrected.append("")
                    break

                next_line = "    " + next_line.lstrip()
                corrected.append(next_line)
                i += 1
            else:
                if corrected and corrected[-1].strip():
                    corrected.append("")
            continue

        # (C) Table block
        if RE_TABLE_LABEL.match(line):
            line = line.lstrip()
            line = "    " + line
            corrected.append(line)

            i += 1
            while i < len(lines):
                next_line = lines[i]
                if (not next_line.strip() or
                    RE_ADMONITION_LABEL.match(next_line) or
                    RE_TABLE_LABEL.match(next_line)):
                    break
                next_line = "    " + next_line.lstrip()
                corrected.append(next_line)
                i += 1
            continue

        # (D) Otherwise, keep line
        corrected.append(line)
        i += 1

    return "\n".join(corrected)

########################################
# Main Script
########################################

def main():
    script_dir = Path(__file__).resolve().parent
    input_dir = script_dir / "docs" / "mdx"
    output_dir = script_dir / "docs" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Recursively gather all .mdx files
    mdx_files = list(input_dir.rglob("*.mdx"))
    if not mdx_files:
        print(f"No .mdx files found in {input_dir} or its subfolders.")
        sys.exit(0)

    for mdx_file in mdx_files:
        with open(mdx_file, "r", encoding="utf-8") as f:
            mdx_content = f.read()

        # 1) Convert MDX to plain text
        converted_text = convert_mdx_to_text(mdx_content)

        # 2) Fix indentation & spacing
        final_text = fix_indentation(converted_text)

        # 3) Save as .txt in docs/processed
        out_file = output_dir / (mdx_file.stem + ".txt")
        with open(out_file, "w", encoding="utf-8") as f:
            f.write(final_text)

        print(f"Converted {mdx_file} -> {out_file}")

if __name__ == "__main__":
    main()
