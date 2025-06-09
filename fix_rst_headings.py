#!/usr/bin/env python3
"""
Script to fix RST heading underline length mismatches in Python and RST files.

THIS FILE IS 100% AI GENERATED (but it seems to work).

This script scans Python and RST files for RST-style headings and ensures
that the underline length matches the heading text length exactly.

For Python files: Checks RST headings in comments and docstrings
For RST files: Checks RST headings in the file content directly

Supports all RST underline characters: = - ` : ' " ~ ^ _ * + # < >

Dependencies:
  - pathspec: For .gitignore support (pip install pathspec)

Exit codes:
  0: No issues found or all fixes applied successfully
  1: Issues found (in --check mode) or issues found in --dry-run mode
  2: Invalid arguments or file system errors

Usage examples:
  # Check for issues (CI mode) - respects .gitignore by default
  python fix_rst_headings.py --check src/ docs/ README.rst

  # Preview fixes without applying them
  python fix_rst_headings.py --dry-run docs/

  # Apply fixes to multiple locations
  python fix_rst_headings.py src/ tests/ docs/ file1.py file2.rst

  # Process all files including gitignored ones
  python fix_rst_headings.py --no-gitignore src/

  # Apply fixes to single location
  python fix_rst_headings.py docs/
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import pathspec

    HAS_PATHSPEC = True
except ImportError:
    HAS_PATHSPEC = False

# All RST underline characters
RST_UNDERLINE_CHARS = "=-`:'\"~^_*+#<>"


def find_docstrings(content: str) -> List[Tuple[int, int, str]]:
    """
    Find docstrings in Python code.

    Returns
    -------
        List of tuples: (start_line, end_line, docstring_content)
    """
    lines = content.split("\n")
    docstrings = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        # Look for triple quotes (both """ and ''')
        if line.startswith('"""') or line.startswith("'''"):
            quote_type = line[:3]
            start_line = i

            # Check if it's a single-line docstring
            if line.count(quote_type) >= 2 and len(line) > 3:
                # Single line docstring
                docstring_content = line[3:-3]
                docstrings.append((start_line, start_line, docstring_content))
                i += 1
                continue

            # Multi-line docstring
            i += 1
            docstring_lines = []

            while i < len(lines):
                if lines[i].strip().endswith(quote_type):
                    # Found the end
                    end_line = i
                    # Remove the closing quotes from the last line
                    last_line = lines[i].rstrip()
                    if last_line.endswith(quote_type):
                        last_line = last_line[:-3]
                    if last_line.strip():
                        docstring_lines.append(last_line)

                    docstring_content = "\n".join(docstring_lines)
                    docstrings.append((start_line, end_line, docstring_content))
                    break
                docstring_lines.append(lines[i])
                i += 1

        i += 1

    return docstrings


def find_rst_headings_in_text(text: str, line_offset: int = 0) -> List[Tuple[int, str, int, str, str]]:
    """
    Find RST heading/underline pairs in plain text.

    Args:
        text: The text to search in
        line_offset: Offset to add to line numbers (for docstrings)

    Returns
    -------
        List of tuples: (heading_line_num, heading_text, underline_line_num, underline_text, underline_char)
    """
    lines = text.split("\n")
    headings = []

    for i in range(len(lines) - 1):
        current_line = lines[i].strip()
        next_line = lines[i + 1].strip()

        # Check if current line has text (potential heading)
        if len(current_line) > 0 and len(next_line) > 0:
            # Check if next line is an underline (consists of only one type of RST underline character)
            if len(set(next_line)) == 1 and next_line[0] in RST_UNDERLINE_CHARS:
                underline_char = next_line[0]
                headings.append((i + line_offset, current_line, i + 1 + line_offset, next_line, underline_char))

    return headings


def find_rst_headings_in_file(content: str, file_path: Path) -> List[Tuple[int, str, int, str, str]]:
    """
    Find RST heading/underline pairs in file content.

    For Python files: Checks comments and docstrings
    For RST files: Checks the file content directly

    Returns
    -------
        List of tuples: (heading_line_num, heading_text, underline_line_num, underline_text, underline_char)
    """
    headings = []

    if file_path.suffix == ".py":
        # Python file - check comments and docstrings
        lines = content.split("\n")

        # Find headings in comments
        for i in range(len(lines) - 1):
            current_line = lines[i].strip()
            next_line = lines[i + 1].strip()

            # Check if current line is a comment with text (potential heading)
            if current_line.startswith("#") and len(current_line) > 1:
                heading_text = current_line[1:].strip()

                # Check if next line is a comment with underline characters
                if next_line.startswith("#") and len(next_line) > 1:
                    underline_text = next_line[1:].strip()

                    # Check if underline consists of only one type of RST underline character
                    if (
                        len(underline_text) > 0
                        and len(set(underline_text)) == 1
                        and underline_text[0] in RST_UNDERLINE_CHARS
                    ):
                        underline_char = underline_text[0]
                        headings.append((i, heading_text, i + 1, underline_text, underline_char))

        # Find headings in docstrings
        docstrings = find_docstrings(content)
        for start_line, end_line, docstring_content in docstrings:
            # Find headings within this docstring
            docstring_headings = find_rst_headings_in_text(docstring_content, start_line + 1)
            headings.extend(docstring_headings)

    elif file_path.suffix == ".rst":
        # RST file - check the content directly
        headings = find_rst_headings_in_text(content, 0)

    return headings


def fix_underline_length(heading_text: str, underline_char: str) -> str:
    """Create a properly sized underline for the given heading text."""
    return underline_char * len(heading_text)


def check_rst_headings_in_content(content: str, file_path: Path) -> List[Tuple[int, str, str, str, str]]:
    """
    Check RST heading underline lengths in the content without fixing them.

    Returns
    -------
        List of issues: (heading_line, heading_text, underline_text, expected_underline, underline_char)
    """
    headings = find_rst_headings_in_file(content, file_path)
    issues = []

    for heading_line, heading_text, underline_line, underline_text, underline_char in headings:
        expected_underline = fix_underline_length(heading_text, underline_char)

        if underline_text != expected_underline:
            issues.append((heading_line, heading_text, underline_text, expected_underline, underline_char))

    return issues


def fix_rst_headings_in_content(content: str, file_path: Path, verbose: bool = True) -> Tuple[str, int]:
    """
    Fix RST heading underline lengths in the content.

    Returns
    -------
        Tuple of (fixed_content, number_of_fixes)
    """
    lines = content.split("\n")
    fixes_made = 0

    headings = find_rst_headings_in_file(content, file_path)

    # Process headings in reverse order to avoid line number shifts
    for heading_line, heading_text, underline_line, underline_text, underline_char in reversed(headings):
        expected_underline = fix_underline_length(heading_text, underline_char)

        if underline_text != expected_underline:
            # Fix the underline
            original_line = lines[underline_line]

            # Check if this is in a comment or in a docstring
            if lines[underline_line].strip().startswith("#"):
                # It's a comment - preserve the original indentation and comment character
                prefix = original_line[: original_line.find(underline_text)]
                lines[underline_line] = prefix + expected_underline
            else:
                # It's in a docstring - preserve the original indentation
                # Find the indentation of the underline
                stripped = original_line.lstrip()
                indentation = original_line[: len(original_line) - len(stripped)]
                lines[underline_line] = indentation + expected_underline

            fixes_made += 1

            if verbose:
                print(f"Fixed heading '{heading_text}' (line {heading_line + 1})")
                print(f"  Old underline: '{underline_text}' ({len(underline_text)} chars)")
                print(f"  New underline: '{expected_underline}' ({len(expected_underline)} chars)")

    return "\n".join(lines), fixes_made


def process_file(file_path: Path, dry_run: bool = False, check_mode: bool = False) -> bool:
    """
    Process a single Python or RST file to fix RST heading underlines.

    Returns
    -------
        True if any issues were found (or fixes were made), False otherwise
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            original_content = f.read()

        if check_mode:
            # Check mode: only report issues, don't fix
            issues = check_rst_headings_in_content(original_content, file_path)

            if issues:
                print(f"\n❌ {file_path}")
                for heading_line, heading_text, underline_text, expected_underline, underline_char in issues:
                    print(f"   Line {heading_line + 1}: Heading '{heading_text}'")
                    print(f"      Expected: {len(expected_underline)} {underline_char} characters")
                    print(f"      Found:    {len(underline_text)} {underline_char} characters")
                return True

            return False

        # Fix mode
        fixed_content, fixes_made = fix_rst_headings_in_content(original_content, file_path, verbose=not dry_run)

        if fixes_made > 0:
            if not dry_run:
                print(f"\n📝 File: {file_path}")
                print(f"   Fixed {fixes_made} heading(s)")

            if not dry_run:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(fixed_content)
                print("   ✅ Changes saved")
            else:
                print("   🔍 Dry run - no changes saved")

            return True

        return False

    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")
        return False


def find_git_root(path: Path) -> Path:
    """Find the git repository root by looking for .git directory."""
    current_path = path.resolve()

    while True:
        if (current_path / ".git").exists():
            return current_path

        parent = current_path.parent
        if parent == current_path:
            # Reached filesystem root, return the original path
            return path.resolve()
        current_path = parent


def find_gitignore_files_for_file(file_path: Path, git_root: Path) -> List[Tuple[Path, Path]]:
    """
    Find all .gitignore files that could affect the given file.

    Returns
    -------
        List of tuples: (gitignore_file_path, directory_where_it_applies)
    """
    gitignore_files = []
    file_path = file_path.resolve()

    # Start from the file's directory and walk up to the git root
    current_dir = file_path.parent

    while current_dir.is_relative_to(git_root) or current_dir == git_root:
        gitignore_path = current_dir / ".gitignore"
        if gitignore_path.exists() and gitignore_path.is_file():
            gitignore_files.append((gitignore_path, current_dir))

        if current_dir == git_root:
            break

        current_dir = current_dir.parent

    # Return in reverse order so more specific (deeper) .gitignore files are processed last
    return list(reversed(gitignore_files))


def is_ignored_by_git(file_path: Path, base_path: Path) -> bool:
    """Check if a file is ignored by git using proper .gitignore hierarchy."""
    if not HAS_PATHSPEC:
        return False

    file_path = file_path.resolve()
    git_root = find_git_root(base_path)

    # Find all relevant .gitignore files
    gitignore_files = find_gitignore_files_for_file(file_path, git_root)

    if not gitignore_files:
        return False

    # Process each .gitignore file in order (general to specific)
    ignored = False

    for gitignore_file_path, gitignore_dir in gitignore_files:
        try:
            with open(gitignore_file_path, encoding="utf-8") as f:
                patterns = [
                    line.strip() for line in f.read().splitlines() if line.strip() and not line.strip().startswith("#")
                ]

            if patterns:
                spec = pathspec.PathSpec.from_lines("gitwildmatch", patterns)

                # Get the file path relative to this .gitignore's directory
                try:
                    relative_path = file_path.relative_to(gitignore_dir)
                    if spec.match_file(str(relative_path)):
                        ignored = True
                    # Check if any pattern explicitly un-ignores this file (starts with !)
                    for pattern in patterns:
                        if pattern.startswith("!"):
                            negation_spec = pathspec.PathSpec.from_lines("gitwildmatch", [pattern[1:]])
                            if negation_spec.match_file(str(relative_path)):
                                ignored = False
                except ValueError:
                    # File is not under this .gitignore's directory
                    continue

        except (OSError, UnicodeDecodeError):
            # Skip files we can't read
            continue

    return ignored


def find_rst_files(directory: Path, recursive: bool = True, respect_gitignore: bool = True) -> List[Path]:
    """Find all Python and RST files in the given directory."""
    files = []
    if recursive:
        files.extend(directory.rglob("*.py"))
        files.extend(directory.rglob("*.rst"))
    else:
        files.extend(directory.glob("*.py"))
        files.extend(directory.glob("*.rst"))

    # Filter out gitignored files if pathspec is available and respect_gitignore is True
    if respect_gitignore and HAS_PATHSPEC:
        files = [f for f in files if not is_ignored_by_git(f, directory)]

    return sorted(files)  # Sort for consistent order


def main():
    parser = argparse.ArgumentParser(description="Fix RST heading underline length mismatches in Python and RST files")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Path(s) to Python/RST file(s) or directory(ies) to process"
    )
    parser.add_argument(
        "--check", action="store_true", help="Check for issues without fixing them. Exit with code 1 if issues found."
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fixed without making changes")
    parser.add_argument("--no-recursive", action="store_true", help="Do not process directories recursively")
    parser.add_argument(
        "--no-gitignore", action="store_true", help="Do not respect .gitignore files (process all files)"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.check and args.dry_run:
        print("❌ Error: --check and --dry-run cannot be used together")
        sys.exit(2)

    # Check if gitignore support is requested but not available
    if not args.no_gitignore and not HAS_PATHSPEC:
        print("⚠️  Warning: pathspec not installed. Install with 'pip install pathspec' for .gitignore support.")
        print("    Proceeding without .gitignore filtering...")
        respect_gitignore = False
    else:
        respect_gitignore = not args.no_gitignore

    # Validate paths and collect all files to process
    files_to_process = []

    for path in args.paths:
        if not path.exists():
            print(f"❌ Path does not exist: {path}")
            sys.exit(2)

        if path.is_file():
            if path.suffix in {".py", ".rst"}:
                files_to_process.append(path)
            else:
                print(f"❌ File is not a Python or RST file: {path}")
                sys.exit(2)
        elif path.is_dir():
            dir_files = find_rst_files(path, not args.no_recursive, respect_gitignore)
            if not dir_files:
                print(f"❌ No Python or RST files found in: {path}")
                sys.exit(2)
            files_to_process.extend(dir_files)

    # Remove duplicates while preserving order
    seen = set()
    unique_files = []
    for file_path in files_to_process:
        if file_path not in seen:
            seen.add(file_path)
            unique_files.append(file_path)
    files_to_process = unique_files

    if not args.check:
        print(f"🔍 Found {len(files_to_process)} Python/RST file(s) to process")

    total_files_with_issues = 0

    for file_path in files_to_process:
        if process_file(file_path, args.dry_run, args.check):
            total_files_with_issues += 1

    if args.check:
        # Check mode - exit with appropriate code
        if total_files_with_issues > 0:
            print(f"\n❌ Found RST heading issues in {total_files_with_issues} file(s)")
            print("   Run without --check to fix these issues")
            sys.exit(1)  # Issues found
        else:
            print(f"\n✅ All {len(files_to_process)} file(s) have properly formatted RST headings")
            sys.exit(0)  # No issues found
    else:
        # Fix/dry-run mode
        print("\n📊 Summary:")
        print(f"   Files processed: {len(files_to_process)}")
        print(f"   Files with issues: {total_files_with_issues}")

        if args.dry_run:
            print("   🔍 This was a dry run - no files were modified")
            if total_files_with_issues > 0:
                sys.exit(1)  # Issues found in dry-run
        elif total_files_with_issues > 0:
            print("   ✅ All fixes have been applied")
        else:
            print("   ✨ No fixes needed - all headings are properly formatted")

        sys.exit(0)  # Success


if __name__ == "__main__":
    main()
