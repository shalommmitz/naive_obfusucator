# Folder Word Replacer & Restorer

A utility to **anonymize/transform words** across an entire folder tree and then **perfectly restore** it later, with **case-preserving** replacements applied to **both file contents and file/directory names**. Includes a robust **validation step**, a **reversible mapping**, **restoration-only** mode, and a **full comparison report** to verify integrity.

The main motivation is to avoid keywords that might prevent overly senstive LLMs from looking/modifing code.
---

## Target & High-Level Functionality

- **Transform step**: Copies an input folder to `<original>_modified` and replaces every case-variant occurrence (no word boundaries) of each word from a curated list with a **short English replacement of the exact same length**. Replacements are chosen so they **do not appear** anywhere in the original tree (filenames + UTF‑8 file contents, case-insensitive) to guarantee reversibility.
- **Restore step**: Using the saved mapping, reconstructs `<original>_unmodified` from `<original>_modified`, restoring file/directory names and file contents exactly (case preserved).
- **Validation**:
  - The input words list is **rejected** if any two entries have a **substring relation** (case-insensitive), e.g., `"aa"` and `"baab"`, which can make restoration ambiguous.
  - The selected replacement words are guaranteed to **not occur** in the original corpus.
- **Verification**: Performs a full comparison between the original folder and the restored folder and writes a human-friendly summary, detailed diffs, and a machine-readable JSON report.
- **Restoration-only**: If you already have a modified tree (even if it was further edited), you can run the **reverse** operation only.

---

## Files Used

- **Script**: `folder_word_replacer_v3.py`
- **Input words list** (next to the script): `<original_folder_name>_words.yaml`
  - YAML list or one-word-per-line; comments with `#` allowed.
- **Output mapping** (next to the script): `<original_folder_name>_replacment_words.yaml`
  - A YAML *dictionary* with `'original': 'replacement'` pairs.
- **Outputs created next to the original folder**:
  - `<original_folder_name>_modified/` — transformed copy (names + UTF‑8 contents).
  - `<original_folder_name>_unmodified/` — restored copy.
  - `<original_folder_name>_compare_report/` — verification reports (`summary.md`, `details.md`, `differences.json`).

---

## How to Use

### Command-Line Usage

> Requires **Python 3.8+**. No external dependencies.

1. Place your words file beside the script: `./<original>_words.yaml`.
2. Run the transform → restore → compare pipeline:

```bash
python obfusucator /path/to/<original>
```

This will:
- Validate the words list (no substrings, no duplicates).
- Generate collision-free, exact-length replacements that don’t appear in the original corpus.
- Create `<original>_modified` with replacements applied to names and UTF‑8 contents.
- Write mapping to `./<original>_replacment_words.yaml` (beside the script).
- Restore to `<original>_unmodified`.
- Compare original vs. unmodified and write a report to `<original>_compare_report/`.

#### Restoration-Only Mode

If you already have a modified tree and a mapping file:

```bash
# Use modified folder as positional input
python folder_word_replacer_v3.py /path/to/<original>_modified --restore-only

# Or specify modified and mapping explicitly
python folder_word_replacer_v3.py /path/to/<original> --restore-only \
  --modified /path/to/<original>_modified \
  --mapping  /path/to/<mapping.yaml> \
  --output   /path/to/<original>_unmodified
```

### Python Usage (Embedding)

```python
from pathlib import Path
import subprocess

script = Path('folder_word_replacer_v3.py')
original = Path('/data/project')

# Full pipeline
subprocess.run(['python', str(script), str(original)], check=True)

# Restore only (later)
subprocess.run([
    'python', str(script), str(original), '--restore-only',
    '--modified', str(original.parent / f"{original.name}_modified"),
    '--mapping',  str(script.parent / f"{original.name}_replacment_words.yaml"),
], check=True)
```

---

## Requirements

- **Python**: 3.8 or newer.
- **No external packages** required (standard library only).
- The script works best when files are **UTF‑8** for content replacement. Non-UTF files are treated as binary and copied as-is.
- Sufficient disk space for duplicated trees (`_modified`, `_unmodified`, and the report directory).

---

## Reversibility Guarantees & Validation Rules

- **Exact-length replacements**: Each original word is mapped to a replacement of the **same length**, enabling precise case transfer and preventing letter-count drift.
- **Case-preserving replacement**: Letter case is preserved for each occurrence (e.g., `AAa → BBb`). For patterns like ALL UPPER / ALL lower / TitleCase, the entire replacement follows that style.
- **No word-boundary restriction**: Replacements match substrings anywhere.
- **Word list validation**:
  - **Reject** if any two words have a **substring relation** (case-insensitive), e.g., `"aa"` and `"baab"`.
  - **Reject** if there are case-insensitive duplicates.
- **Replacement uniqueness & safety**:
  - Replacement words are **unique** across the mapping and **do not occur** in the original corpus (file/directory names + UTF‑8 text contents), case-insensitive.
- **Name collision detection**: If transformed names collide (even before restore), the process aborts with a clear error to protect reversibility.
- **Binary safety**: Binary files are never altered; they are copied byte-for-byte.

---

## How Conversion is Verified

After restoration, the tool compares the **original** tree with the **unmodified** tree:

- `summary.md` — counts of files, missing/extra, matches/mismatches; quick lists.
- `details.md` — unified **diffs** for text file mismatches; binary/type mismatch notes.
- `differences.json` — machine-friendly structure (useful for automated agents or CI).

A perfect run will show **0 mismatches** and no missing/extra files.

---

## Credits

This tool was written with the assistance of **M365 Copilot (Microsoft)**.

---

## License — MIT

```
MIT License

Copyright (c) 2026 Shalom Mitz

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

