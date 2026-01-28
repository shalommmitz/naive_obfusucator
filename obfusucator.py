#!/usr/bin/env python3
"""
folder_word_replacer_v3.py

Features:
- Reads words from <original_folder_name>_words.yaml placed in the SAME FOLDER as this script.
- Validates the word list: rejects if any two words have a substring relation (case-insensitive),
  e.g., "aa" and "baab" -> ERROR (restoration may become ambiguous).
- Generates replacement words of EXACT same length as originals:
  * prefer curated short English words (3..8 chars) not appearing in original corpus;
  * fallback to pronounceable English-like tokens (still collision-free).
- Ensures replacements DO NOT APPEAR ANYWHERE in the original corpus (filenames + UTF-8 text), case-insensitive.
- Writes mapping as YAML dictionary to <original_folder_name>_replacment_words.yaml IN THE SCRIPT FOLDER.
- Copies recursively to <original>_modified applying replacements in:
  * file/directory NAMES and
  * UTF-8 text file CONTENTS (binary files copied as-is).
- Case-insensitive matching; case is PRESERVED letter-by-letter (AAa -> BBb).
- No word boundary constraints (substring matches are replaced).
- Reconstructs <original>_unmodified using the mapping.
- Full compare (original vs unmodified) with summary.md, details.md (diffs), differences.json.

Also supports:
- --restore-only: perform only the reverse operation (useful when modified tree was further edited).
- --modified / --mapping / --output flags for restoration-only mode.

No external dependencies required.
"""

import os
import sys
import re
import difflib
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# -----------------------------
# Utility: paths & IO
# -----------------------------

def script_dir() -> Path:
    return Path(__file__).resolve().parent

def assert_dir_exists(p: Path, label: str):
    if not p.exists() or not p.is_dir():
        raise SystemExit(f"Error: {label} does not exist or is not a directory: {p}")

def read_bytes_safe(p: Path) -> bytes:
    return p.read_bytes()

def is_text_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8")
        return True
    except Exception:
        return False

# -----------------------------
# Word list load & validation
# -----------------------------

def load_words_yaml(yaml_path: Path) -> List[str]:
    """
    Load a list of words from a simple YAML list or one word per line.
    Lines starting with '#' ignored. De-duplicates while preserving order.
    """
    if not yaml_path.exists():
        raise SystemExit(f"Error: words file not found: {yaml_path}")

    words: List[str] = []
    seen = set()
    with yaml_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("-"):
                item = line.lstrip("-").strip()
            else:
                item = line
            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                item = item[1:-1]
            item = item.strip()
            if item and item not in seen:
                words.append(item)
                seen.add(item)
    return words

def validate_word_list(words: List[str]):
    """
    Validates:
    - Non-empty
    - No duplicates (case-insensitive)
    - No substring relations (case-insensitive) among different words.
    """
    if not words:
        raise SystemExit("Error: words list is empty.")

    # Remove exact duplicates (case-insensitive) for validation; but keep originals for mapping
    lower_list = [w.lower() for w in words]
    if len(set(lower_list)) != len(lower_list):
        # find duplicates
        dupes = sorted(set([w for w in lower_list if lower_list.count(w) > 1]))
        raise SystemExit(f"Error: duplicate words (case-insensitive) detected: {dupes}")

    # Substring validation
    # If any pair has a substring relation, abort (restoration may become ambiguous).
    sorted_by_len = sorted(lower_list, key=len)
    for i in range(len(sorted_by_len)):
        for j in range(i+1, len(sorted_by_len)):
            a = sorted_by_len[i]
            b = sorted_by_len[j]
            if a in b or b in a:
                # Report original casings that triggered
                raise SystemExit(
                    "Error: invalid words list — substring relation detected between two words.\n"
                    f"Example pair: '{a}' and '{b}'.\n"
                    "This can make restoration ambiguous. Please remove one of them."
                )

# -----------------------------
# Corpus scanning
# -----------------------------

def build_corpus_snapshot(root: Path) -> str:
    """
    Create a single lowercase string of:
    - all directory and file names
    - UTF-8 text file contents
    Binary files are skipped for contents.
    """
    parts = []
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            parts.append(d.lower())
        for fn in filenames:
            parts.append(fn.lower())
            fpath = Path(dirpath) / fn
            try:
                data = read_bytes_safe(fpath)
                text = data.decode("utf-8")
                parts.append(text.lower())
            except Exception:
                pass
    return "\n".join(parts)

# -----------------------------
# Case-preserving replacement (improved)
# -----------------------------

def _case_profile(s: str) -> str:
    letters = [ch for ch in s if ch.isalpha()]
    if not letters:
        return 'mixed'
    if all(ch.isupper() for ch in letters):
        return 'upper'
    if all(ch.islower() for ch in letters):
        return 'lower'
    # title if first alpha upper and all other alphas lower
    first_alpha_idx = next((i for i,ch in enumerate(s) if ch.isalpha()), None)
    if first_alpha_idx is not None:
        first_upper = s[first_alpha_idx].isupper()
        rest_letters = [ch for i,ch in enumerate(s) if ch.isalpha() and i != first_alpha_idx]
        if first_upper and all(ch.islower() for ch in rest_letters):
            return 'title'
    return 'mixed'

def apply_case_pattern(source_match: str, replacement_base: str) -> str:
    """
    Robust case transfer:
    - If source is ALL UPPER -> uppercase replacement.
    - If ALL lower -> lowercase replacement.
    - If TitleCase -> capitalize replacement.
    - Else letter-wise; for extra characters beyond source length,
      continue with the last seen alpha-case (default to lower).
    """
    profile = _case_profile(source_match)
    if profile == 'upper':
        return replacement_base.upper()
    if profile == 'lower':
        return replacement_base.lower()
    if profile == 'title':
        return (replacement_base[:1].upper() + replacement_base[1:].lower()) if replacement_base else replacement_base

    out = []
    last_alpha_is_upper = None
    for i, ch in enumerate(replacement_base):
        if i < len(source_match):
            s = source_match[i]
            if s.isalpha():
                last_alpha_is_upper = s.isupper()
                out.append(ch.upper() if last_alpha_is_upper else ch.lower())
            else:
                out.append(ch.lower())
        else:
            if last_alpha_is_upper is None:
                out.append(ch.lower())
            else:
                out.append(ch.upper() if last_alpha_is_upper else ch.lower())
    return "".join(out)

def compile_patterns(words: List[str]) -> List[Tuple[str, re.Pattern]]:
    """
    Compile case-insensitive regex for each word. No word boundaries.
    Sort by descending length to reduce overlap artifacts.
    """
    sorted_words = sorted(words, key=len, reverse=True)
    compiled = []
    for w in sorted_words:
        pat = re.compile(re.escape(w), flags=re.IGNORECASE)
        compiled.append((w, pat))
    return compiled

def multi_replace_text(text: str, mapping: Dict[str, str], compiled: List[Tuple[str, re.Pattern]]) -> str:
    for original, pattern in compiled:
        base = mapping[original]
        def _repl(m: re.Match) -> str:
            return apply_case_pattern(m.group(0), base)
        text = pattern.sub(_repl, text)
    return text

# -----------------------------
# Replacement word generation
# -----------------------------

# Curated pool of short English words (3..8 chars). You can expand this list as needed.
SHORT_ENGLISH_WORDS = [
    "able","acid","aged","also","area","army","away","baby","back","ball","band","bank","base","bath","bear","beat","been",
    "beer","bell","belt","best","bill","bird","blow","blue","boat","body","bone","book","born","both","bowl","bulb","busy",
    "cake","call","calm","came","camp","card","care","case","cash","cast","cell","chat","city","clay","club","coal","coat",
    "code","cold","come","cook","cool","cope","copy","cord","core","cost","crew","crop","dark","data","dawn","dead","deal",
    "dear","debt","deep","deny","desk","dial","diet","dine","dirt","disc","dive","door","dose","down","draw","drew","drop",
    "drum","duck","dull","duly","duty","each","earn","ease","east","easy","echo","edge","edit","else","even","ever","evil",
    "exit","face","fact","fail","fair","fall","farm","fast","fate","fear","feed","feel","file","fill","film","find","fine",
    "fire","firm","fish","five","flag","flat","flow","fold","folk","food","foot","ford","form","fort","four","free","from",
    "fuel","full","fund","gain","game","gang","gate","gave","gear","gene","gift","girl","give","goal","goat","gold","golf",
    "gone","good","gray","grow","gulf","hair","half","hall","hand","hang","hard","harm","hate","have","head","heal","hear",
    "heat","held","hell","help","herb","hero","high","hill","hire","hold","hole","holy","home","hope","host","hour","huge",
    "hunt","hurt","idea","inch","into","iron","item","join","joke","jury","just","keep","kept","kick","kill","kind","king",
    "knee","knew","know","lack","lady","laid","lake","lamp","land","lane","last","late","lead","leaf","leak","lean","leap",
    "left","lend","lens","less","life","lift","like","lime","limb","line","link","lion","list","live","load","loan","lock",
    "logo","long","look","loop","lord","lose","loss","lost","lots","loud","love","luck","made","mail","main","make","male",
    "mall","many","mark","mass","mate","math","meal","mean","meat","meet","melt","mild","mile","milk","mill","mind","mine",
    "mini","mint","miss","mode","mood","moon","more","most","move","much","must","name","navy","near","neat","neck","need",
    "news","next","nice","nick","nine","none","noon","nose","note","okay","once","only","open","oral","pack","page","paid",
    "pain","pair","pale","park","part","past","path","peak","pear","peel","peer","pick","pill","pine","pink","pipe","plan",
    "play","plot","plug","plum","plus","poem","poet","pole","poll","pool","poor","port","pose","post","pour","pray","pure",
    "push","quit","race","rack","rain","rank","rare","rate","read","real","rear","redo","rent","rest","rice","rich","ride",
    "ring","rise","risk","road","roam","rock","role","roll","roof","room","root","rope","rose","ruin","rule","rush","safe",
    "sail","salt","same","sand","save","scan","scar","seed","seek","seem","seen","self","sell","send","sent","sept","ship",
    "shoe","shop","shot","show","shut","sick","side","sign","silk","silo","sing","sink","site","size","skin","slab","slam",
    "slip","slot","slow","snow","sock","soft","soil","sold","sole","solo","some","song","soon","sort","soul","soup","sour",
    "spin","spit","spot","spur","star","stay","stem","step","stop","store","storm","story","stub","stud","stuff","suit",
    "sure","swim","tale","talk","tall","task","team","tear","tech","tell","tend","term","test","text","than","that","them",
    "then","they","thin","this","thus","tide","tied","tier","tile","till","time","tire","told","toll","tone","took","tool",
    "tour","town","trap","tray","tree","trim","trip","tune","turn","twin","type","unit","upon","urge","used","user","vary",
    "vast","veal","veer","veil","vein","vice","view","vine","visa","void","volt","vote","wage","wait","wake","walk","wall",
    "want","ward","warm","warn","wash","wave","weak","wear","weed","week","well","went","were","west","what","when","whom",
    "wide","wife","wild","will","wind","wine","wing","wink","wire","wise","wish","with","wood","wool","word","work","worm",
    "yard","yarn","year","yell","zeal","zest"
]

VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"

def gen_english_like_token_exact_length(n: int) -> str:
    """Generate a pronounceable English-like token of EXACT length n."""
    if n <= 0:
        return "a"
    token = []
    use_consonant = True
    i = 0
    while i < n:
        ch = random.choice(CONSONANTS if use_consonant else VOWELS)
        token.append(ch)
        i += 1
        # small chance to double a consonant if room allows
        if use_consonant and i < n and random.random() < 0.12:
            token.append(ch)
            i += 1
        use_consonant = not use_consonant
    return "".join(token[:n])

def generate_pool_exact_len(corpus_lower: str, desired: int, L: int) -> List[str]:
    """
    Produce a pool of EXACT length-L replacements that do NOT appear in corpus (case-insensitive).
    Prefer real English words; fallback to generated tokens.
    """
    pool = []
    seen = set()
    # 1) curated words of exact length
    for w in SHORT_ENGLISH_WORDS:
        wl = w.lower()
        if len(wl) != L:
            continue
        if wl in seen:
            continue
        if wl in corpus_lower:
            continue
        pool.append(w)
        seen.add(wl)
        if len(pool) >= desired:
            return pool

    # 2) fallback to generated pronounceable tokens
    attempts = 0
    while len(pool) < desired:
        attempts += 1
        cand = gen_english_like_token_exact_length(L)
        cl = cand.lower()
        if cl in seen:
            continue
        if cl in corpus_lower:
            continue
        pool.append(cand)
        seen.add(cl)
        if attempts > 20000:
            raise SystemExit("Error: could not generate enough collision-free replacement words.")
    return pool

def generate_mapping(words: List[str], original_root: Path) -> Dict[str, str]:
    """
    Produce original->replacement mapping with EXACT length equality, uniqueness,
    and absence from original corpus.
    """
    corpus_lower = build_corpus_snapshot(original_root).lower()
    words_sorted = sorted(words, key=len, reverse=True)
    mapping: Dict[str, str] = {}
    used = set()

    for w in words_sorted:
        L = len(w)
        # At least one candidate per word
        candidates = generate_pool_exact_len(corpus_lower, 3, L)  # try a few
        chosen = None
        for c in candidates:
            if c.lower() not in used:
                chosen = c
                break
        if not chosen:
            # brute try
            for _ in range(5000):
                c = gen_english_like_token_exact_length(L)
                cl = c.lower()
                if cl not in used and cl not in corpus_lower:
                    chosen = c
                    break
        if not chosen:
            raise SystemExit(f"Error: cannot allocate unique replacement for word '{w}'.")
        mapping[w] = chosen
        used.add(chosen.lower())

    # Final sanity: ensure no replacement appears in original corpus (redundant but explicit)
    for orig, rep in mapping.items():
        if rep.lower() in corpus_lower:
            raise SystemExit(f"Error: chosen replacement '{rep}' for '{orig}' appears in original corpus. Aborting.")
    return mapping

# -----------------------------
# Name & content transformation
# -----------------------------

def multi_replace_name_component(name: str, mapping: Dict[str, str], compiled: List[Tuple[str, re.Pattern]]) -> str:
    return multi_replace_text(name, mapping, compiled)

def copy_and_transform(src_root: Path, dst_root: Path, mapping: Dict[str, str]) -> None:
    """
    Copy src_root to dst_root with transformation applied to components and text files.
    Detects collisions and aborts to preserve reversibility.
    """
    if dst_root.exists():
        raise SystemExit(f"Error: destination already exists: {dst_root}")
    compiled = compile_patterns(list(mapping.keys()))
    created = set()

    for dirpath, dirnames, filenames in os.walk(src_root):
        rel_dir = Path(dirpath).relative_to(src_root)
        transformed_parts = [multi_replace_name_component(p, mapping, compiled) for p in rel_dir.parts]
        transformed_rel_dir = Path(*transformed_parts) if transformed_parts else Path()
        dst_dir = dst_root / transformed_rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fn in filenames:
            new_fn = multi_replace_name_component(fn, mapping, compiled)
            src_path = Path(dirpath) / fn
            dst_path = dst_dir / new_fn

            if dst_path in created or dst_path.exists():
                raise SystemExit(f"Error: name collision detected when creating: {dst_path}")
            created.add(dst_path)

            data = read_bytes_safe(src_path)
            if is_text_utf8(data):
                text = data.decode("utf-8")
                new_text = multi_replace_text(text, mapping, compiled)
                dst_path.write_text(new_text, encoding="utf-8")
            else:
                dst_path.write_bytes(data)

# -----------------------------
# Reverse transformation
# -----------------------------

def invert_mapping(mapping: Dict[str, str]) -> Dict[str, str]:
    inv: Dict[str, str] = {}
    for k, v in mapping.items():
        if v in inv:
            raise SystemExit(f"Error: non-bijective mapping; duplicate replacement: {v}")
        inv[v] = k
    return inv

def reverse_transform(mod_root: Path, unmod_root: Path, mapping: Dict[str, str]) -> None:
    """
    Reconstruct original tree from <folder>_modified using mapping.
    """
    if unmod_root.exists():
        raise SystemExit(f"Error: destination already exists: {unmod_root}")

    inverse = invert_mapping(mapping)
    compiled_inv = compile_patterns(list(inverse.keys()))
    created = set()

    for dirpath, dirnames, filenames in os.walk(mod_root):
        rel_dir = Path(dirpath).relative_to(mod_root)
        rev_parts = [multi_replace_text(p, inverse, compiled_inv) for p in rel_dir.parts]
        rev_rel_dir = Path(*rev_parts) if rev_parts else Path()

        dst_dir = unmod_root / rev_rel_dir
        dst_dir.mkdir(parents=True, exist_ok=True)

        for fn in filenames:
            new_fn = multi_replace_text(fn, inverse, compiled_inv)
            src_path = Path(dirpath) / fn
            dst_path = dst_dir / new_fn

            if dst_path in created or dst_path.exists():
                raise SystemExit(f"Error: reverse name collision detected when creating: {dst_path}")
            created.add(dst_path)

            data = read_bytes_safe(src_path)
            if is_text_utf8(data):
                text = data.decode("utf-8")
                restored = multi_replace_text(text, inverse, compiled_inv)
                dst_path.write_text(restored, encoding="utf-8")
            else:
                dst_path.write_bytes(data)

# -----------------------------
# Comparison & reporting
# -----------------------------

def walk_tree_files(root: Path) -> Dict[str, Path]:
    files = {}
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            ap = Path(dirpath) / fn
            rp = ap.relative_to(root).as_posix()
            files[rp] = ap
    return files

def read_text_if_utf8(p: Path):
    data = read_bytes_safe(p)
    try:
        s = data.decode("utf-8")
        return True, s
    except Exception:
        import hashlib
        h = hashlib.sha256(data).hexdigest()
        return False, f"<binary sha256={h} size={len(data)}>"

def compare_and_report(orig_root: Path, restored_root: Path, report_root: Path) -> None:
    report_root.mkdir(parents=True, exist_ok=True)
    summary_path = report_root / "summary.md"
    details_path = report_root / "details.md"
    json_path = report_root / "differences.json"

    orig_files = walk_tree_files(orig_root)
    new_files = walk_tree_files(restored_root)

    orig_set = set(orig_files.keys())
    new_set = set(new_files.keys())

    missing = sorted(orig_set - new_set)
    extra = sorted(new_set - orig_set)
    common = sorted(orig_set & new_set)

    mismatches = []
    matches = []
    dlines = []

    for rel in common:
        p1 = orig_files[rel]
        p2 = new_files[rel]
        t1, c1 = read_text_if_utf8(p1)
        t2, c2 = read_text_if_utf8(p2)

        if t1 and t2:
            if c1 == c2:
                matches.append(rel)
            else:
                mismatches.append(rel)
                diff = difflib.unified_diff(
                    c1.splitlines(keepends=True),
                    c2.splitlines(keepends=True),
                    fromfile=f"original/{rel}",
                    tofile=f"unmodified/{rel}",
                    n=3
                )
                dlines.append(f"### DIFF: {rel}\n")
                dlines.append("```\n")
                dlines.extend(diff)
                dlines.append("```\n\n")
        elif (not t1) and (not t2):
            if c1 == c2:
                matches.append(rel)
            else:
                mismatches.append(rel)
                dlines.append(f"### BINARY MISMATCH: {rel}\n- original: {c1}\n- unmodified: {c2}\n\n")
        else:
            mismatches.append(rel)
            dlines.append(f"### TYPE MISMATCH: {rel}\n- original is_text_utf8={t1}\n- unmodified is_text_utf8={t2}\n\n")

    summary = []
    summary.append("# Comparison Summary\n")
    summary.append(f"- Original files: {len(orig_set)}\n")
    summary.append(f"- Unmodified files: {len(new_set)}\n")
    summary.append(f"- Missing in unmodified: {len(missing)}\n")
    summary.append(f"- Extra in unmodified: {len(extra)}\n")
    summary.append(f"- Matches: {len(matches)}\n")
    summary.append(f"- Mismatches: {len(mismatches)}\n\n")

    if missing:
        summary.append("## Missing Files in Unmodified\n")
        summary.extend(f"- {m}\n" for m in missing)
        summary.append("\n")
    if extra:
        summary.append("## Extra Files in Unmodified\n")
        summary.extend(f"- {e}\n" for e in extra)
        summary.append("\n")
    if mismatches:
        summary.append("## Mismatch Overview\n")
        summary.extend(f"- {m}\n" for m in mismatches)
        summary.append("\n> See `details.md` for diffs and binary/type mismatches.\n")

    summary_path.write_text("".join(summary), encoding="utf-8")
    details_text = "".join(dlines) if dlines else "# No detailed differences.\n"
    details_path.write_text(details_text, encoding="utf-8")

    import json
    machine = {
        "stats": {
            "original_files": len(orig_set),
            "unmodified_files": len(new_set),
            "missing": len(missing),
            "extra": len(extra),
            "matches": len(matches),
            "mismatches": len(mismatches),
        },
        "missing": missing,
        "extra": extra,
        "mismatches": mismatches,
        "matches": matches
    }
    json_path.write_text(json.dumps(machine, indent=2), encoding="utf-8")

# -----------------------------
# YAML writer (mapping)
# -----------------------------

def write_mapping_yaml(mapping_path: Path, mapping: Dict[str, str]) -> None:
    """
    Write YAML dictionary with single-quoted keys/values and proper escaping for single quotes.
    """
    def q(s: str) -> str:
        return "'" + s.replace("'", "''") + "'"
    lines = ["# original_to_replacement mapping\n"]
    for k in mapping:
        v = mapping[k]
        lines.append(f"{q(k)}: {q(v)}\n")
    mapping_path.write_text("".join(lines), encoding="utf-8")

def load_mapping_yaml(mapping_path: Path) -> Dict[str, str]:
    """
    Load YAML dictionary with format: key: value
    Only supports simple single-line key/value pairs (single-quoted or bare).
    """
    if not mapping_path.exists():
        raise SystemExit(f"Error: mapping file not found: {mapping_path}")
    mapping: Dict[str, str] = {}
    with mapping_path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # very simple parser: split on first ':'
            if ":" not in line:
                continue
            k,v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            # strip quotes if present
            if (k.startswith("'") and k.endswith("'")) or (k.startswith('"') and k.endswith('"')):
                k = k[1:-1].replace("''","'")
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                v = v[1:-1].replace("''","'")
            mapping[k] = v
    if not mapping:
        raise SystemExit(f"Error: mapping file empty or unreadable: {mapping_path}")
    return mapping

# -----------------------------
# CLI Orchestration
# -----------------------------

def derive_names_for_paths(input_path: Path, restore_only: bool, modified_override: Path=None) -> Tuple[Path, Path, Path, str]:
    """
    Returns (original_root, modified_root, unmodified_root, original_name)
    If restore_only:
      - if input_path name ends with '_modified', treat it as modified_root and derive original_root/unmodified_root accordingly.
      - else if --modified given, use that as modified_root and treat input_path as original root.
    If not restore_only:
      - input_path is original_root; derive modified/unmodified.
    """
    if restore_only:
        if input_path.name.endswith("_modified"):
            mod_root = input_path
            original_name = input_path.name[:-9]  # strip "_modified"
            orig_parent = input_path.parent
            orig_root_guess = orig_parent / original_name
            unmod_root = orig_parent / f"{original_name}_unmodified"
            return (orig_root_guess, mod_root, unmod_root, original_name)
        else:
            if modified_override is None:
                # interpret input as modified_root if it exists; else error
                if input_path.exists() and input_path.is_dir():
                    mod_root = input_path
                    name = input_path.name
                    original_name = name[:-9] if name.endswith("_modified") else name
                    unmod_root = input_path.parent / f"{original_name}_unmodified"
                    orig_root_guess = input_path.parent / original_name
                    return (orig_root_guess, mod_root, unmod_root, original_name)
                else:
                    raise SystemExit("Error: in --restore-only mode, provide a modified folder (positional) or use --modified.")
            else:
                # input is original, override provides modified
                mod_root = modified_override
                original_name = input_path.name
                unmod_root = mod_root.parent / f"{original_name}_unmodified"
                return (input_path, mod_root, unmod_root, original_name)
    else:
        original_name = input_path.name
        mod_root = input_path.parent / f"{original_name}_modified"
        unmod_root = input_path.parent / f"{original_name}_unmodified"
        return (input_path, mod_root, unmod_root, original_name)

def main():
    parser = argparse.ArgumentParser(description="Replace/reverse words in a folder with case-preserving mapping.")
    parser.add_argument("folder", type=str, help="Path to original folder (or modified folder when --restore-only).")
    parser.add_argument("--restore-only", action="store_true", help="Perform only the reverse operation from a modified tree.")
    parser.add_argument("--modified", type=str, default=None, help="Path to the modified folder (used with --restore-only when 'folder' is the original).")
    parser.add_argument("--mapping", type=str, default=None, help="Path to mapping YAML (defaults to script_dir/<original>_replacment_words.yaml).")
    parser.add_argument("--output", type=str, default=None, help="Destination for unmodified output (optional override).")
    args = parser.parse_args()

    input_path = Path(args.folder).resolve()
    assert_dir_exists(input_path, "input folder")

    # Derive roots and original name
    orig_root, mod_root, unmod_root, original_name = derive_names_for_paths(
        input_path, args.restore_only, Path(args.modified).resolve() if args.modified else None
    )

    # Determine locations for words/mapping next to the script
    words_path = script_dir() / f"{original_name}_words.yaml"
    mapping_path_default = script_dir() / f"{original_name}_replacment_words.yaml"
    mapping_path = Path(args.mapping).resolve() if args.mapping else mapping_path_default

    # If output override is given, use it
    if args.output:
        unmod_root = Path(args.output).resolve()

    if args.restore_only:
        # Reverse only
        print(f"[RESTORE] Loading mapping from: {mapping_path}")
        mapping = load_mapping_yaml(mapping_path)
        print(f"[RESTORE] Reversing from: {mod_root} -> {unmod_root}")
        reverse_transform(mod_root, unmod_root, mapping)
        print(f"[RESTORE] Done. Unmodified: {unmod_root}")
        return

    # Forward pipeline
    assert_dir_exists(orig_root, "original folder")

    # 1) Load and validate words
    print(f"[1/7] Loading words from: {words_path}")
    words = load_words_yaml(words_path)
    print(f"[1/7] Validating {len(words)} words (no duplicates / no substrings)...")
    validate_word_list(words)

    # 2) Generate mapping (exact-length, collision-free vs original corpus)
    print(f"[2/7] Scanning corpus and generating replacements...")
    mapping = generate_mapping(words, orig_root)

    # 3) Write mapping next to the script (as required)
    print(f"[3/7] Writing mapping to: {mapping_path}")
    write_mapping_yaml(mapping_path, mapping)

    # 4) Copy & transform to modified
    print(f"[4/7] Creating transformed copy at: {mod_root}")
    copy_and_transform(orig_root, mod_root, mapping)

    # 5) Reverse into unmodified
    print(f"[5/7] Restoring into: {unmod_root}")
    reverse_transform(mod_root, unmod_root, mapping)

    # 6) Compare and write report
    report_root = orig_root.parent / f"{original_name}_compare_report"
    print(f"[6/7] Comparing original vs unmodified -> report at: {report_root}")
    compare_and_report(orig_root, unmod_root, report_root)

    print(f"[7/7] Done.\n- Modified:   {mod_root}\n- Unmodified: {unmod_root}\n- Mapping:    {mapping_path}\n- Report:     {report_root}")

if __name__ == "__main__":
    random.seed()
    main()
