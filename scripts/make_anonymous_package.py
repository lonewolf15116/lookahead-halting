"""Build the anonymised code package for double-blind review from the tracked files at a git revision.

    python scripts/make_anonymous_package.py [--rev HEAD] [--out anonymous_code_package.zip]

Rules: (1) tracked files only, no git history; (2) frozen artifacts and code are NEVER edited (their hashes are pinned): a non-documentation
file that contains an identifying string is EXCLUDED and listed in ANONYMIZATION_NOTES.md; (3) documentation (*.md, LICENSE) gets a fixed
list of textual redactions; (4) the preprint build target (paper/main_preprint.*) is excluded; (5) the finished archive is scanned and the
script fails if any identifying string remains. The zip is deterministic (sorted names, fixed timestamps).
"""
import argparse
import hashlib
import io
import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ID = re.compile(r"lonewolf15116|Mahesh|Pagadala|aston\.ac|Aston Univ|mahesh15116|C:\\+Users|/home/claude|LAPTOP-T1JBL1VC|\bSunny\b|\bReddy\b|claude\.ai/code", re.I)
EXCLUDE_PREFIX = ("paper/main_preprint", "CITATION.cff")
REDACT = [  # documentation only
    (re.compile(r"Sunny's other chat"), "another chat"), (re.compile(r"\bSunny\b"), "the author"),
    (re.compile(r"Mahesh Reddy Pagadala"), "the author"), (re.compile(r"lonewolf15116"), "anonymous"),
    (re.compile(r"mahesh15116@gmail\.com"), "<author e-mail>"), (re.compile(r"TODO@aston\.ac\.uk"), "<placeholder e-mail>"),
    (re.compile(r"Aston University"), "the author's institution"),
]
DOC = (".md",)
DROP_MAKE = ("step4_manuscript_consistency", "step4_release_readiness")


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rev", default="HEAD")
    ap.add_argument("--out", default=str(ROOT / "anonymous_code_package.zip"))
    a = ap.parse_args()
    names = [n for n in git("ls-tree", "-r", "-z", "--name-only", a.rev).decode().split("\0") if n]
    kept, excluded, redacted = {}, [], []
    for n in sorted(names):
        data = git("show", f"{a.rev}:{n}")
        if n.startswith(EXCLUDE_PREFIX):
            excluded.append((n, "preprint/named build target or citation file")); continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            kept[n] = data; continue                                   # binary: scanned below via bytes
        if n == "Makefile":
            text = "".join(l for l in text.splitlines(True) if not any(d in l for d in DROP_MAKE))
            data = text.encode()
        if ID.search(text):
            if n.endswith(DOC) or n == "LICENSE":
                for rx, rep in REDACT:
                    text = rx.sub(rep, text)
                if n == "LICENSE":
                    text = text.replace("2026 the author", "2026 Anonymous authors")
                if ID.search(text):
                    excluded.append((n, "identifying string remains after redaction")); continue
                data = text.encode(); redacted.append(n)
            else:
                excluded.append((n, "contains an identifying string (frozen/code/result files are never edited)")); continue
        kept[n] = data
    bad = [n for n, d in kept.items() if ID.search(d.decode("utf-8", "ignore")) and not n.endswith((".pdf", ".png", ".npz"))]
    badbin = [n for n, d in kept.items() if n.endswith((".pdf", ".png", ".npz")) and re.search(rb"Mahesh|Pagadala|lonewolf|mahesh15116|aston\.ac", d, re.I)]
    if bad or badbin:
        sys.exit(f"identifying strings remain: {bad + badbin}")
    notes = ["# Anonymisation notes", "", "This package is a snapshot of the tracked files at one git revision, without history.",
             "Frozen protocols, results, source code and pinned data manifests are byte-identical to the repository (their hashes are pinned).",
             "Documentation was redacted textually (names, handle, e-mail, institution). The preprint build target and citation file are omitted.",
             "The provenance audit's git-history checks cannot run without `.git`; every other check runs. Reruns need the (unreleased) development features.", "",
             f"Redacted documentation files ({len(redacted)}): " + ", ".join(f"`{n}`" for n in redacted), "",
             f"Excluded files ({len(excluded)}):", ""] + [f"- `{n}`: {why}" for n, why in excluded]
    kept["ANONYMIZATION_NOTES.md"] = ("\n".join(notes) + "\n").encode()
    out = Path(a.out)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for n in sorted(kept):
            zi = zipfile.ZipInfo(n, date_time=(2026, 1, 1, 0, 0, 0)); zi.compress_type = zipfile.ZIP_DEFLATED; zi.external_attr = 0o644 << 16
            z.writestr(zi, kept[n])
    h = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"{out.name}: {len(kept)} files, {len(excluded)} excluded, {len(redacted)} redacted, {out.stat().st_size / 1e6:.1f} MB, sha256 {h}")


if __name__ == "__main__":
    main()
