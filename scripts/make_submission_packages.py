"""Build local (unpublished) submission packages: the arXiv source bundle (named preprint) and the anonymised review mirror.

    python scripts/make_submission_packages.py arxiv  --out dist/    # needs latexmk/pdflatex/bibtex with lmodern
    python scripts/make_submission_packages.py mirror --out dist/    # git + python only

Nothing is uploaded or pushed. arxiv: copies the preprint sources into a clean directory, builds them (latexmk) to produce the .bbl,
then test-compiles the bundle with pdflatex alone (what arXiv does with a shipped .bbl) and fails on undefined references/citations.
mirror: copies `git ls-files`, drops files that carry the author's identity (LICENSE, CITATION.cff, preprint sources and PDFs),
redacts identifying strings in the remaining text files and scans the result; it fails if any identifier survives.
Frozen artifacts contain no identifiers and are copied byte for byte. The mirror has no git history, so the audit steps that read git
(provenance audit) cannot run in it; the numeric audit and the unit tests can.
"""
import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARXIV_FILES = ["main_preprint.tex", "body.tex", "math_commands.tex", "preamble_common.tex", "tmlr.sty", "tmlr.bst", "fancyhdr.sty", "refs.bib",
               "figures/fig1_rule_dependent_operating_points.pdf"]
IDENT = re.compile(r"mahesh|pagadala|lonewolf|aston|gmail|sunny|reddy", re.I)
SUBS = [(re.compile(r"Mahesh\s+Reddy(\s+Pagadala)?"), "the author"), (re.compile(r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/\"'\r\n]+"), "<USER_HOME>"), (re.compile(r"/sessions/[^/\s\"']+"), "<SESSION>"),
        (re.compile(r"https://github\.com/lonewolf15116/lookahead-halting"), "<ANONYMOUS_REPOSITORY>"), (re.compile(r"lonewolf15116"), "anonymous"),
        (re.compile(r"[\w.+-]+@gmail\.com"), "<EMAIL>"), (re.compile(r"[\w.+-]+@aston\.ac\.uk"), "<EMAIL>"),
        (re.compile(r"Mahesh Reddy Pagadala|Mahesh Reddy|Pagadala"), "the author"), (re.compile(r"Aston University"), "the author's institution"),
        (re.compile(r"\bSunny\b"), "the author")]
DROP = {"scripts/make_submission_packages.py", "scripts/step4_manuscript_consistency.py", "scripts/step4_release_readiness.py", "LICENSE", "CITATION.cff", "paper/main_preprint.tex", "paper/main_preprint.pdf", "paper/main_preprint.aux", "paper/main_preprint.bbl",
        "results/audit/final/main_preprint.pdffonts.txt", "results/audit/post_rebuild/main_preprint.pdffonts.txt"}
TEXT_EXT = {".py", ".ps1", ".md", ".txt", ".json", ".csv", ".yaml", ".yml", ".toml", ".tex", ".bib", ".cff", ".sha256", ".cfg", ""}


def run(cmd, cwd):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def arxiv(out: Path) -> Path:
    with tempfile.TemporaryDirectory() as t:
        w = Path(t) / "src"
        for f in ARXIV_FILES:
            (w / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "paper" / f, w / f)
        rc, log = run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main_preprint.tex"], w)
        if rc:
            sys.exit("latexmk failed:\n" + log[-2000:])
        shutil.copy2(w / "main_preprint.bbl", w / "main_preprint.bbl.keep")
        for p in list(w.glob("main_preprint.*")):
            if p.suffix != ".tex" and p.name != "main_preprint.bbl.keep":
                p.unlink()
        (w / "main_preprint.bbl.keep").rename(w / "main_preprint.bbl")
        (w / "refs.bib").unlink()                              # arXiv uses the shipped .bbl; keep the bib out to avoid a second bibtex pass
        for _ in range(2):
            rc, log = run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "main_preprint.tex"], w)
            if rc:
                sys.exit("pdflatex (bbl only) failed:\n" + log[-2000:])
        lg = (w / "main_preprint.log").read_text(errors="replace")
        if re.search(r"undefined", lg):
            sys.exit("undefined references/citations in the arXiv test build")
        (w / "main_preprint.pdf").rename(w / "check_main_preprint.pdf")
        out.mkdir(parents=True, exist_ok=True)
        z = out / "arxiv_source.zip"
        if z.exists():
            z.unlink()
        with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
            for p in sorted(w.rglob("*")):
                if p.is_file() and p.suffix not in {".aux", ".log", ".out", ".pdf", ".blg", ".fls", ".fdb_latexmk"} or p.suffix == ".pdf" and p.parent.name == "figures":
                    zf.write(p, p.relative_to(w).as_posix())
        shutil.copy2(w / "check_main_preprint.pdf", out / "arxiv_test_build.pdf")
    print("arXiv bundle:", z)
    return z


def mirror(out: Path) -> Path:
    files = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines()
    dst = out / "anonymous_mirror"
    if dst.exists():
        shutil.rmtree(dst)
    n_red = 0
    for f in files:
        if f in DROP:
            continue
        s, d = ROOT / f, dst / f
        d.parent.mkdir(parents=True, exist_ok=True)
        if s.suffix in TEXT_EXT:
            try:
                txt = s.read_text(encoding="utf-8", newline="")
            except UnicodeDecodeError:
                shutil.copy2(s, d)
                continue
            new = txt
            for rx, rep in SUBS:
                new = rx.sub(rep, new)
            n_red += new != txt
            d.write_bytes(new.encode("utf-8"))
        else:
            shutil.copy2(s, d)
    (dst / "README_ANONYMOUS.md").write_text("Anonymised review copy of the project repository (no git history, author names, handles, e-mail or local paths).\n"
        "Redacted text is marked <USER_HOME>, <ANONYMOUS_REPOSITORY>, <EMAIL> or 'the author'. Frozen result artifacts are byte-identical to the originals.\n"
        "Three tooling scripts that contain identity search patterns are omitted. Provenance checks and tests that read git history (provenance audit, tests/test_c1_test.py marker test) cannot run here; `python scripts/step4_numeric_audit.py` and `python -m pytest` can.\n", encoding="utf-8")
    bad = []
    for p in dst.rglob("*"):
        if p.is_file():
            try:
                t = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if IDENT.search(t):
                bad.append(str(p.relative_to(dst)))
    for p in dst.rglob("*.pdf"):                                   # rendered text of any PDF
        rc, txt = run(["pdftotext", str(p), "-"], dst)
        if rc == 0 and IDENT.search(txt):
            bad.append(str(p.relative_to(dst)) + " (pdf text)")
    if bad:
        sys.exit("identifiers survive redaction in: " + ", ".join(bad[:20]))
    out.mkdir(parents=True, exist_ok=True)
    z = out / "anonymous_mirror.zip"
    if z.exists():
        z.unlink()
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(dst.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(dst).as_posix())
    print(f"anonymous mirror: {z} ({len(files) - len(DROP & set(files))} files, {n_red} text files redacted, identifier scan clean)")
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["arxiv", "mirror"])
    ap.add_argument("--out", default=str(ROOT / "dist"))
    a = ap.parse_args()
    (arxiv if a.what == "arxiv" else mirror)(Path(a.out))


if __name__ == "__main__":
    main()
