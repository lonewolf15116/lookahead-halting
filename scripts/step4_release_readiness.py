"""Step 4 release-readiness scan: what would stop an independent researcher from reproducing the findings? Reads tracked files only.

    python scripts/step4_release_readiness.py [--out results/audit/step4_release_readiness.json]

Each finding has a severity (BLOCKER / MAJOR / MINOR / INFO), the evidence and a proposed remedy. Exit code is always 0 (this is an inventory).
"""
import argparse
import sys
import json
import re
import subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
F = []


def add(fid, severity, finding, evidence, remedy):
    F.append({"id": fid, "severity": severity, "finding": finding, "evidence": evidence, "remedy": remedy})


def git(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout


tracked = [p for p in git("ls-files").splitlines() if p]
text_ext = (".py", ".ps1", ".md", ".yaml", ".yml", ".toml", ".txt", ".tex", ".bib", ".json", ".csv")
SELF = {"scripts/step4_release_readiness.py", "scripts/step4_manuscript_consistency.py"}


def scan(pattern, exts=text_ext, skip=()):
    rx = re.compile(pattern)
    hits = {}
    for p in tracked:
        if not p.endswith(exts) or p in SELF or any(p.startswith(s) for s in skip):
            continue
        try:
            t = (ROOT / p).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        n = len(rx.findall(t))
        if n:
            hits[p] = n
    return hits


# 1 machine-specific paths ---------------------------------------------------------------------------------------------
h = scan(r"C:\\\\?Users|C:/Users|\$HOME[\\/]+Projects|/home/claude|/sessions/|Mahesh Reddy\\\\")
code_hits = {k: v for k, v in h.items() if k.startswith(("scripts/", "src/", "tests/", "configs/"))}
add("RR.paths.code", "MAJOR" if code_hits else "PASS", "machine-specific absolute paths in code, scripts and configs", json.dumps(code_hits) or "none",
    "replace by paths relative to REPO_ROOT or a documented environment variable; the .ps1 runners assume $HOME\\Projects\\lookahead-halting")
add("RR.paths.docs", "MINOR" if any(k.startswith(("docs/", "README")) for k in h) else "PASS", "machine-specific paths in documentation/README (informational)",
    json.dumps({k: v for k, v in h.items() if not k.startswith(("scripts/", "src/", "tests/", "configs/"))})[:600], "keep in the decision log; remove from README/paper-facing text")
ps1 = [p for p in tracked if p.endswith(".ps1")]
add("RR.windows_only", "PASS" if "Makefile" in tracked else "MAJOR", "all one-command runners are Windows PowerShell scripts; there is no POSIX equivalent", f"{len(ps1)} .ps1 runners, 0 .sh",
    "add a Makefile / run_all.sh that executes the same python entry points (the Python scripts themselves ran unchanged on Linux in this audit)")

# 2 documentation ------------------------------------------------------------------------------------------------------
readme = (ROOT / "README.md").read_text(encoding="utf-8")
stale = "Working title" in readme and "Phase 0" in readme
add("RR.readme", "MAJOR" if stale else "PASS", "README describes the project as of Phase 0 (old working title, 'Next: Phase 1'); it has no reproduction instructions for the paper",
    "README.md mentions 'Gate 0', 'Next: Phase 1' and the title 'Predicting the Value of Another Thought…'",
    "rewrite: paper title, one table 'claim -> command -> artifact -> hash', environment, hardware, expected runtimes, what cannot be rerun and why")
add("RR.license", "MAJOR" if not any(Path(p).name.upper().startswith(("LICENSE", "COPYING")) for p in tracked) else "PASS", "no LICENSE file", "no LICENSE/COPYING in the repository", "add a licence (and a data licence note for derived features)")
add("RR.citation", "MINOR", "no CITATION.cff / release tag for the paper version", "tags: " + " ".join(git("tag", "-l").split()), "tag the submission commit (e.g. paper-v1); the cloud session cannot push tags, so tag from the laptop")

# 3 environment --------------------------------------------------------------------------------------------------------
req = (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
unpinned = [l for l in req if l.strip() and not l.startswith("#") and "==" not in l]
env = json.loads((ROOT / "results/gate0/environment.json").read_text(encoding="utf-8"))
add("RR.pins", "PASS" if "requirements-lock.txt" in tracked else "MAJOR", "requirements.txt pins only transformers and lm_eval; numpy/scipy/scikit-learn/pandas/pyarrow/torch are unpinned",
    f"unpinned: {unpinned}; actual environment recorded in results/gate0/environment.json: Python {env['python'].split()[0]}, torch {env['packages']['torch']}, numpy {env['packages']['numpy']}, "
    f"scikit-learn {env['packages']['scikit-learn']}",
    "generate a lock file (pip freeze) from the laptop environment and commit it; state the tested range. This audit reproduced results on numpy 2.2.6 / scikit-learn 1.7.2 / Python 3.10 with max deviations of 1e-5 (see reproducibility section)")
add("RR.gpu", "INFO", "feature extraction needs a CUDA GPU (RTX 5050 laptop, bf16); GPU runs are not bitwise reproducible", "D-t2: 16/20 questions differ across default repeats; deterministic mode removes differences",
    "document `torch.use_deterministic_algorithms(True)` + CUBLAS_WORKSPACE_CONFIG=:4096:8 as the reproducibility mode; state that stored features, not re-extraction, are the reproducible unit")

# 4 data ---------------------------------------------------------------------------------------------------------------
ign = (ROOT / ".gitignore").read_text(encoding="utf-8")
add("RR.features", "BLOCKER", "the per-question extracted features (hidden states, option scores; ~0.9 GB for development, TEST separately) are git-ignored and not released",
    "ignored: data/extracted/, data/extracted_confirm/, *.npy, *.parquet; results/*.json hold only summaries",
    "release the development and (after the paper is accepted or at submission, with the sealed-TEST rule documented) TEST feature tables on Zenodo/Hugging Face with the recorded SHA-256 (256 development files are hash-pinned in frozen.json); without them every number requires GPU re-extraction")
add("RR.test_unrepro", "MAJOR", "C1 TEST numbers and their bootstrap intervals cannot be recomputed by anyone without the TEST feature files", f"test_report.json pins {len(json.loads((ROOT / 'results/c1/test_report.json').read_text())['provenance']['test_data_files'])} TEST file hashes but the files are not distributed",
    "release the TEST per-question (exit-loop, correctness) arrays after unsealing so the 10,000-resample intervals can be regenerated; keep the one-time-run marker as evidence")
add("RR.datasets", "INFO", "benchmark datasets are downloaded from the Hugging Face hub at pinned revisions recorded in the manifests", "data/manifests/*.json, results/phase3/inventory.json", "state dataset licences in the README")
dirty = git("status", "--porcelain").splitlines()
add("RR.untracked", "MINOR", "untracked or modified files exist in the working tree used for this audit", json.dumps(dirty[:10]), "commit or remove before tagging the release")

# 5 reproduction commands present --------------------------------------------------------------------------------------
need = {"C1 freeze": "scripts/c1_freeze.py", "C1 TEST run (one-time, sealed)": "scripts/c1_test_run.py", "E1": "scripts/e1_run.py", "X1": "scripts/x1_flip_check.py", "G3a": "scripts/phase3_g3a.py",
        "D-t": "scripts/dt_truncation_check.py", "D-t2": "scripts/dt2_determinism_check.py"}
missing = {k: v for k, v in need.items() if v not in tracked}
add("RR.entrypoints", "PASS" if not missing else "MAJOR", "every analysis in the paper has a committed entry point", json.dumps(missing) if missing else f"{len(need)} entry points present",
    "map each to a paper table/figure in the README")
add("RR.figure_script", "MAJOR" if not any("fig1" in p and p.endswith(".py") for p in tracked) else "PASS", "the script that draws Figure 1 is committed",
    "figures/fig1_*.png/.pdf are committed; no tracked .py mentions fig1" if not any("fig1" in p and p.endswith(".py") for p in tracked) else "present",
    "commit the plotting script (it reads results/c1/test_report.json) so Figure 1 is regenerable")
add("RR.table_script", "MINOR", "tables in the paper are typed by hand from JSON; no generator script", "docs/PAPER_NUMBER_AUDIT.md and this audit's scripts/step4_numeric_audit.py recompute them",
    "ship scripts/step4_numeric_audit.py as `make check-numbers`")
add("RR.sealed", "INFO", "the TEST set is spent; a future reader cannot rerun the confirmatory study on fresh data from this repository alone", "docs/C1_REPORT.md; marker TEST_OPENED.json",
    "state this in the README; point to the pre-registration (tag c1-protocol)")
add("RR.anonymous_mirror", "PASS" if "scripts/make_anonymous_package.py" in tracked else "MAJOR", "TMLR double-blind review needs an anonymous code mirror; the GitHub repository is private and its owner handle appears in remotes/docs", "origin: github.com/lonewolf15116/lookahead-halting",
    "create an anonymised snapshot (anonymous.4open.science or a zip) without .git history, DECISIONS.md author references and the local paths above")
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "results/audit/step4_release_readiness.json"))
args = ap.parse_args()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps({"script": "scripts/step4_release_readiness.py", "findings": F}, indent=1) + "\n", encoding="utf-8")
for f in F:
    print(f"{f['severity']:8s} {f['id']:20s} {f['finding'][:110]}")
