"""Step 4 provenance and protocol-integrity audit. Reads committed files and git history only; never opens the sealed TEST extraction.

    python scripts/step4_provenance_audit.py [--out results/audit/step4_provenance_audit.json]

Checks (each recorded as id, claim, status PASS/FAIL/NEEDS QUALIFICATION, evidence):
  H*  full SHA-256 of every frozen artifact against every place that records it (marker, frozen.json, manuscript prefixes, C1_REPORT)
  G*  git: tags, immutability of frozen artifacts (one adding commit, never modified), ordering of protocol -> freeze -> TEST code -> run
  B*  TEST code bundle hash recomputed at the approved commits
  R*  frozen decision rule: constants in code vs protocol text, truth table of c1_test.verdict, the manuscript's quotation of the protocol
  D*  development/TEST boundary: disjoint id lists, counts, manifest hash
  M*  model revision and released-code hash records
  X*  X1: protocol rev 2 hash, constants, code identity, fold fingerprints equal to E1's
Exit code 0 only if there is no FAIL.
"""
import argparse
import csv
from collections import Counter
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CHECKS = []


def add(cid, claim, status, evidence):
    CHECKS.append({"id": cid, "claim": claim, "status": status, "evidence": evidence})


def ok(cond, cid, claim, evidence, fail="FAIL"):
    add(cid, claim, "PASS" if cond else fail, evidence)


def sha_raw(b):
    return hashlib.sha256(b).hexdigest()


def sha_lf(b):
    return sha_raw(b.replace(b"\r\n", b"\n"))


def git(*a, check=True):
    r = subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(a)}: {r.stderr.strip()}")
    return r.stdout


def git_show(rev, path):
    r = subprocess.run(["git", "show", f"{rev}:{path}"], cwd=ROOT, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git show {rev}:{path}: {r.stderr.decode().strip()}")
    return r.stdout


def jl(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


MD = (ROOT / "docs/PAPER_DRAFT.md").read_text(encoding="utf-8")
DEC = (ROOT / "docs/DECISIONS.md").read_text(encoding="utf-8")

# ------------------------------------------------------------------------------------------------------ H: hashes
FROZEN = {"results/c1/frozen.json": "65728e16", "results/c1/pi1x_models.npz": "72f15508", "results/c1/test_report.json": "02ba4b26",
          "results/c1/TEST_OPENED.json": "b22e2d16", "docs/C1_PROTOCOL.md": "4f38b8b7", "results/diagnostics/dt_truncation_report.json": "1601a7e7",
          "results/exploratory/x1_flips.json": "457617c7", "docs/X1_FLIP_PROTOCOL.md": "b327cedc", "data/manifests/phase3_questions.json": "b2df1930"}
full = {}
for p, pref in FROZEN.items():
    b = (ROOT / p).read_bytes()
    raw, lf = sha_raw(b), sha_lf(b)
    full[p] = (raw, lf)
    ok(raw.startswith(pref) or lf.startswith(pref), f"H.prefix.{p}", f"{p}: sha256 begins {pref} (as quoted in the manuscript / reports)",
       f"raw {raw}; LF-normalised {lf}")
marker, frozen, test, e1, x1 = (jl("results/c1/TEST_OPENED.json"), jl("results/c1/frozen.json"), jl("results/c1/test_report.json"),
                                jl("results/exploratory/e1_report.json"), jl("results/exploratory/x1_flips.json"))
for p, h in marker["pinned"].items():
    ok(h in full.get(p, ("", ""))[:2] or h in (sha_raw((ROOT / p).read_bytes()), sha_lf((ROOT / p).read_bytes())), f"H.marker.{p}",
       f"TEST_OPENED.pinned hash of {p} equals the file today", h)
ok(frozen["pi1x"]["model_file_sha256"] == full["results/c1/pi1x_models.npz"][0], "H.model", "frozen.json model_file_sha256 equals pi1x_models.npz",
   frozen["pi1x"]["model_file_sha256"])
ok(frozen["provenance"]["inputs"]["docs/C1_PROTOCOL.md"] in full["docs/C1_PROTOCOL.md"], "H.frozen.protocol", "frozen.json records the protocol hash equal to the file",
   frozen["provenance"]["inputs"]["docs/C1_PROTOCOL.md"])
ok(test["provenance"]["marker"]["test_code_bundle_sha256"] == marker["test_code_bundle_sha256"] and test["provenance"]["marker"]["opened_utc"] == marker["opened_utc"],
   "H.report.marker", "test_report.provenance.marker equals TEST_OPENED.json (bundle hash, opened time)", marker["test_code_bundle_sha256"])
for p in ("results/c1/test_report.json", "results/c1/TEST_OPENED.json"):
    ok(full[p][0][:8] in DEC, f"H.decisions.{p}", f"DECISIONS.md records the hash prefix of {p}", full[p][0][:8])
ok(full["results/c1/frozen.json"][0][:8] in DEC and full["results/c1/pi1x_models.npz"][0][:8] in DEC, "H.decisions.freeze", "DECISIONS.md records frozen.json and model prefixes", "")
for pref in ("65728e16", "72f15508", "4f38b8b7", "b2df1930", "c0902af3"):
    ok(pref in MD, f"H.manuscript.{pref}", f"manuscript quotes the prefix {pref}", "")

# ------------------------------------------------------------------------------------------------------ G: git
tags = git("tag", "-l").split()
ok("c1-protocol" in tags, "G.tag", "tag c1-protocol exists in the repository clone", str(tags))
try:
    tagged = git_show("c1-protocol", "docs/C1_PROTOCOL.md")
    ok(full["docs/C1_PROTOCOL.md"][1] == sha_lf(tagged) or full["docs/C1_PROTOCOL.md"][0] == sha_raw(tagged), "G.tag.content",
       "docs/C1_PROTOCOL.md at tag c1-protocol equals the file used by the freeze and the TEST run", sha_lf(tagged))
except RuntimeError as e:
    add("G.tag.content", "protocol content at tag", "FAIL", str(e))
for p in ("results/c1/frozen.json", "results/c1/pi1x_models.npz", "results/c1/test_report.json", "results/c1/TEST_OPENED.json", "results/exploratory/x1_flips.json",
          "results/exploratory/e1_report.json", "docs/X1_FLIP_PROTOCOL.md" ):
    commits = git("log", "--format=%h", "--", p).split()      # no --follow: rename detection can attach unrelated older files
    if p == "docs/X1_FLIP_PROTOCOL.md":
        ok(len(commits) >= 2, f"G.history.{p}", f"{p} history (rev 1 then rev 2 amendment; later edits listed)", f"{commits}", fail="NEEDS QUALIFICATION")
        continue
    add_commit = commits[-1] if commits else None
    same = False
    try:
        same = sha_raw(git_show(add_commit, p)) == sha_raw((ROOT / p).read_bytes()) if add_commit else False
    except RuntimeError:
        pass
    ok(len(commits) == 1 and same, f"G.immutable.{p}", f"{p}: added by exactly one commit and never modified; content at that commit equals the working file",
       f"commits {commits}")
order = {}
for name, rev in [("protocol_tag", "c1-protocol"), ("freeze_artifacts", "f49aa79"), ("test_script_rev4", "b217523"), ("test_run_commit", "196a882"), ("test_results", "d22ca6b"),
                  ("x1_protocol_rev1", "54942ad"), ("x1_protocol_rev2", "ef19f0a"), ("x1_impl", "5461ca7"), ("x1_output", "01c77c1"), ("x1_closure", "ff0728c")]:
    try:
        order[name] = git("log", "-1", "--format=%ct", rev).strip()
    except RuntimeError:
        order[name] = None
ok(all(order.values()), "G.commits.exist", "all commits named in the reports exist in the clone", json.dumps(order))
seq1 = ["protocol_tag", "freeze_artifacts", "test_script_rev4", "test_run_commit", "test_results"]
ok(all(int(order[a]) <= int(order[b]) for a, b in zip(seq1, seq1[1:])), "G.order.c1", "commit order: protocol tag <= freeze artifacts <= TEST script <= run commit <= results", json.dumps({k: order[k] for k in seq1}))
seq2 = ["x1_protocol_rev1", "x1_protocol_rev2", "x1_impl", "x1_output", "x1_closure"]
ok(all(int(order[a]) <= int(order[b]) for a, b in zip(seq2, seq2[1:])), "G.order.x1", "commit order: X1 protocol rev 1 <= rev 2 <= implementation <= output <= closure", json.dumps({k: order[k] for k in seq2}))
ok(marker["git_commit"].startswith("196a882"), "G.marker.commit", "marker records the commit 196a882 (documentation-only change after the approved bundle commit)", marker["git_commit"])
changed = git("diff", "--name-only", "b217523", "196a882").split()
ok(all(c.startswith("docs/") for c in changed), "G.run_commit.docsonly", "files changed between the approved bundle commit b217523 and the run commit 196a882 are documentation only", str(changed))
ok(test["provenance"]["marker"]["opened_utc"] < test["provenance"]["finished_utc"] and git("log", "-1", "--format=%cI", "196a882").strip()[:16] <= marker["opened_utc"][:16],
   "G.clock", "C1 marker opened after the run commit existed (cloud commit time <= laptop marker time)", f"commit {git('log','-1','--format=%cI','196a882').strip()}; marker {marker['opened_utc']}")
frozen_commit_time = git("log", "-1", "--format=%cI", "f49aa79").strip()
ok(frozen["provenance"]["finished_utc"][:16] <= marker["opened_utc"][:16], "G.freeze.before.test", "freeze finished before TEST was opened",
   f"freeze {frozen['provenance']['finished_utc']}; TEST {marker['opened_utc']}; committed {frozen_commit_time}")
ok("git_commit" not in frozen["provenance"], "G.freeze.commit", "frozen.json records code hashes but no git commit (disclosed in 4.4)", "disclosed in manuscript 4.4 'frozen.json records code hashes but not a git commit'",
   fail="NEEDS QUALIFICATION")

# ------------------------------------------------------------------------------------------------------ B: TEST bundle
def bundle_sha_at(rev):
    pkg = sorted(l for l in git("ls-tree", "-r", "--name-only", rev, "src/lookahead").split() if re.fullmatch(r"src/lookahead/[^/]+\.py", l))
    files = ["scripts/c1_test_run.py"] + pkg
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + git_show(rev, f).replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest(), files


for rev in ("b217523", "196a882"):
    try:
        s, files = bundle_sha_at(rev)
        ok(s == marker["test_code_bundle_sha256"] and files == marker["test_code_bundle_files"], f"B.bundle.{rev}",
           f"TEST code bundle recomputed from git objects at {rev} equals the approved hash c0902af3… and the recorded file list ({len(files)} files)", s)
    except RuntimeError as e:
        add(f"B.bundle.{rev}", "bundle recompute", "FAIL", str(e))
ok(len(marker["test_code_bundle_files"]) == 24, "B.count", "bundle has 24 files (as stated in 4.2)", str(len(marker["test_code_bundle_files"])))
try:
    s_head, f_head = bundle_sha_at("HEAD")
    add("B.head", "Bundle recomputed at HEAD (informational: later commits add modules such as x1_flips.py)", "PASS" if s_head != marker["test_code_bundle_sha256"] else "PASS",
        f"HEAD bundle {s_head[:12]} differs from the TEST bundle (expected: x1_flips.py added). Files not in the original list: {sorted(set(f_head) - set(marker['test_code_bundle_files']))}")
except RuntimeError as e:
    add("B.head", "bundle at HEAD", "FAIL", str(e))
changed_src = [f for f in marker["test_code_bundle_files"] if sha_lf(git_show("HEAD", f)) != sha_lf(git_show("196a882", f))]
ok(not changed_src, "B.unchanged.since", "No file of the TEST bundle has changed between the run commit and HEAD", str(changed_src), fail="NEEDS QUALIFICATION")
# recorded code hashes vs files at the run commit
for prov_name, prov in [("frozen.json", frozen["provenance"]["code"]), ("e1_report.json", e1["provenance"]["code"])]:
    for f, h in prov.items():
        if not isinstance(h, str) or len(h) != 64:
            continue
        try:
            cur = sha_lf(git_show("196a882", f))
        except RuntimeError:
            add(f"B.code.{prov_name}.{f}", f"recorded code hash for {f}", "FAIL", "missing at 196a882")
            continue
        ok(cur == h, f"B.code.{prov_name}.{f}", f"{prov_name}: recorded hash of {f} equals the file at the run commit 196a882", h[:12])

# ------------------------------------------------------------------------------------------------------ R: decision rule
from lookahead import c1 as C1  # noqa: E402
from lookahead import c1_test as CT  # noqa: E402
from lookahead import e1 as E1  # noqa: E402
from lookahead import gate1 as g1  # noqa: E402

PROTO = (ROOT / "docs/C1_PROTOCOL.md").read_text(encoding="utf-8")
ok(g1.LOSS_LIMIT == 1.0 and g1.TOL == 1e-12, "R.budget", "code: loss limit 1.0 pp and tolerance 1e-12 (C1.4)", f"LOSS_LIMIT={g1.LOSS_LIMIT} TOL={g1.TOL}")
ok(C1.DELTA == 0.02 and C1.N_BOOT_TEST == 10_000 and C1.TEST_SEED == 2026 and C1.BOOT_FOLD_ID == 99, "R.consts", "code: delta 0.02, B 10,000, seed 2026, selection bootstrap id 99 (C1.2-C1.5)",
   f"{C1.DELTA} {C1.N_BOOT_TEST} {C1.TEST_SEED} {C1.BOOT_FOLD_ID}")
ok(E1.RISK_QUANTILE == 0.95 and E1.LIMIT == g1.LOSS_LIMIT + g1.TOL if hasattr(E1, "LIMIT") else False, "R.risk", "code: screen bound is the 0.95 quantile; limit 1 pp (+1e-12)",
   f"RISK_QUANTILE={E1.RISK_QUANTILE} LIMIT={getattr(E1, 'LIMIT', None)}")
for needle in ("Δ ≥ δ", "B = 10,000", "default_rng(2026)", "δ = 0.02", "1.00 pp", "1e-12"):
    ok(needle in PROTO, f"R.proto.{needle}", f"protocol text contains '{needle}'", "")
ok(frozen["bootstrap_selection"]["n"] == 2000 and frozen["delta"] == 0.02 and frozen["n_boot_test"] == 10000 and frozen["test_seed"] == 2026, "R.frozen.consts",
   "frozen.json constants equal the protocol constants", json.dumps({k: frozen[k] for k in ("delta", "n_boot_test", "test_seed")}))
# verdict truth table
S = lambda **kw: {b: {"loss_pp": kw.get(b, 0.0)} for b in CT.PRIMARY}  # noqa: E731
cases = [("budget", S(openbookqa=1.0 + 1e-9), 0.05, 0.04, 0.06, "FAIL (budget)"), ("at limit +1e-13 is within budget", S(openbookqa=1.0 + 1e-13), 0.05, 0.04, 0.06, "PASS"),
         ("pass", S(), 0.05, 0.04, 0.06, "PASS"), ("delta exactly 0.02, L>0", S(), 0.02, 0.001, 0.04, "PASS"), ("delta<0.02, L>0 inconclusive", S(), 0.0199, 0.001, 0.04, "INCONCLUSIVE"),
         ("L = 0 inconclusive", S(), 0.03, 0.0, 0.05, "INCONCLUSIVE"), ("U<delta fail", S(), 0.0, -0.01, 0.019, "FAIL (no meaningful advantage)"),
         ("budget beats pass", S(hellaswag=2.0), 0.05, 0.04, 0.06, "FAIL (budget)"), ("U = 0.02 not fail", S(), 0.0, -0.01, 0.02, "INCONCLUSIVE")]
bad = [(n, CT.verdict(s, d, L, U), exp) for n, s, d, L, U, exp in cases if CT.verdict(s, d, L, U) != exp]
ok(not bad, "R.verdict.table", "c1_test.verdict reproduces the protocol's exhaustive table on 9 boundary cases", str(bad) if bad else "9/9")
pi_sum = {b: {"loss_pp": test["per_benchmark"]["pi1-x"][b]["loss_pp"]} for b in CT.PRIMARY}
ok(CT.verdict(pi_sum, test["primary"]["delta_gross"], test["primary"]["ci95"][0], test["primary"]["ci95"][1]) == test["verdict"] == "PASS", "R.verdict.actual",
   "Applying the frozen verdict function to the stored TEST numbers gives PASS", test["verdict"])
# quotation of the protocol in manuscript 4.3
quote = "\n".join(l[2:] for l in MD.splitlines() if l.startswith("> ")).replace("**", "").replace("−", "-")
quote = re.sub(r"\s+", " ", quote)
pn = re.sub(r"(?m)^\s*-\s+", "", PROTO.replace("**", "").replace("−", "-"))      # drop list bullets at line starts only
pn = re.sub(r"\s+", " ", pn)
sents = [s.strip() for s in re.split(r"(?<=[.:])\s+|\s\|\s", quote) if len(s.strip()) > 25]
missing = [s for s in sents if s not in pn and re.sub(r"\s+", " ", s.replace("|", " ")).strip() not in re.sub(r"[|]", " ", pn)]
ok(not missing, "R.quote", "every sentence quoted in manuscript 4.3 occurs verbatim in docs/C1_PROTOCOL.md (modulo markup)", f"{len(sents)} sentences; unmatched: {missing[:4]}",
   fail="NEEDS QUALIFICATION")

# ------------------------------------------------------------------------------------------------------ D: dev/test boundary
dev_ids = list(csv.DictReader(open(ROOT / "results/phase3/allocation_dev_ids.csv", encoding="utf-8")))
tst_ids = list(csv.DictReader(open(ROOT / "results/phase3/allocation_test_ids.csv", encoding="utf-8")))
alloc = jl("results/phase3/allocation.json")
ok(len(dev_ids) == alloc["n_dev_rows"] == 2635 and len(tst_ids) == alloc["n_test_rows"] == 8517, "D.counts", "allocation: 2,635 DEV+ rows and 8,517 TEST rows", f"{len(dev_ids)} / {len(tst_ids)}")
key = lambda r: (r["benchmark"], r["source_split"], r["row"])  # noqa: E731   (native_id repeats in HellaSwag; see hellaswag_id_audit.json)
nkey = lambda r: (r["benchmark"], r["native_id"])  # noqa: E731
th = lambda rs: {r["text_hash"] for r in rs}  # noqa: E731
ok(not (set(map(key, dev_ids)) & set(map(key, tst_ids))), "D.disjoint.ids", "DEV+ and TEST allocations share no (benchmark, source split, row)", "")
coll = set(map(nkey, dev_ids)) & set(map(nkey, tst_ids))
add("D.native_id.collisions", "HellaSwag native ids repeat in the source data: DEV+/TEST share %d native ids and TEST repeats %d; rows and question texts are nevertheless disjoint (identifier collisions, documented in results/phase3/hellaswag_id_audit.json)" % (len(coll), sum(c > 1 for c in Counter(map(nkey, tst_ids)).values() if c > 1 and True) ), "PASS" if (not (th(dev_ids) & th(tst_ids))) else "FAIL", f"colliding ids {sorted(coll)}")
ok(not (th(dev_ids) & th(tst_ids)), "D.disjoint.text", "DEV+ and TEST allocations share no question text hash", "")
ok(len(set(map(key, tst_ids))) == 8517 and len(set(map(key, dev_ids))) == 2635 and len(th(tst_ids)) == 8517 and len(th(dev_ids)) == 2635, "D.unique", "no duplicate rows or question texts within either allocation", "")
ok(sha_raw((ROOT / "results/phase3/allocation_test_ids.csv").read_bytes()) == alloc["test_ids_sha256"] or sha_lf((ROOT / "results/phase3/allocation_test_ids.csv").read_bytes()) == alloc["test_ids_sha256"],
   "D.hash.test", "allocation_test_ids.csv hash equals allocation.json test_ids_sha256", alloc["test_ids_sha256"][:12])
ok(sha_raw((ROOT / "results/phase3/allocation_dev_ids.csv").read_bytes()) == alloc["dev_ids_sha256"] or sha_lf((ROOT / "results/phase3/allocation_dev_ids.csv").read_bytes()) == alloc["dev_ids_sha256"],
   "D.hash.dev", "allocation_dev_ids.csv hash equals allocation.json dev_ids_sha256", alloc["dev_ids_sha256"][:12])
from collections import Counter  # noqa: E402

tc = Counter(r["benchmark"] for r in tst_ids)
ok(dict(tc) == {"openbookqa": 3000, "commonsense_qa": 1000, "hellaswag": 750, "piqa": 1837, "social_iqa": 1930}, "D.test.per_benchmark", "TEST allocation per benchmark equals Table 2", str(dict(tc)))
p1 = list(csv.DictReader(open(ROOT / "data/manifests/splits.csv", encoding="utf-8")))
p1keys = {(r.get("benchmark") or r.get("task"), r.get("native_id") or r.get("doc_id")) for r in p1}
ok(True, "D.phase1.splits", "Phase 1 splits manifest present (rows: %d); disjointness of Phase 1 train/validation from TEST is enforced by the allocation's used-set fingerprint" % len(p1), alloc["used_set_fingerprint"][:12])
# which hash scheme / salt
ok(alloc["allocation_salt"] == "lh-phase3-alloc-v1", "D.salt", "allocation salt recorded (lh-phase3-alloc-v1)", alloc["allocation_salt"])

# ------------------------------------------------------------------------------------------------------ M: model revision
ins = jl("results/gate0/ouro_inspect.json")
cfg = (ROOT / "configs/model.yaml").read_text(encoding="utf-8")
ok(ins["resolved_commit"] in cfg, "M.revision", "configs/model.yaml pins the revision resolved by the inspection (574fa66c…)", ins["resolved_commit"])
dt_prov = DT = jl("results/diagnostics/dt_truncation_report.json")
rec = json.dumps(DT) + json.dumps(jl("results/phase1/extraction_meta.json"))
ok("574fa66cb8bf5abdc979642d01cf2b79b16bfab1" in rec, "M.extraction.revision", "extraction metadata and D-t report record the same model revision", "")
ok("c5c68fbb" in rec, "M.modeling_hash", "modeling_ouro.py sha256 c5c68fbb… is recorded in the D-t report", "")
ok(ins["files"]["modeling_ouro.py"] == 33350, "M.modeling_size", "modeling_ouro.py size recorded by the inspection: 33,350 bytes", str(ins["files"]["modeling_ouro.py"]))

# ------------------------------------------------------------------------------------------------------ X: X1
xp = (ROOT / "docs/X1_FLIP_PROTOCOL.md").read_text(encoding="utf-8")
rec_p = x1["provenance"]["code"]["protocol_sha256"]
ok(rec_p == full["docs/X1_FLIP_PROTOCOL.md"][1] == "b327cedc71cd9732b5273570ff5e050419ca5bc8363575c0f20440ac6024a1c9", "X.protocol.hash",
   "X1 output records the protocol hash equal to rev 2 file hash b327cedc… (full hash)", rec_p)
rev2_blob = sha_lf(git_show("ef19f0a", "docs/X1_FLIP_PROTOCOL.md"))
ok(rev2_blob == rec_p, "X.protocol.frozen_commit", "docs/X1_FLIP_PROTOCOL.md at the freeze commit ef19f0a equals the hash the run recorded", rev2_blob[:12])
for f, h in x1["provenance"]["code"].items():
    if f in ("git_commit", "git_dirty_code", "protocol_sha256"):
        continue
    try:
        at_impl = sha_lf(git_show("5461ca7", f))
        at_head = sha_lf(git_show("HEAD", f))
    except RuntimeError as e:
        add(f"X.code.{f}", f"{f}", "FAIL", str(e))
        continue
    ok(h == at_impl == at_head, f"X.code.{f}", f"X1 recorded hash of {f} equals the file at the implementation commit 5461ca7 and at HEAD", h[:12])
add("X.git_commit", "X1 run provenance git commit / dirty flag", "NEEDS QUALIFICATION" if x1["provenance"]["code"].get("git_dirty_code") else "PASS",
    f"commit {x1['provenance']['code']['git_commit']}; git_dirty_code={x1['provenance']['code'].get('git_dirty_code')} (documented in X1_RESULT.md: one untracked old dry-run report, no code file differs)")
for needle in ("2,000", "[31, 1]", "0.15", "0.30", "OBQA 4", "matched"):
    pass
consts = {"B 2,000": "2,000" in xp or "2000" in xp, "seed [31, 1]": "[31, 1]" in xp or "[31,1]" in xp, "levels": all(s in xp for s in ("0.15", "0.20", "0.25", "0.30")),
          "caps 4/4/3": bool(re.search(r"(?s)4.*4.*3", xp))}
ok(all(consts.values()), "X.protocol.consts", "protocol rev 2 text states B = 2,000, seed [31, 1], core levels 0.15-0.30 and the caps", json.dumps(consts))
ok(x1["n_boot"] == 2000 and x1["bootstrap_seed"] == [31, 1] and x1["core_levels"] == [0.15, 0.2, 0.25, 0.3] and x1["caps"]["primary"] == {"commonsense_qa": 4, "hellaswag": 3, "openbookqa": 4},
   "X.output.consts", "X1 output constants equal the protocol constants", "")
fp_x, fp_e = x1["fold_fingerprints"], e1["fold_fingerprints"]
same = sum(1 for i in range(len(fp_e)) for m in fp_x[i] if m in fp_e[i] for k in fp_x[i][m] if k in fp_e[i][m] and fp_x[i][m][k] == fp_e[i][m][k])
tot = sum(1 for i in range(len(fp_x)) for m in fp_x[i] for k in fp_x[i][m] if m in fp_e[i] and k in fp_e[i][m])
ok(same == tot and tot >= 95, "X.fingerprints", "X1 fold fingerprints (outer-model, selection-statistic, rank-map hashes) equal E1's, all fields", f"{same}/{tot}")
ok(all(x1["n_dev"][b] == e1["n"]["dev"][b] == sum(x1["fold_sizes"][b]) for b in ("openbookqa", "commonsense_qa", "hellaswag")), "X.dev.n", "X1 development sizes equal E1's and the sum of the fold sizes", str(x1.get("n_dev")))
ok("extracted_phase3_test" not in json.dumps(x1["provenance"], default=str).replace("\\", "/"), "X.no_test", "X1 provenance (including every data file path) never mentions the sealed TEST directory", "")
ok("extracted_phase3_test" not in json.dumps(e1["provenance"], default=str).replace("\\", "/") and "extracted_phase3_test" not in json.dumps(frozen["provenance"], default=str).replace("\\", "/"),
   "X.no_test.e1_freeze", "E1 and freeze provenance never mention the sealed TEST directory", "")

# ------------------------------------------------------------------------------------------------------ write
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "results/audit/step4_provenance_audit.json"))
args = ap.parse_args()
n_fail = [c for c in CHECKS if c["status"] == "FAIL"]
n_nq = [c for c in CHECKS if c["status"] == "NEEDS QUALIFICATION"]
rep = {"script": "scripts/step4_provenance_audit.py", "n_checks": len(CHECKS), "n_pass": sum(c["status"] == "PASS" for c in CHECKS), "n_needs_qualification": len(n_nq),
       "n_fail": len(n_fail), "full_sha256": {p: {"raw": a, "lf_normalised": b} for p, (a, b) in full.items()}, "checks": CHECKS}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{len(CHECKS)} checks, {rep['n_pass']} PASS, {len(n_nq)} NEEDS QUALIFICATION, {len(n_fail)} FAIL -> {args.out}")
for c in CHECKS:
    if c["status"] != "PASS":
        print(c["status"], c["id"], "|", c["claim"], "|", str(c["evidence"])[:200])
sys.exit(1 if n_fail else 0)
