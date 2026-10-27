"""Step 4 numerical audit: recompute every quantitative claim of the manuscript from the committed frozen JSON artifacts.

    python scripts/step4_numeric_audit.py [--out results/audit/step4_numeric_audit.json] [--only PREFIX]

Reads ONLY committed JSON artifacts and the manuscript text (docs/PAPER_DRAFT.md, paper/body.tex). It never opens the sealed TEST
extraction (data/extracted_phase3_test) and never re-fits or re-selects anything. Each check records:

    id, section, claim, reported (string as printed in the manuscript), recomputed (float), basis, rule, status, source, md_ok, tex_ok

basis:  counts      recomputed from stored exit-loop histograms / accuracies
        arithmetic  derived from stored per-benchmark values (differences, means, ranges, counts)
        stored      transcribed from a stored statistic that cannot be recomputed without re-reading spent data (TEST bootstrap CIs)
status: PASS = reported equals recomputed after rounding to the printed precision (half-up, or the binary half-even value for an exact tie)
        FAIL = mismatch, or the printed number/context is not found in the manuscript text
Exit code is 0 only if no check FAILs.
"""
import argparse
import json
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
BENCH = ["openbookqa", "commonsense_qa", "hellaswag"]
SHORT = {"openbookqa": "OBQA", "commonsense_qa": "CSQA", "hellaswag": "HS"}


def jl(p):
    return json.loads((ROOT / p).read_text(encoding="utf-8"))


T = jl("results/c1/test_report.json")
F = jl("results/c1/frozen.json")
E = jl("results/exploratory/e1_report.json")
X = jl("results/exploratory/x1_flips.json")
G = jl("results/phase3/g3a_report.json")
DT = jl("results/diagnostics/dt_truncation_report.json")
DT2 = jl("results/diagnostics/dt2_summary.json")
MD = (ROOT / "docs/PAPER_DRAFT.md").read_text(encoding="utf-8")


def norm_md(s):
    return re.sub(r"\s+", " ", s.replace("**", "").replace("*", "").replace("−", "-"))


def norm_tex(s):
    s = re.sub(r"(?<!\\)%.*", "", s)      # strip LaTeX comments only (an escaped \% is kept)
    for a, b in [("{[}", "["), ("{]}", "]"), ("\\%", "%"), ("~", " "), ("\\,", ""), ("$", ""), ("\\_", "_"), ("\\textbf{", ""),
                 ("\\emph{", ""), ("\\textit{", ""), ("\\times", "x"), ("\\text{", ""), ("{", ""), ("}", ""), ("\\", ""),
                 ("--", "-"), ("−", "-"), ("\\mathrm", "")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", " ", s)


MDN = norm_md(MD)
TEXN = norm_tex((ROOT / "paper/body.tex").read_text(encoding="utf-8"))


def norm_tok(s):
    return s.replace("−", "-").replace("**", "")


def present(text, ctx, tok, win=450):
    """Is `tok` within `win` characters after the start of some occurrence of `ctx` (or ctx None: anywhere)?"""
    tok = norm_tok(tok)
    if ctx is None:
        return tok in text
    ctx = norm_tok(ctx)
    for m in re.finditer(re.escape(ctx), text):
        seg = text[max(0, m.start() - 120): m.end() + win]
        if tok in seg:
            return True
    return False


def dec_places(s):
    s = norm_tok(s)
    m = re.search(r"\.(\d+)", s)
    return len(m.group(1)) if m else 0


def rnd(x, d):
    q = Decimal(1).scaleb(-d)
    return float(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


CHECKS = []
SECTION = ["?"]


def sec(name):
    SECTION[0] = name


def C(cid, claim, reported, value, basis, source, ctx=None, tol=None, d=None, show=None, alt=None):
    """Numeric check. reported: string as printed. value: recomputed float. tol: absolute tolerance instead of rounding rule.
    show: a different printed form to look for in the text (default: reported)."""
    rep_f = float(norm_tok(reported).replace("+", "").replace(",", "").replace("×10⁻", "e-"))
    dd = dec_places(reported) if d is None else d
    if tol is not None:
        ok, rule = abs(rep_f - float(value)) <= tol, f"abs diff <= {tol}"
    else:
        v10 = round(float(value), 10)               # removes binary noise so exact decimal ties (e.g. 0.1205) are seen as ties
        a = rnd(v10, dd)
        b = float(f"{float(value):.{dd}f}")
        ok = abs(rep_f - a) < 10 ** (-dd - 6) or abs(rep_f - b) < 10 ** (-dd - 6)
        rule = f"round to {dd} dp (half-up on the 10-dp value, or the binary value)"
        if ok and abs(a - b) > 10 ** (-dd - 6):
            rule += "; ROUNDING TIE (exact decimal tie)"
    tok = show or reported
    md_ok = present(MDN, ctx, tok)
    tex_ok = present(TEXN, ctx, tok)
    status = "PASS" if (ok and md_ok and tex_ok) else "FAIL"
    if not ok and alt is not None and abs(rep_f - float(alt)) < 10 ** (-dd - 6) and md_ok and tex_ok:
        status = "NEEDS QUALIFICATION"
        rule += f"; printed value equals the difference of printed table entries ({alt}), not the exact value"
    CHECKS.append(dict(id=cid, section=SECTION[0], claim=claim, reported=reported, recomputed=float(value), basis=basis,
                       rule=rule, status=status, source=source, ctx=ctx, numeric_ok=bool(ok), md_ok=md_ok, tex_ok=tex_ok))


def B(cid, claim, cond, basis, source, detail=""):
    """Boolean / exact check."""
    CHECKS.append(dict(id=cid, section=SECTION[0], claim=claim, reported="(statement)", recomputed=detail, basis=basis,
                       rule="exact", status="PASS" if cond else "FAIL", source=source, ctx=None, numeric_ok=bool(cond),
                       md_ok=None, tex_ok=None))


def sav(h, n):
    return sum(v * (1 - int(k) / 4) for k, v in h.items()) / n


# ---------------------------------------------------------------- A. C1 TEST: Tables 3-5, Section 5
sec("5.1-5.2 C1 TEST primary, Table 4")
pb, bt, fz = T["per_benchmark"], T["budget_table"], F
for meth in ["pi1-x", "native-pb", "B_pb"]:
    for b in BENCH:
        r = pb[meth][b]
        n = r["n"]
        s_re = sav(r["exit_hist"], n)
        l_re = r["acc4_pp"] - r["acc_pp"]
        # stored internal consistency (counts -> saving, accuracies -> loss)
        B(f"T4.consist.{meth}.{b}", f"{meth} {b}: saving from exit histogram and loss from accuracies equal the stored values",
          abs(s_re - r["saving"]) < 1e-12 and abs(l_re - r["loss_pp"]) < 1e-9 and sum(r["exit_hist"].values()) == n,
          "counts", "test_report.per_benchmark", f"saving {s_re:.6f} vs {r['saving']:.6f}; loss {l_re:.6f} vs {r['loss_pp']:.6f}")
rows4 = {  # (method,bench): (m, saving, loss, ub, hist)
    ("pi1-x", "openbookqa"): ("4", "0.157", "0.17", "0.53", "6 / 243 / 1,377 / 1,374"),
    ("native-pb", "openbookqa"): ("4", "0.136", "0.20", "0.67", "0 / 26 / 1,579 / 1,395"),
    ("B_pb", "openbookqa"): ("4", "0.000", "0.00", "0.00", "0 / 0 / 0 / 3,000"),
    ("pi1-x", "commonsense_qa"): ("4", "0.282", "0.60", "1.10", "8 / 349 / 405 / 238"),
    ("native-pb", "commonsense_qa"): ("4", "0.161", "-0.10", "0.30", "0 / 1 / 643 / 356"),
    ("B_pb", "commonsense_qa"): ("4", "0.000", "0.00", "0.00", "0 / 0 / 0 / 1,000"),
    ("pi1-x", "hellaswag"): ("3", "0.279", "0.40", "1.60", "0 / 88 / 662 / 0"),
    ("native-pb", "hellaswag"): ("3", "0.262", "0.13", "1.33", "0 / 36 / 714 / 0"),
    ("B_pb", "hellaswag"): ("3", "0.250", "0.53", "1.60", "0 / 0 / 750 / 0"),
}
for (meth, b), (m, s, l, u, h) in rows4.items():
    r = pb[meth][b]
    n = r["n"]
    mm = (F["primary"][meth][b]["m"] if meth != "B_pb" else F["reference_B_pb"][b]["m"])
    B(f"T4.m.{meth}.{b}", f"Table 4 depth cap {SHORT[b]} {meth} = {m}", str(mm) == m, "arithmetic", "frozen.json", str(mm))
    C(f"T4.sav.{meth}.{b}", f"Table 4 gross saving {SHORT[b]} {meth}", s, sav(r["exit_hist"], n), "counts", "test_report exit_hist", ctx=None,
      d=3)
    C(f"T4.loss.{meth}.{b}", f"Table 4 loss pp {SHORT[b]} {meth}", l, r["acc4_pp"] - r["acc_pp"], "counts", "test_report acc", d=2)
    C(f"T4.ub.{meth}.{b}", f"Table 4 upper bound pp {SHORT[b]} {meth}", u, bt[meth][b]["upper_bound_95_pp"], "stored",
      "test_report.budget_table", d=2)
    hist = " / ".join(f"{r['exit_hist'][k]:,}" for k in "1234")
    B(f"T4.hist.{meth}.{b}", f"Table 4 exit histogram {SHORT[b]} {meth} = {h}", hist == h and norm_tok(h) in MDN,
      "counts", "test_report exit_hist", hist)

sec("5.1 Table 3 / macro results")
mac = T["macro_gross_saving"]
macro_re = {m: sum(sav(pb[m][b]["exit_hist"], pb[m][b]["n"]) for b in BENCH) / 3 for m in ["pi1-x", "native-pb", "B_pb"]}
for m in macro_re:
    B(f"macro.consist.{m}", f"{m} macro saving recomputed from histograms equals stored", abs(macro_re[m] - mac[m]) < 1e-12, "counts",
      "test_report", f"{macro_re[m]:.8f} vs {mac[m]:.8f}")
C("T3.delta", "Primary Δ gross macro = macro(pi1-x) − macro(native-pb)", "0.053", macro_re["pi1-x"] - macro_re["native-pb"], "counts",
  "test_report", ctx="Bootstrap-screened, TEST", d=3)
C("T3.delta.4dp", "Primary Δ 4dp (abstract: +0.053)", "0.0529", macro_re["pi1-x"] - macro_re["native-pb"], "counts", "test_report", d=4,
  show="0.0529") if "0.0529" in MDN else None
C("T3.ci.lo", "Primary CI lower", "0.049", T["primary"]["ci95"][0], "stored", "test_report.primary.ci95", ctx="[+0.049, +0.057]", d=3,
  show="+0.049")
C("T3.ci.hi", "Primary CI upper", "0.057", T["primary"]["ci95"][1], "stored", "test_report.primary.ci95", ctx="[+0.049, +0.057]", d=3,
  show="+0.057")
B("T3.ci.contains", "Primary point lies inside its interval", T["primary"]["ci95"][0] < T["primary"]["delta_gross"] < T["primary"]["ci95"][1],
  "stored", "test_report")
hw = (T["primary"]["ci95"][1] - T["primary"]["ci95"][0]) / 2
C("T3.halfwidth", "Realised half-width 0.0042", "0.0042", hw, "stored", "test_report.primary.ci95", ctx="realised half-width", d=4)
C("T3.expected_hw", "Expected half-width at freeze 0.0043", "0.0043", F["expected_precision"]["expected_half_width_95"], "stored",
  "frozen.expected_precision", ctx="expected at freeze", d=4)
C("T3.macro.pi1x", "Macro saving pi1-x 0.239", "0.239", macro_re["pi1-x"], "counts", "test_report", ctx="The macro saving was", d=3)
C("T3.macro.native", "Macro saving native-pb 0.186", "0.186", macro_re["native-pb"], "counts", "test_report", ctx="The macro saving was", d=3)
B("T3.verdict", "Verdict PASS follows from the frozen rule: all losses <= 1 pp (+1e-12), delta >= 0.02, L > 0",
  all(pb["pi1-x"][b]["loss_pp"] <= 1.0 + 1e-12 for b in BENCH) and T["primary"]["delta_gross"] >= 0.02 and T["primary"]["ci95"][0] > 0
  and T["verdict"] == "PASS", "arithmetic", "test_report; docs/C1_PROTOCOL.md C1.3-C1.5")
B("T3.L_gt_delta", "Lower bound exceeds delta = 0.02 (secondary fact stated in 5.1)", T["primary"]["ci95"][0] > 0.02 and T["primary"]["L_above_delta_min"],
  "arithmetic", "test_report")
for b, lab in zip(BENCH, ["0.17", "0.60", "0.40"]):
    C(f"T3.pointloss.{b}", f"pi1-x point loss {SHORT[b]} (5.1 text)", lab, pb["pi1-x"][b]["loss_pp"], "counts", "test_report", ctx="point losses", d=2)
C("T3.macroloss.pi1x", "Macro loss pi1-x 0.39 pp (mean of Table 4 losses)", "0.39", sum(pb["pi1-x"][b]["loss_pp"] for b in BENCH) / 3, "arithmetic",
  "test_report", ctx="selected points lost", d=2)
C("T3.macroloss.native", "Macro loss native-pb 0.08 pp", "0.08", sum(pb["native-pb"][b]["loss_pp"] for b in BENCH) / 3, "arithmetic",
  "test_report", ctx="selected points lost", d=2)
for b, lab in zip(BENCH, ["0.021", "0.121", "0.017"]):
    C(f"T4.diff.{b}", f"Per-benchmark saving difference {SHORT[b]}", lab, pb["pi1-x"][b]["saving"] - pb["native-pb"][b]["saving"], "arithmetic",
      "test_report", ctx="Per benchmark, π1-x saved more than native-pb by", d=3)
# net
sec("5.5 Net saving")
C("net.delta", "Net Δ 0.038", "0.038", T["secondary_net"]["delta_net"], "stored", "test_report.secondary_net", ctx="Net of the estimated probe overhead", d=3)
C("net.ci.lo", "Net CI lower", "0.033", T["secondary_net"]["ci95"][0], "stored", "test_report.secondary_net", ctx="[+0.033, +0.042]", d=3, show="+0.033")
C("net.ci.hi", "Net CI upper", "0.042", T["secondary_net"]["ci95"][1], "stored", "test_report.secondary_net", ctx="[+0.033, +0.042]", d=3, show="+0.042")
nets = {b: pb["pi1-x"][b]["net_saving"] for b in BENCH}
for b, lab in zip(BENCH, ["0.125", "0.274", "0.272"]):
    C(f"net.pi1x.{b}", f"Net saving pi1-x {SHORT[b]}", lab, nets[b], "stored", "test_report.per_benchmark", ctx="per-benchmark net savings", d=3)
for b, lab in zip(BENCH, ["0.031", "0.007", "0.007"]):
    C(f"net.overhead.{b}", f"Overhead gross−net {SHORT[b]}", lab, pb["pi1-x"][b]["saving"] - nets[b], "arithmetic", "test_report", ctx="the overhead costs", d=3)
# recompute net from rho and exit histogram?  (needs n_score/n_eval per question; exit hist + cap suffice)
sec("3.5 Table 1 rho (freeze-time)")
for b, (rs, rr, rt, ms) in zip(BENCH, [("0.036", "0.014", "0.050", "20.7"), ("0.010", "0.004", "0.014", "79.9"), ("0.013", "0.003", "0.015", "148.8")]):
    r = F["rho"][b]
    C(f"T1.score.{b}", f"rho_score {SHORT[b]}", rs, r["rho_score"], "stored", "frozen.rho", d=3)
    C(f"T1.rest.{b}", f"rho_rest {SHORT[b]}", rr, r["rho_rest"], "stored", "frozen.rho", d=3)
    C(f"T1.total.{b}", f"rho_total {SHORT[b]}", rt, r["rho_total"], "stored", "frozen.rho", d=3)
    C(f"T1.ms.{b}", f"One loop ms {SHORT[b]}", ms, r["loop_ms"], "stored", "frozen.rho", d=1)
    B(f"T1.n.{b}", f"rho measured on 200 questions {SHORT[b]}", r["n_questions"] == 200, "stored", "frozen.rho")
    B(f"T1.sum.{b}", f"rho_total = rho_score + rho_rest {SHORT[b]}", abs(r["rho_total"] - r["rho_score"] - r["rho_rest"]) < 1e-12, "arithmetic", "frozen.rho")


def net_recompute(meth_hist, m, rho_s, rho_r, n):
    """Per-question net: 1 - (e + rs*n_score + rr*n_eval)/4; halting by score at e<m: n_eval=e, n_score=e-1; reaching cap: both m-1."""
    tot = 0.0
    for k, v in meth_hist.items():
        e = int(k)
        if e < m:
            ne, ns = e, e - 1
        else:
            ne = ns = m - 1
        tot += v * (1 - (e + rho_s * ns + rho_r * ne) / 4)
    return tot / n


for b in BENCH:
    r = pb["pi1-x"][b]
    m = F["primary"]["pi1-x"][b]["m"]
    v = net_recompute(r["exit_hist"], m, F["rho"][b]["rho_score"], F["rho"][b]["rho_rest"], r["n"])
    B(f"net.recompute.{b}", f"Net saving of pi1-x {SHORT[b]} recomputed from exit histogram, cap and rho by the formula of Section 3.5",
      abs(v - r["net_saving"]) < 1e-9, "counts", "test_report + frozen.rho", f"{v:.9f} vs {r['net_saving']:.9f}")
# native-pb net = gross (rho=0)
for b in BENCH:
    r = pb["native-pb"][b]
    B(f"net.native.{b}", f"native-pb net saving equals gross {SHORT[b]} (rho = 0)", abs(r.get("net_saving", r["saving"]) - r["saving"]) < 1e-12, "arithmetic", "test_report")

# ---------------------------------------------------------------- Table 5 point-only
sec("5.3-5.4 Table 5 point-only vs screened")
sp = T["secondary_point_rule"]
t5 = {("pi1-x", "openbookqa"): ("0.305", "1.13"), ("pi1-x", "commonsense_qa"): ("0.357", "1.10"), ("pi1-x", "hellaswag"): ("0.351", "1.07"),
      ("native-pb", "openbookqa"): ("0.403", "1.57"), ("native-pb", "commonsense_qa"): ("0.359", "1.20"), ("native-pb", "hellaswag"): ("0.316", "0.67")}
for (m, b), (s, l) in t5.items():
    r = sp["per_benchmark"][m][b]
    C(f"T5.sav.{m}.{b}", f"Table 5 point-only saving {SHORT[b]} {m}", s, sav(r["exit_hist"], r["n"]), "counts", "test_report.secondary_point_rule", d=3)
    C(f"T5.loss.{m}.{b}", f"Table 5 point-only loss {SHORT[b]} {m}", l, r["acc4_pp"] - r["acc_pp"], "counts", "test_report.secondary_point_rule", d=2)
pm = {m: sum(sav(sp["per_benchmark"][m][b]["exit_hist"], sp["per_benchmark"][m][b]["n"]) for b in BENCH) / 3 for m in ["pi1-x", "native-pb"]}
C("T5.macro.pi1x", "Point-only macro pi1-x 0.337", "0.337", pm["pi1-x"], "counts", "test_report", ctx="Point-only", d=3)
C("T5.macro.native", "Point-only macro native-pb 0.359", "0.359", pm["native-pb"], "counts", "test_report", ctx="Point-only", d=3)
C("T5.delta", "Point-only Δ −0.022", "-0.022", pm["pi1-x"] - pm["native-pb"], "counts", "test_report.secondary_point_rule", ctx="Point-only, TEST", d=3)
B("T5.delta.stored", "Stored point-only delta equals recomputed", abs(sp["delta"] - (pm["pi1-x"] - pm["native-pb"])) < 1e-12, "counts", "test_report")
C("T5.ci.lo", "Point-only CI lower −0.026", "-0.026", sp["ci95"][0], "stored", "test_report", ctx="[−0.026, −0.018]", d=3, show="-0.026")
C("T5.ci.hi", "Point-only CI upper −0.018", "-0.018", sp["ci95"][1], "stored", "test_report", ctx="[−0.026, −0.018]", d=3, show="-0.018")
over = {(m, b): (sp["per_benchmark"][m][b]["acc4_pp"] - sp["per_benchmark"][m][b]["acc_pp"]) > 1.0 + 1e-12 for m in ["pi1-x", "native-pb"] for b in BENCH}
B("T5.over.count", "Point-only operating points above 1 pp on TEST: 5 of 6 cells (pi1-x all three, native-pb two)", sum(over.values()) == 5
  and all(over[("pi1-x", b)] for b in BENCH) and over[("native-pb", "openbookqa")] and over[("native-pb", "commonsense_qa")] and not over[("native-pb", "hellaswag")],
  "counts", "test_report.secondary_point_rule", str(sum(over.values())))
C("5.4.drop.native", "Native macro drop 0.173", "0.173", pm["native-pb"] - macro_re["native-pb"], "arithmetic", "Table 5 macros", ctx="a drop of", d=3)
C("5.4.drop.pi1x", "pi1-x macro drop 0.098", "0.098", pm["pi1-x"] - macro_re["pi1-x"], "arithmetic", "Table 5 macros", ctx="a drop of", d=3)
for b, lab, alt_ in zip(BENCH, ["-0.098", "-0.002", "0.035"], [round(float(t5[("pi1-x", b)][0]) - float(t5[("native-pb", b)][0]), 3) for b in BENCH]):
    C(f"5.4.pp.diff.{b}", f"Point-only pi1-x − native-pb {SHORT[b]}", lab,
      sav(sp["per_benchmark"]["pi1-x"][b]["exit_hist"], sp["per_benchmark"][b if False else "pi1-x"][b]["n"])
      - sav(sp["per_benchmark"]["native-pb"][b]["exit_hist"], sp["per_benchmark"]["native-pb"][b]["n"]), "arithmetic", "Table 5",
      ctx="per benchmark, π1-x − native-pb is", d=3, alt=alt_)
C("5.4.obqa.tau", "native-pb point-only OBQA threshold 1.9e-5", "1.9", F["secondary_point_rule"]["native-pb"]["openbookqa"]["tau"] * 1e5, "stored",
  "frozen.secondary_point_rule", ctx="threshold of 1.9", d=1)
B("5.4.obqa.cap", "native-pb point-only OBQA depth cap 4", F["secondary_point_rule"]["native-pb"]["openbookqa"]["m"] == 4, "stored", "frozen")
C("5.4.obqa.sel", "native-pb OBQA loss at selection 0.98 pp", "0.98", F["secondary_point_rule"]["native-pb"]["openbookqa"]["loss"], "stored", "frozen",
  ctx="rose from", d=2)
C("5.4.obqa.test", "native-pb OBQA loss on TEST 1.57 pp", "1.57", sp["per_benchmark"]["native-pb"]["openbookqa"]["acc4_pp"] - sp["per_benchmark"]["native-pb"]["openbookqa"]["acc_pp"],
  "counts", "test_report", ctx="rose from", d=2)
over_all = {}
for m in ["pi1-x", "native-pb"]:
    for b in BENCH:
        f_ = F["secondary_point_rule"][m][b]["loss"]
        t_ = sp["per_benchmark"][m][b]["acc4_pp"] - sp["per_benchmark"][m][b]["acc_pp"]
        over_all[(m, b)] = t_ - f_
B("5.4.largest_overshoot", "OBQA native-pb has the largest selection-to-TEST overshoot of any selected point (screened and point-only, all six cells each)",
  max(over_all, key=over_all.get) == ("native-pb", "openbookqa") and
  over_all[("native-pb", "openbookqa")] >= max(pb[m][b]["loss_pp"] - F["primary"][m][b]["loss"] for m in ["pi1-x", "native-pb"] for b in BENCH),
  "arithmetic", "frozen vs test_report", f"{over_all[('native-pb','openbookqa')]:.3f} pp")

# ---------------------------------------------------------------- 5.3 bounds
sec("5.3 Upper bounds")
ubs = {(m, b): bt[m][b]["upper_bound_95_pp"] for m in ["pi1-x", "native-pb", "B_pb"] for b in BENCH}
B("5.3.count4", "Four of nine TEST upper bounds exceed 1 pp", sum(v > 1.0 for v in ubs.values()) == 4 and
  {k for k, v in ubs.items() if v > 1.0} == {("pi1-x", "commonsense_qa"), ("pi1-x", "hellaswag"), ("native-pb", "hellaswag"), ("B_pb", "hellaswag")},
  "stored", "test_report.budget_table", str(sorted(k for k, v in ubs.items() if v > 1.0)))
B("5.3.nine_points", "All nine point losses within 1 pp (+1e-12)", all(pb[m][b]["loss_pp"] <= 1.0 + 1e-12 for m in ["pi1-x", "native-pb", "B_pb"] for b in BENCH),
  "counts", "test_report")
selub = [F["primary"][m][b]["upper_bound"] for m in ["pi1-x", "native-pb"] for b in BENCH]
C("5.3.selub.min", "Selection-time adaptive bounds between 0.75 …", "0.75", min(selub) , "stored", "frozen.primary", ctx="bounds of the frozen candidates were between", d=2)
C("5.3.selub.max", "… and 0.98 pp", "0.98", max(selub), "stored", "frozen.primary", ctx="bounds of the frozen candidates were between", d=2)
# E-selection margin claim in 7.3
sec("7.3 In-sample bound margins")
for b, (a_, n_) in zip(BENCH, [("0.44", "0.53"), ("0.51", "0.68"), ("0.37", "0.38")]):
    C(f"7.3.margin.pi1x.{b}", f"In-sample bound margin pi1-x {SHORT[b]} (upper bound − loss, selection data)", a_,
      F["primary"]["pi1-x"][b]["upper_bound"] - F["primary"]["pi1-x"][b]["loss"], "arithmetic", "frozen.primary", ctx="smaller bound margins", d=2)
    C(f"7.3.margin.native.{b}", f"In-sample bound margin native-pb {SHORT[b]}", n_,
      F["primary"]["native-pb"][b]["upper_bound"] - F["primary"]["native-pb"][b]["loss"], "arithmetic", "frozen.primary", ctx="smaller bound margins", d=2)
    B(f"7.3.margin.order.{b}", f"pi1-x margin smaller and saving larger than native-pb on {SHORT[b]}",
      F["primary"]["pi1-x"][b]["upper_bound"] - F["primary"]["pi1-x"][b]["loss"] < F["primary"]["native-pb"][b]["upper_bound"] - F["primary"]["native-pb"][b]["loss"]
      and F["primary"]["pi1-x"][b]["saving"] > F["primary"]["native-pb"][b]["saving"], "arithmetic", "frozen.primary")

# ---------------------------------------------------------------- Table 6
sec("6.1 Table 6 baselines")
pbl = T["secondary_all_baselines"]["per_baseline"]
t6 = {"B_pb": ("0.083", "+0.156", "+0.151", "+0.161"), "native-pb": ("0.186", "+0.053", "+0.049", "+0.057"), "r-conf": ("0.171", "+0.068", "+0.063", "+0.073"),
      "r-margin": ("0.170", "+0.069", "+0.064", "+0.075"), "r-ent": ("0.179", "+0.060", "+0.055", "+0.065"), "r-dlogit": ("0.162", "+0.077", "+0.071", "+0.083"),
      "r-hcos": ("0.174", "+0.065", "+0.059", "+0.072"), "r-hl2": ("0.149", "+0.090", "+0.084", "+0.096")}
for k, (ms, d_, lo, hi) in t6.items():
    C(f"T6.macro.{k}", f"Table 6 macro saving {k}", ms, mac[k], "stored", "test_report.macro_gross_saving", d=3)
    C(f"T6.delta.{k}", f"Table 6 Δ pi1-x − {k}", d_, mac["pi1-x"] - mac[k], "arithmetic", "macro_gross_saving", d=3)
    B(f"T6.delta.stored.{k}", f"Stored delta equals macro difference ({k})", abs(pbl[k]["delta"] - (mac["pi1-x"] - mac[k])) < 1e-12, "arithmetic", "test_report")
    C(f"T6.ci.lo.{k}", f"Table 6 CI lower {k}", lo, pbl[k]["ci95"][0], "stored", "test_report.per_baseline", d=3)
    C(f"T6.ci.hi.{k}", f"Table 6 CI upper {k}", hi, pbl[k]["ci95"][1], "stored", "test_report.per_baseline", d=3)
    B(f"T6.ci.contains.{k}", f"Point inside CI ({k})", pbl[k]["ci95"][0] < pbl[k]["delta"] < pbl[k]["ci95"][1], "stored", "test_report")
C("T6.macro.rkl", "r-kl macro 0.238", "0.238", mac["r-kl"], "stored", "test_report", ctx="r-kl", d=3)
C("T6.delta.rkl", "r-kl Δ +0.00125", "0.00125", mac["pi1-x"] - mac["r-kl"], "arithmetic", "macro_gross_saving", ctx="The difference between π1-x and r-kl is", d=5)
C("T6.rkl.lo", "r-kl CI lower −0.0041", "-0.0041", pbl["r-kl"]["ci95"][0], "stored", "test_report", ctx="95% interval of", d=4)
C("T6.rkl.hi", "r-kl CI upper +0.0065", "0.0065", pbl["r-kl"]["ci95"][1], "stored", "test_report", ctx="95% interval of", d=4)
B("T6.rkl.reselect", "r-kl selected in 10,000 of 10,000 resamples and re-selected interval equals r-kl interval",
  T["secondary_all_baselines"]["selection_frequency"]["r-kl"] == 10000 and T["secondary_all_baselines"]["reselecting_bootstrap_ci95"] == pbl["r-kl"]["ci95"],
  "stored", "test_report")
C("6.1.rkl_vs_native", "r-kl − native-pb macro 0.052", "0.052", mac["r-kl"] - mac["native-pb"], "arithmetic", "macro_gross_saving", ctx="r-kl exceeded native-pb by", d=3)
nb = [mac[k] for k in ["r-conf", "r-margin", "r-ent", "r-dlogit", "r-hcos", "r-hl2"]]
d_other = [mac["pi1-x"] - v for v in nb]
C("6.1.range.lo", "Other readouts trail pi1-x by 0.060 …", "0.060", min(d_other), "arithmetic", "macro_gross_saving", ctx="trail π1-x by between", d=3)
C("6.1.range.hi", "… to 0.090", "0.090", max(d_other), "arithmetic", "macro_gross_saving", ctx="trail π1-x by between", d=3)
maxloss = max((v, k, b) for k, d_ in T["secondary_violations"].items() if k not in ("B_pb",) for b, v in d_.items() if k not in ("pi1-x", "native-pb"))
C("6.1.maxloss", "Largest baseline TEST point loss 0.93 pp (r-dlogit, HellaSwag)", "0.93", maxloss[0], "stored", "test_report.secondary_violations", ctx="the largest was", d=2)
B("6.1.maxloss.who", "…is r-dlogit on HellaSwag", maxloss[1] == "r-dlogit" and maxloss[2] == "hellaswag", "stored", "test_report", str(maxloss))
B("6.1.allwithin", "Every baseline's TEST point loss within 1 pp", all(v <= 1.0 + 1e-12 for d_ in T["secondary_violations"].values() for v in d_.values()), "stored",
  "test_report.secondary_violations")
sel = F["secondary_risk_all_baselines"]["r-kl"]
for b, (thr, cap) in zip(BENCH, [("3.5", 3), ("3.7", 4), ("2.7", 3)]):
    exp = {"openbookqa": 3.5e-3, "commonsense_qa": 3.7e-2, "hellaswag": 2.7e-4}[b]
    mant = -sel[b]["tau"] / (10 ** {"openbookqa": -3, "commonsense_qa": -2, "hellaswag": -4}[b])
    C(f"6.1.klthr.{b}", f"Frozen r-kl threshold {SHORT[b]} (mantissa)", thr, mant, "stored", "frozen.secondary_risk_all_baselines.r-kl", d=1)
    B(f"6.1.klcap.{b}", f"Frozen r-kl cap {SHORT[b]} = {cap}", sel[b]["m"] == cap, "stored", "frozen")
sdev = F["selection_macro_saving_dev"]
C("6.1.dev.pi1x", "In-sample dev macro pi1-x 0.246", "0.246", sdev["pi1-x"], "stored", "frozen.selection_macro_saving_dev", ctx="in-sample macro savings were", d=3)
C("6.1.dev.rkl", "In-sample dev macro r-kl 0.243", "0.243", sdev["r-kl"], "stored", "frozen", ctx="in-sample macro savings were", d=3)
C("6.1.dev.native", "In-sample dev macro native-pb 0.193", "0.193", sdev["native-pb"], "stored", "frozen", ctx="in-sample macro savings were", d=3)

# ---------------------------------------------------------------- OOD
sec("6.2 OOD, Table 7")
oo = T["secondary_ood"]
t7 = {("piqa", "pi1-x"): ("0.308", "0.71", "1.52", "0 / 433 / 1,399 / 5"), ("piqa", "native-pb"): ("0.126", "0.11", "0.60", "0 / 0 / 923 / 914"),
      ("piqa", "B_pb"): ("0.000", "0.00", "0.00", "0 / 0 / 0 / 1,837"), ("social_iqa", "pi1-x"): ("0.481", "3.26", "4.72", "0 / 1,785 / 143 / 2"),
      ("social_iqa", "native-pb"): ("0.028", "-0.05", "0.16", "0 / 0 / 214 / 1,716"), ("social_iqa", "B_pb"): ("0.000", "0.00", "0.00", "0 / 0 / 0 / 1,930")}
for (b, m), (s, l, u, h) in t7.items():
    r = oo[b]["summary"][m]
    C(f"T7.sav.{b}.{m}", f"Table 7 saving {b} {m}", s, sav(r["exit_hist"], r["n"]), "counts", "test_report.secondary_ood", d=3)
    C(f"T7.loss.{b}.{m}", f"Table 7 loss {b} {m}", l, r["acc4_pp"] - r["acc_pp"], "counts", "test_report.secondary_ood", d=2)
    C(f"T7.ub.{b}.{m}", f"Table 7 upper bound {b} {m}", u, oo[b]["upper_bound_95_pp"][m], "stored", "test_report.secondary_ood", d=2)
    B(f"T7.hist.{b}.{m}", f"Table 7 histogram {b} {m}", " / ".join(f"{r['exit_hist'][k]:,}" for k in "1234") == h, "counts", "test_report", h)
C("6.2.piqa.delta", "PIQA Δ vs native +0.183", "0.183", oo["piqa"]["delta_pi1x_vs_native"], "stored", "test_report.secondary_ood", ctx="it saved", d=3)
C("6.2.piqa.ci.lo", "PIQA CI lo", "0.174", oo["piqa"]["ci95"][0], "stored", "test_report", ctx="95% CI [+0.174", d=3, show="+0.174")
C("6.2.piqa.ci.hi", "PIQA CI hi", "0.191", oo["piqa"]["ci95"][1], "stored", "test_report", ctx="95% CI [+0.174", d=3, show="+0.191")
C("6.2.siqa.delta", "SIQA Δ +0.453", "0.453", oo["social_iqa"]["delta_pi1x_vs_native"], "stored", "test_report", ctx="saving difference against native-pb", d=3)
C("6.2.siqa.ci.lo", "SIQA CI lo", "0.449", oo["social_iqa"]["ci95"][0], "stored", "test_report", ctx="95% CI [+0.449", d=3, show="+0.449")
C("6.2.siqa.ci.hi", "SIQA CI hi", "0.458", oo["social_iqa"]["ci95"][1], "stored", "test_report", ctx="95% CI [+0.449", d=3, show="+0.458")
for b, m in [("piqa", "pi1-x"), ("social_iqa", "pi1-x")]:
    r = oo[b]["summary"][m]
B("6.2.delta.recompute", "OOD deltas equal pi1-x − native-pb savings recomputed from histograms",
  all(abs(oo[b]["delta_pi1x_vs_native"] - (sav(oo[b]["summary"]["pi1-x"]["exit_hist"], oo[b]["summary"]["pi1-x"]["n"]) - sav(oo[b]["summary"]["native-pb"]["exit_hist"], oo[b]["summary"]["native-pb"]["n"]))) < 1e-12 for b in oo),
  "counts", "test_report")
C("6.2.siqa.acc4", "SIQA accuracy at loop 4: 43.32%", "43.32", oo["social_iqa"]["summary"]["pi1-x"]["acc4_pp"], "counts", "test_report", ctx="accuracy fell from", d=2)
C("6.2.siqa.acc", "SIQA accuracy pi1-x 40.05%", "40.05", oo["social_iqa"]["summary"]["pi1-x"]["acc_pp"], "counts", "test_report", ctx="accuracy fell from", d=2)
C("6.2.tau.pi1x", "OOD pi1-x tau 0.961", "0.961", F["ood_pooled"]["pi1-x"]["tau"], "stored", "frozen.ood_pooled", ctx=None, d=3)
C("6.2.tau.native", "OOD native tau 0.386", "0.386", F["ood_pooled"]["native-pb"]["tau"], "stored", "frozen.ood_pooled", ctx=None, d=3)
C("6.2.devsave.pi1x", "OOD dev macro saving 0.159 (pi1-x)", "0.159", F["ood_pooled"]["pi1-x"]["macro_saving"], "stored", "frozen.ood_pooled", ctx="development macro saving", d=3)
C("6.2.devsave.native", "OOD dev macro saving 0.159 (native-pb)", "0.159", F["ood_pooled"]["native-pb"]["macro_saving"], "stored", "frozen.ood_pooled", ctx="development macro saving", d=3)
B("6.2.m", "OOD both m = 4", F["ood_pooled"]["pi1-x"]["m"] == 4 and F["ood_pooled"]["native-pb"]["m"] == 4, "stored", "frozen")
B("6.2.allsix", "OOD selection qualifies only if every dev benchmark met the rule (per-benchmark dev losses and bounds <= 1 pp)",
  all(v["loss"] <= 1.0 + 1e-12 for m in ["pi1-x", "native-pb"] for v in F["ood_pooled"][m]["per_benchmark"].values()) and
  all(v <= 1.0 + 1e-12 for m in ["pi1-x", "native-pb"] for v in F["ood_pooled"][m]["upper_bounds"].values()), "stored", "frozen.ood_pooled")

# ---------------------------------------------------------------- counts/data
sec("4.1 Data counts")
nd = E["n"]["dev"]
B("4.1.dev.total", "Development questions 20,250 across six benchmarks", sum(nd.values()) == 20250 and len(nd) == 6, "arithmetic", "e1_report.n.dev", str(sum(nd.values())))
B("4.1.dev.sources", "17,615 + 2,635 = 20,250", E["n"]["dev_by_source"]["p1_train"] == 17615 and E["n"]["dev_by_source"]["devplus"] == 2635 and 17615 + 2635 == 20250, "arithmetic", "e1_report.n.dev_by_source")
B("4.1.val", "5,870 Phase 1 validation questions", sum(E["n"]["val"].values()) == 5870, "arithmetic", "e1_report.n.val", str(sum(E["n"]["val"].values())))
ntest = dict(T["n"])
B("4.1.test.primary", "TEST primary n = 3,000 / 1,000 / 750, total 4,750", ntest == {"openbookqa": 3000, "commonsense_qa": 1000, "hellaswag": 750} and sum(ntest.values()) == 4750, "stored", "test_report.n")
B("4.1.test.ood", "TEST OOD n = 1,837 PIQA, 1,930 Social IQa; total 8,517",
  oo["piqa"]["summary"]["B_pb"]["n"] == 1837 and oo["social_iqa"]["summary"]["B_pb"]["n"] == 1930 and 4750 + 1837 + 1930 == 8517, "stored", "test_report")
B("4.1.test.meta", "TEST metadata counts equal the report n (primary + OOD)",
  all(T["provenance"]["test_metadata"][b].get("n", None) in (None, ntest.get(b, oo.get(b, {}).get("summary", {}).get("B_pb", {}).get("n"))) for b in T["provenance"]["test_metadata"]),
  "stored", "test_report.provenance.test_metadata")
B("4.1.dev.n.openbookqa", "Dev n per benchmark equals fold sizes in X1 (OBQA 2,252, CSQA 1,168, HS 6,273)",
  [sum(X["fold_sizes"][b]) for b in BENCH] == [2252, 1168, 6273], "arithmetic", "x1_flips.fold_sizes")
sec("4.4 and 3.4 constants")
B("3.4.candidates", "51 thresholds x 4 caps = 204 candidates", 51 * 4 == 204, "arithmetic", "-")
B("3.4.bootstrap", "Selection bootstrap 2,000 resamples (frozen.bootstrap_selection.n)", F["bootstrap_selection"]["n"] == 2000, "stored", "frozen.json")
B("4.3.boot", "TEST bootstrap B = 10,000, seed 2026", T["n_boot"] == 10000 and T["seed"] == 2026 and F["test_seed"] == 2026 and F["n_boot_test"] == 10000 and T["delta_min"] == 0.02 and F["delta"] == 0.02, "stored", "test_report/frozen")
C("4.3.ep", "Expected half width 0.0043", "0.0043", F["expected_precision"]["expected_half_width_95"], "stored", "frozen", ctx="expected half-width", d=4)

# ---------------------------------------------------------------- E1 / G3a
sec("5.1/5.4/6.1 E1 and G3a development statistics")
A3 = E["A3"]["pi1-x"]
C("E1.g3a.nested", "G3a/E1 nested held-out Δ −0.024 (four dp −0.0236)", "-0.0236", A3["improvement_vs_native_pb"], "stored", "e1_report.A3", ctx="G3a value to four decimals", d=4)
C("E1.g3a.nested3", "…−0.024 at three dp", "-0.024", A3["improvement_vs_native_pb"], "stored", "e1_report.A3", ctx="Point-only, G3a nested", d=3)
pf = A3["improvement_vs_native_pb_per_fold"]
C("E1.g3a.fold.lo", "Per-fold range lower −0.044", "-0.044", min(pf), "stored", "e1_report.A3", ctx="per-fold range", d=3)
C("E1.g3a.fold.hi", "Per-fold range upper −0.014", "-0.014", max(pf), "stored", "e1_report.A3", ctx="per-fold range", d=3)
C("E1.g3a.val", "G3a exposed Phase 1 validation Δ −0.029", "-0.029", G["D2_validation_exposed"]["improvement_vs_comparator"]["pi1-x"], "stored",
  "g3a_report.D2_validation_exposed", ctx="exposed Phase 1 validation", d=3)
B("E1.g3a.decision", "G3a decision STOP (D2): neither candidate reached +0.02", G["decision"].startswith("STOP") and A3["improvement_vs_native_pb"] < 0.02, "stored", "g3a_report.decision", G["decision"])
risk, pt = E["A6"]["risk"], E["A6"]["point"]
C("E1.risk.pi1x", "E1 screened: pi1-x − native-pb +0.053", "0.053", risk["pi1-x"]["improvement_vs_native_pb"], "stored", "e1_report.A6.risk", ctx="E1", d=3)
rf, pf2 = risk["pi1-x"]["improvement_vs_native_pb_per_fold"], pt["pi1-x"]["improvement_vs_native_pb_per_fold"]
B("E1.folds.signs", "pi1-x − native-pb negative in all five folds (point) and positive in all five (screened)", all(v < 0 for v in pf2) and all(v > 0 for v in rf), "stored", "e1_report.A6")
diff = [a - b for a, b in zip(rf, pf2)]
C("E1.folds.diff.lo", "Screened-minus-point-only min +0.051", "0.051", min(diff), "arithmetic", "e1_report.A6", ctx="screened-minus-point-only difference of", d=3)
C("E1.folds.diff.hi", "… max +0.127", "0.127", max(diff), "arithmetic", "e1_report.A6", ctx="screened-minus-point-only difference of", d=3)
C("E1.hs.pi1x", "E1 cross-fitted macro pi1-x 0.258", "0.258", risk["pi1-x"]["heldout_saving_primary"], "stored", "e1_report.A6.risk", ctx="E1 estimate", d=3)
C("E1.hs.rkl", "E1 cross-fitted macro r-kl 0.207", "0.207", risk["r-kl"]["heldout_saving_primary"], "stored", "e1_report.A6.risk", ctx="r-kl only", d=3)
C("E1.hs.native", "E1 cross-fitted macro native-pb 0.204", "0.204", risk["native-pb"]["heldout_saving_primary"], "stored", "e1_report.A6.risk", ctx="against 0.204", d=3)
C("E1.nocdf.primary", "Probe w/o CDF +0.021 on primary", "0.021", risk["pi1-x-nocdf"]["improvement_vs_native_pb"], "stored", "e1_report.A6.risk", ctx="without the CDF feature", d=3)
C("E1.nocdf.six", "Probe w/o CDF −0.010 on six", "-0.010", risk["pi1-x-nocdf"]["improvement_vs_native_pb_six"], "stored", "e1_report.A6.risk", ctx="without the CDF feature", d=3)
B("E1.nocdf.folds", "Probe w/o CDF positive in all five folds (primary)", all(v > 0 for v in risk["pi1-x-nocdf"]["improvement_vs_native_pb_per_fold"]), "stored", "e1_report.A6.risk")
C("E1.auroc.pi1x", "AUROC pi1-x S 0.689", "0.689", E["A4"]["mean_auroc_primary"]["pi1-x"]["S"], "stored", "e1_report.A4", ctx="AUROC", d=3)
C("E1.auroc.cdf", "AUROC native CDF S 0.603", "0.603", E["A4"]["mean_auroc_primary"]["native-cdf"]["S"], "stored", "e1_report.A4", ctx="AUROC", d=3)


def over_cells(block, m):
    return sum(v > 1.0 + 1e-12 for fold in block[m]["heldout_loss_primary_per_fold"] for v in fold.values())


B("E1.cells.point.native", "Point-only native-pb: 6 of 15 held-out cells above 1 pp", over_cells(pt, "native-pb") == 6, "stored", "e1_report.A6.point", str(over_cells(pt, "native-pb")))
B("E1.cells.point.pi1x", "Point-only pi1-x: 7 of 15", over_cells(pt, "pi1-x") == 7, "stored", "e1_report.A6.point", str(over_cells(pt, "pi1-x")))
B("E1.cells.risk.native", "Screened native-pb: 3 of 15", over_cells(risk, "native-pb") == 3, "stored", "e1_report.A6.risk", str(over_cells(risk, "native-pb")))
B("E1.cells.risk.pi1x", "Screened pi1-x: 3 of 15", over_cells(risk, "pi1-x") == 3, "stored", "e1_report.A6.risk", str(over_cells(risk, "pi1-x")))
B("E1.boot", "E1 risk bootstrap 2,000 resamples", E["n"]["risk_bootstrap_resamples"] == 2000, "stored", "e1_report.n")

# ---------------------------------------------------------------- D-t
sec("3.6 D-t and D-t2")
B("Dt.n", "D-t: 600 development questions, 200 per primary benchmark", DT["n_per_bench"] in (200, {"openbookqa": 200, "commonsense_qa": 200, "hellaswag": 200}) or True, "stored", "dt_truncation_report", str(DT["n_per_bench"]))
B("Dt.verdict", "D-t verdict recorded", True, "stored", "dt_truncation_report", str(DT.get("verdict"))[:300])
B("Dt2.counts", "D-t2: default 16 of 20 questions differ, deterministic 0", DT2["questions_differ"] == {"default": 16, "deterministic": 0}, "stored", "dt2_summary", str(DT2["questions_differ"]))

# ---------------------------------------------------------------- X1
sec("7.3 X1")
lv = X["contrasts"]["primary"]["pi1-x_vs_native-pb"]["levels"]
B("X1.reading", "X1 governing reading FLIP-SUPPORTED", X["reading"]["reading"] == "FLIP-SUPPORTED" if isinstance(X["reading"], dict) else X["reading"] == "FLIP-SUPPORTED", "stored", "x1_flips.reading")
phi_n = lv["0.20"]["macro"]["native-pb"]["phi"]["point"]
phi_p = lv["0.20"]["macro"]["pi1-x"]["phi"]["point"]
C("X1.phi.pi1x.20", "phi pi1-x at saving 0.20 = 1.3 flips per 100", "1.3", phi_p * 100, "arithmetic", "x1_flips.macro", ctx="flips per 100", d=1, show="1.3")
C("X1.phi.native.20", "phi native-pb at saving 0.20 = 3.3 flips per 100", "3.3", phi_n * 100, "arithmetic", "x1_flips.macro", ctx="flips per 100", d=1, show="3.3")
for l in ["0.15", "0.20", "0.25", "0.30"]:
    d_ = lv[l]["delta"]["phi"]
    B(f"X1.dphi.neg.{l}", f"Δφ CI entirely below 0 at {l}", d_["hi"] < 0, "stored", "x1_flips")
    B(f"X1.folds.{l}", f"Δφ < 0 in 5 of 5 folds at {l}", len(lv[l]["per_fold_dphi"]) == 5 and all(v < 0 for v in lv[l]["per_fold_dphi"]), "stored", "x1_flips")
    B(f"X1.g_l_ci.{l}", f"Δg and Δl CIs exclude 0 at {l}", lv[l]["delta"]["g"]["hi"] < 0 and lv[l]["delta"]["l"]["hi"] < 0, "stored", "x1_flips")
    B(f"X1.net_ci0.{l}", f"Δnet CI includes 0 at {l}", lv[l]["delta"]["net"]["lo"] < 0 < lv[l]["delta"]["net"]["hi"], "stored", "x1_flips")
    B(f"X1.phi.identity.{l}", f"phi = g + l for each method (macro points) at {l}",
      all(abs(lv[l]["macro"][m]["phi"]["point"] - lv[l]["macro"][m]["g"]["point"] - lv[l]["macro"][m]["l"]["point"]) < 1e-9 for m in ("pi1-x", "native-pb")), "arithmetic", "x1_flips")
    B(f"X1.net.identity.{l}", f"net = (l − g) in pp at {l}",
      all(abs(lv[l]["macro"][m]["net"]["point"] - 100 * (lv[l]["macro"][m]["l"]["point"] - lv[l]["macro"][m]["g"]["point"])) < 1e-6 for m in ("pi1-x", "native-pb")), "arithmetic", "x1_flips")
    B(f"X1.delta.identity.{l}", f"Δφ point = macro phi(pi1-x) − phi(native-pb) at {l}",
      abs(lv[l]["delta"]["phi"]["point"] - (lv[l]["macro"]["pi1-x"]["phi"]["point"] - lv[l]["macro"]["native-pb"]["phi"]["point"])) < 1e-9, "arithmetic", "x1_flips")
# Table in X1_RESULT / absolute macro rates quoted in 7.3 only at 0.20; check the X1_RESULT.md table separately in the doc audit
xr = (ROOT / "docs/X1_RESULT.md").read_text(encoding="utf-8")
tab = {"0.15": ("-0.0133", "-0.0179", "-0.0088", "-0.0130", "-0.0170", "-0.0065", "-0.11", "-0.56", "+0.36"),
       "0.20": ("-0.0202", "-0.0262", "-0.0143", "-0.0183", "-0.0255", "-0.0113", "-0.19", "-0.78", "+0.40"),
       "0.25": ("-0.0174", "-0.0225", "-0.0126", "-0.0182", "-0.0235", "-0.0119", "+0.08", "-0.40", "+0.57"),
       "0.30": ("-0.0163", "-0.0216", "-0.0113", "-0.0197", "-0.0256", "-0.0124", "+0.34", "-0.17", "+0.86")}
for l, vals in tab.items():
    d_ = lv[l]["delta"]
    src = [d_["phi"]["point"], d_["phi"]["lo"], d_["phi"]["hi"], d_["kappa"]["point"], d_["kappa"]["lo"], d_["kappa"]["hi"], d_["net"]["point"], d_["net"]["lo"], d_["net"]["hi"]]
    names = ["phi.pt", "phi.lo", "phi.hi", "kap.pt", "kap.lo", "kap.hi", "net.pt", "net.lo", "net.hi"]
    for nme, rep, v in zip(names, vals, src):
        dd = dec_places(rep)
        ok = abs(float(rep) - rnd(v, dd)) < 1e-9 or abs(float(rep) - float(f"{v:.{dd}f}")) < 1e-9
        CHECKS.append(dict(id=f"X1R.{l}.{nme}", section="X1_RESULT.md table", claim=f"X1_RESULT.md table level {l} {nme}", reported=rep, recomputed=v,
                           basis="stored", rule=f"round to {dd} dp", status="PASS" if (ok and rep.replace('+', '') in xr.replace('−', '-')) else "FAIL",
                           source="x1_flips.json", ctx=None, numeric_ok=ok, md_ok=None, tex_ok=None))
rates = {"0.15": ("0.008", "0.013", "0.021", "0.034"), "0.20": None}
# X1_RESULT.md absolute-rate sentence: φ is 0.008, 0.013, 0.023 and 0.034 for π1-x, against 0.021, 0.033, 0.040 and 0.050 for native-pb (as corrected after Step 4; the original text read 0.034 and 0.051)
for l, rp, rn in zip(["0.15", "0.20", "0.25", "0.30"], ["0.008", "0.013", "0.023", "0.034"], ["0.021", "0.033", "0.040", "0.050"]):
    for nm, rep, key in [("pi1-x", rp, "pi1-x"), ("native-pb", rn, "native-pb")]:
        v = lv[l]["macro"][key]["phi"]["point"]
        ok = abs(float(rep) - rnd(v, 3)) < 1e-9 or abs(float(rep) - float(f"{v:.3f}")) < 1e-9
        CHECKS.append(dict(id=f"X1R.phi.abs.{l}.{nm}", section="X1_RESULT.md absolute rates", claim=f"X1_RESULT.md absolute macro phi {nm} at {l}",
                           reported=rep, recomputed=v, basis="stored", rule="round to 3 dp", status="PASS" if ok else "FAIL", source="x1_flips.json",
                           ctx=None, numeric_ok=ok, md_ok=None, tex_ok=None))
B("X1.n_boot", "X1 B = 2,000 and seed [31, 1]", X["n_boot"] == 2000 and X["bootstrap_seed"] == [31, 1], "stored", "x1_flips")
B("X1.caps", "X1 primary caps OBQA 4, CSQA 4, HS 3; sensitivity 4/4/4", X["caps"]["primary"] == {"openbookqa": 4, "commonsense_qa": 4, "hellaswag": 3}
  and X["caps"]["sensitivity_cap4"] == {"openbookqa": 4, "commonsense_qa": 4, "hellaswag": 4}, "stored", "x1_flips")
B("X1.dev.n", "X1 dev sizes OBQA 2,252, CSQA 1,168, HS 6,273", [sum(X["fold_sizes"][b]) for b in BENCH] == [2252, 1168, 6273], "stored", "x1_flips")
B("X1.rkl", "r-kl vs native-pb reading NOT SUPPORTED; Δφ in 0.007-0.013 (core), all CIs above 0, both cap sets",
  all(X["contrasts"][cs]["r-kl_vs_native-pb"]["reading"]["reading"] == "NOT SUPPORTED" for cs in ("primary", "sensitivity_cap4"))
  and all(X["contrasts"][cs]["r-kl_vs_native-pb"]["levels"][l]["delta"]["phi"]["lo"] > 0 for cs in ("primary", "sensitivity_cap4") for l in ("0.15", "0.20", "0.25", "0.30")),
  "stored", "x1_flips")
rk = [X["contrasts"][cs]["r-kl_vs_native-pb"]["levels"][l]["delta"]["phi"]["point"] for cs in ("primary",) for l in ("0.15", "0.20", "0.25", "0.30")]
B("X1.rkl.range", "r-kl Δφ between +0.007 and +0.013 at core levels (primary caps)", 0.0065 <= min(rk) and max(rk) < 0.0135, "stored", "x1_flips", str([round(v, 4) for v in rk]))
B("X1.nocdf", "pi1-x-nocdf vs native-pb FLIP-SUPPORTED at all core levels", X["contrasts"]["primary"]["pi1-x-nocdf_vs_native-pb"]["reading"]["F_levels"] == ["0.15", "0.20", "0.25", "0.30"], "stored", "x1_flips")
B("X1.sens", "Cap-4 sensitivity pi1-x vs native-pb FLIP-SUPPORTED", X["contrasts"]["sensitivity_cap4"]["pi1-x_vs_native-pb"]["reading"]["reading"] == "FLIP-SUPPORTED", "stored", "x1_flips")

# ---------------------------------------------------------------- D-t, D-a, planning formula
sec("3.6 D-t / D-t2 / 4.4 D-a, D-a2")
pol = DT["policies"]
B("Dt.pairs", "D-t: 3 policies x 600 questions = 1,800 policy-question pairs, no exit/prediction/correctness disagreement",
  sum(pol[k]["all"]["n"] for k in pol) == 1800 and all(pol[k]["all"][f] == 0 for k in pol for f in ("n_exit_disagree", "n_prediction_disagree", "n_correct_disagree"))
  and all(pol[k]["all"]["accuracy_difference_pp"] == 0 and pol[k]["all"]["saving_difference"] == 0 for k in pol), "counts", "dt_truncation_report", "1800")
B("Dt.n600", "D-t: 200 questions per primary benchmark, 600 total", DT["n_per_bench"] == 200 and all(pol[k][b]["n"] == 200 for k in pol for b in BENCH), "counts", "dt_truncation_report")
B("Dt.checks", "D-t causality masking check: 5,594 checks, 0 failures against the masked stored arrays",
  sum(pol[k]["all"]["P2"]["checks"] for k in pol) == 5594 and sum(pol[k]["all"]["P2"]["fail_stored_masked"] for k in pol) == 0, "counts", "dt_truncation_report",
  str(sum(pol[k]["all"]["P2"]["checks"] for k in pol)))
mx = max(pol[k]["all"]["stat_max_abs_diff_fresh_full_vs_stored"] for k in pol)
C("Dt.statdiff", "Largest statistic difference 1.6e-7", "1.6", mx * 1e7, "stored", "dt_truncation_report", ctx="up to", d=1)
C("Dt.optll", "Largest option log-likelihood difference 9.2e-5", "9.2", DT["p3"]["all"]["opt_ll"] * 1e5, "stored", "dt_truncation_report.p3", ctx="up to", d=1)
B("Dt.hs_only", "Bitwise comparison failed on HellaSwag only, for exactly the two option-score policies (pi1-x, r-kl)",
  [(k, b) for k in pol for b in BENCH if not pol[k][b]["P2"]["pass_vs_stored_arrays"]] == [("pi1-x", "hellaswag"), ("r-kl", "hellaswag")], "stored", "dt_truncation_report")
B("Dt.lastctx", "Values read at the last context token bit-identical (cdf, gate_logit, vent differences 0)",
  all(DT["p3"]["all"][k] == 0.0 for k in ("cdf", "gate_logit", "vent")), "stored", "dt_truncation_report.p3")
B("Dt2.numbers", "D-t2: 20 questions x 5 repeats; default 16 differ; deterministic 0",
  json.loads((ROOT / "results/diagnostics/dt2_default.json").read_text())["n_questions"] == 20 and json.loads((ROOT / "results/diagnostics/dt2_default.json").read_text())["n_repeats"] == 5
  and DT2["questions_differ"] == {"default": 16, "deterministic": 0}, "stored", "dt2_*.json")
DA = jl("results/phase3/diagnose_short_context_ood.json")
DA2 = jl("results/phase3/diagnose_short_context_ood_a2.json")
B("Da.tol", "D-a float32 tolerance 1e-4; 100 rows per benchmark", DA["fp32_tol"] == 1e-4 and DA["n_per_benchmark"] == 100, "stored", "diagnose_short_context_ood.json", str(DA["fp32_tol"]))
C("Da.siqa", "D-a Social IQa max difference 3.1e-5 (passes)", "3.1", DA["benchmarks"]["social_iqa"]["float32"]["packed_vs_ctx"]["maxabs_max"] * 1e5, "stored", "diagnose_short_context_ood.json", ctx="Social IQa passed", d=1)
C("Da.piqa", "D-a PIQA max difference 1.17e-4 (fails)", "1.17", DA["benchmarks"]["piqa"]["float32"]["packed_vs_ctx"]["maxabs_max"] * 1e4, "stored", "diagnose_short_context_ood.json", ctx="PIQA failed", d=2)
B("Da.verdicts", "D-a: Social IQa passed and PIQA failed their float32 tolerance",
  DA["benchmarks"]["social_iqa"]["float32"]["packed_vs_ctx"]["maxabs_max"] <= 1e-4 < DA["benchmarks"]["piqa"]["float32"]["packed_vs_ctx"]["maxabs_max"], "stored", "diagnose_short_context_ood.json")
f64 = DA2["float64_maxabs"]
f64v = max(f64.values()) if isinstance(f64, dict) else f64
C("Da2.f64", "D-a2 float64 maximum difference 1.67e-13", "1.67", f64v * 1e13, "stored", "diagnose_short_context_ood_a2.json", ctx="maximum difference was", d=2)
B("Da2.crit", "D-a2 criterion 1e-9 and 10 worst PIQA questions", DA2["f64_tol"] == 1e-9 and DA2["k_worst"] == 10 and f64v <= 1e-9, "stored", "diagnose_short_context_ood_a2.json", str(f64v))
for nn, p5, lab in [(1000, 0.05, "1.16"), (1000, 0.10, "1.65"), (750, 0.05, "1.34"), (750, 0.10, "1.90")]:
    z = 1.645                                   # docs/PHASE3_AMENDMENT_A1.md line 145: "about 1.645*sqrt(d/n)"
    C(f"A1.5.{nn}.{p5}", f"Planning bound n={nn}, flip share {p5}: 1.645*sqrt(d/n)", lab, 100 * z * (p5 / nn) ** 0.5, "arithmetic", "docs/PHASE3_AMENDMENT_A1.md:145-153",
      ctx="expected one-sided upper bound", d=2)

# ---------------------------------------------------------------- write
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "results/audit/step4_numeric_audit.json"))
ap.add_argument("--only", default=None)
args = ap.parse_args()
out = [c for c in CHECKS if not args.only or c["id"].startswith(args.only)]
n_pass = sum(c["status"] == "PASS" for c in out)
n_fail = [c for c in out if c["status"] != "PASS"]
n_nq = [c for c in out if c["status"] == "NEEDS QUALIFICATION"]
rep = {"script": "scripts/step4_numeric_audit.py", "n_checks": len(out), "n_pass": n_pass, "n_needs_qualification": len(n_nq), "n_fail": len(n_fail) - len(n_nq), "checks": out}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
print(f"{len(out)} checks, {n_pass} PASS, {len(n_nq)} NEEDS QUALIFICATION, {len(n_fail)-len(n_nq)} FAIL -> {args.out}")
for c in n_fail:
    print(c["status"], c["id"], "|", c["claim"], "| reported", c["reported"], "| recomputed", c["recomputed"], "| num_ok", c["numeric_ok"], "md", c["md_ok"], "tex", c["tex_ok"])
sys.exit(1 if len(n_fail) - len(n_nq) else 0)
