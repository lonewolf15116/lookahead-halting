"""Gate 2, step 3: fit probes on TRAIN, select and freeze on VALIDATION only (docs/GATE2_PLAN.md v3, sections 5-6).

    python scripts/gate2_select.py [--root PATH] [--benchmarks ...] [--families 1 2 3] [--dry-run] [--force-refreeze]

Reads only data/extracted/<benchmark> train and validation rows. It never opens the Gate 1 test split, data/extracted_confirm
or data/questions_confirm (explicit assertions below and in gate2.load_probe_data).
Writes results/gate2/frozen_selection.json (+ models/*.npz), results/gate2/validation_report.json and
docs/GATE2_VALIDATION_REPORT.md. --dry-run (family 1 only, no secondary report) writes to results/gate2_dryrun/ and
docs/GATE2_VALIDATION_REPORT_DRYRUN.md instead, so it can never occupy the real frozen path.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

from lookahead import gate1 as g
from lookahead import gate2 as g2
from lookahead.config import REPO_ROOT
from lookahead.questions import BENCHMARKS, SYNTHETIC

LABEL = "validation (not the Gate 2 verdict)"


def assert_inputs_allowed(root: Path, benchmarks) -> None:
    for b in benchmarks:
        assert b in BENCHMARKS and b not in SYNTHETIC, f"{b!r} is not one of the six natural benchmarks"
    for p in (root, root / "data" / "extracted"):
        g2._assert_allowed_path(p)
    # the sealed confirmation directories are never touched, not even listed
    assert "extracted_confirm" not in str(root / "data" / "extracted")
    assert "questions_confirm" not in str(root / "data" / "extracted")


def md_configs(records, chosen):
    rows = []
    for r in records:
        rows.append([r["name"], "yes" if r["qualified"] else "no", g.f(r["threshold"], 6), g.f(r["saving"]),
                     g.f(r["loss"]), g.f(r["acc"], 2), "**chosen**" if chosen is not None and r["name"] == chosen["name"] else ""])
    return g.md_table(["config", "qualified", "threshold", "val macro saving", "val macro loss (pp)", "val macro acc (%)",
                       ""], rows)


def md_policies(rows):
    return g.md_table(["policy", "qualified", "threshold", "macro saving", "macro loss (pp)", "macro acc (%)"],
                      [[n, "yes" if q else "no", g.f(t, 6), g.f(s), g.f(l), g.f(a, 2)] for n, q, t, s, l, a in rows])


def md_secondary(sec):
    if not sec:
        return "_skipped (dry run)_\n"
    rows = []
    for key, v in sec.items():
        if key.startswith("WCvsCW"):
            continue
        rows.append([key, v["metric"], g.f(v["pooled"]), *[g.f(v["per_loop"][str(t)]) for t in (1, 2, 3)]])
    out = g.md_table(["target/input", "metric", "pooled", "loop 1", "loop 2", "loop 3"], rows)
    rows = [[k, v["n_pairs"], v["n_wc"], v["n_cw"], g.f(v["auroc_wc_vs_cw_direction_probes"]),
             g.f(v["auroc_wc_vs_cw_from_F_probe"])] for k, v in sec.items() if k.startswith("WCvsCW")]
    out += "\nWC vs CW among WC/CW transitions (positive = WC):\n\n"
    out += g.md_table(["input", "pairs", "WC", "CW", "AUROC P(WC)-P(CW)", "AUROC F-probe score"], rows)
    return out


def render(report: dict) -> str:
    out = [f"# Gate 2 validation report ({LABEL})\n",
           f"Generated {report['timestamp_utc']}. Plan {g2.PLAN_VERSION}. Train and validation splits only: the Gate 1 test "
           "split and the confirmation set were not opened. Nothing here is a Gate 2 verdict.\n",
           "Macro = unweighted mean over benchmarks. Operating point: Gate 1's rule (max saving with macro validation loss "
           "<= 1 pp).\n",
           "## Failed questions\n",
           "Questions with status != ok are excluded from probe fitting and kept in policy evaluation exactly as in Gate 1 "
           "(wrong at every loop, 4 loops, no statistic fires).\n",
           g.md_table(["benchmark", "train n", "train failed", "validation n", "validation failed"],
                      [[b, report["n_failed"]["train"][b]["n"], report["n_failed"]["train"][b]["failed"],
                        report["n_failed"]["validation"][b]["n"], report["n_failed"]["validation"][b]["failed"]]
                       for b in report["benchmarks"]]),
           "## Probe configurations (validation)\n", md_configs(report["configs"], report["chosen"]),
           "## P (chosen), Q (cheap features) and baselines at their validation operating points\n",
           md_policies(report["policy_rows"]),
           "## Controls\n", md_policies(report["control_rows"]),
           "## Hyper-parameters chosen on train\n"]
    for r in report["configs"] + report["control_records"]:
        out.append(f"- {r['name']}: {r['hyper']}")
    out += ["\n## Secondary predictive quality on validation (family 1; no selection uses these)\n",
            md_secondary(report["secondary"]),
            "\n## Probe cost (multiply-adds per question per loop, chosen P)\n",
            f"{report['p_macs']}\n",
            "\n## Selection rules\n", g2.PROBE_SELECTION_RULES + "\n"]
    if report["chosen"] is None:
        out.append("\n**No probe configuration qualified (no grid point within the 1 pp budget).**\n")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO_ROOT))
    ap.add_argument("--benchmarks", nargs="+", default=list(BENCHMARKS))
    ap.add_argument("--families", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force-refreeze", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    root = Path(args.root)
    benchmarks = list(args.benchmarks)
    assert_inputs_allowed(root, benchmarks)
    families = [1] if args.dry_run else sorted(set(args.families))
    if not set(families) <= {1, 2, 3}:
        raise SystemExit("families must be among 1 2 3")
    suffix = "_dryrun" if args.dry_run else ""
    out_dir = root / "results" / f"gate2{suffix}"
    if (out_dir / "frozen_selection.json").exists() and not args.force_refreeze:
        print(f"refusing: {out_dir / 'frozen_selection.json'} exists (use --force-refreeze to archive and replace)",
              file=sys.stderr)
        return 2

    t0 = time.time()
    prov = {}
    train = g2.load_probe_data(root, "train", benchmarks, provenance=prov)
    val = g2.load_probe_data(root, "validation", benchmarks, g_clip=train.g_clip, provenance=prov)
    assert set(np.unique(train.bench)) == set(benchmarks) == set(np.unique(val.bench))
    print(f"loaded train {train.N} / validation {val.N} questions, d={train.d} ({time.time() - t0:.0f}s)", flush=True)

    # baselines on validation, Gate 1 code unchanged
    base_sel = g.run_selection(val.arrays, benchmarks)

    records, probes = [], {}
    for fam in families:
        for var in g2.VARIANTS:
            pr = g2.fit_probe(train, fam, var, "S", device=args.device)
            op = g2.select_operating_point(g2.probe_policy_data(val, pr.predict(val)), benchmarks)
            rec = g2.config_record(pr, op)
            records.append(rec)
            probes[rec["name"]] = pr
            print(f"  {rec['name']}: qualified={rec['qualified']} saving={rec['saving']} loss={rec['loss']} "
                  f"({time.time() - t0:.0f}s)", flush=True)
    chosen = g2.choose_config(records)

    # controls: Q (family 1 on cheap features, the one used in the verdict) and (c) label-shuffled h
    q_pr = g2.fit_probe(train, 1, "x", "S")
    q_op = g2.select_operating_point(g2.probe_policy_data(val, q_pr.predict(val)), benchmarks)
    q_rec = g2.config_record(q_pr, q_op, "Q-family1-x")
    shuf_pr = g2.fit_probe(train, 1, "h", "S", labels=g2.shuffle_labels_within(train, train.targets["S"]))
    shuf_pred = shuf_pr.predict(val)
    shuf_op = g2.select_operating_point(g2.probe_policy_data(val, shuf_pred), benchmarks)
    shuf_rec = g2.config_record(shuf_pr, shuf_op, "control-c-family1-h-shuffled")
    for r in (q_rec, shuf_rec):
        print(f"  {r['name']}: qualified={r['qualified']} saving={r['saving']} loss={r['loss']}", flush=True)

    p_name = None if chosen is None else chosen["name"]
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    frozen = g2.freeze_gate2(
        root, chosen_rec=chosen, chosen_probe=None if chosen is None else probes[p_name], q_rec=q_rec, q_probe=q_pr,
        baseline_selection=base_sel, benchmarks=benchmarks, all_records=records + [q_rec, shuf_rec], out_dir=out_dir,
        force_refreeze=args.force_refreeze, timestamp=stamp,
        provenance={"parquet_and_hidden_sha256": prov, "families": families, "dry_run": bool(args.dry_run),
                    "g_clip_train": list(train.g_clip)})

    # secondary predictive report: S for P / Q / shuffled, then the other targets (family 1)
    secondary = {}
    if not args.dry_run:
        secondary = g2.secondary_report(train, val, args.device)
    s_metrics = {}
    if chosen is not None:
        s_metrics["S/P:" + p_name] = g2.predictive_metrics(val, probes[p_name].predict(val), "S")
    s_metrics["S/Q:family1-x"] = g2.predictive_metrics(val, q_pr.predict(val), "S")
    s_metrics["S/control-c:family1-h-shuffled"] = g2.predictive_metrics(val, shuf_pred, "S")
    secondary = {**s_metrics, **secondary}

    policy_rows = []
    if chosen is not None:
        policy_rows.append(("P " + p_name, chosen["qualified"], chosen["threshold"], chosen["saving"], chosen["loss"],
                            chosen["acc"]))
    policy_rows.append(("Q " + q_rec["name"], q_rec["qualified"], q_rec["threshold"], q_rec["saving"], q_rec["loss"],
                        q_rec["acc"]))
    for name in g.CHEAP_BASELINES:
        b = base_sel["baselines"][name]
        s = b["selected"]
        policy_rows.append((name, b["qualified"], b["threshold"], s and s["macro_saving"], s and s["macro_loss"],
                            s and s["macro_acc"]))
    control_rows = [(shuf_rec["name"], shuf_rec["qualified"], shuf_rec["threshold"], shuf_rec["saving"],
                     shuf_rec["loss"], shuf_rec["acc"])]
    report = {"label": LABEL, "timestamp_utc": stamp, "plan_version": g2.PLAN_VERSION, "benchmarks": benchmarks,
              "dry_run": bool(args.dry_run), "families": families,
              "n_failed": {"train": train.n_failed, "validation": val.n_failed},
              "configs": records, "chosen": chosen, "control_records": [q_rec, shuf_rec],
              "policy_rows": policy_rows, "control_rows": control_rows, "secondary": secondary,
              "p_macs": None if chosen is None else g2.probe_macs(probes[p_name]),
              "baseline_selection": {n: {"qualified": b["qualified"], "selected": b["selected"]}
                                     for n, b in base_sel["baselines"].items()},
              "hyper_grids": {n: p.hyper["grid"] for n, p in probes.items()},
              "frozen_file": (out_dir / "frozen_selection.json").relative_to(root).as_posix()}
    g.write_json(out_dir / "validation_report.json", report)
    doc = root / "docs" / f"GATE2_VALIDATION_REPORT{'_DRYRUN' if args.dry_run else ''}.md"
    doc.parent.mkdir(parents=True, exist_ok=True)
    doc.write_text(render(report), encoding="utf-8")
    print(f"[{LABEL}] chosen P: {p_name}; Q saving {q_rec['saving']}; frozen -> {report['frozen_file']} "
          f"({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
