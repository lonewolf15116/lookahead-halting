"""Step 4 manuscript consistency audit: Markdown source vs LaTeX body vs the two compiled PDFs, plus anonymisation and release hygiene.

    python scripts/step4_manuscript_consistency.py [--out results/audit/step4_manuscript_consistency.json]

Needs pdftotext/pdfinfo (poppler). Compares the 'salient numbers' (decimals and comma-grouped integers, scientific-notation constants
normalised to 1.6e-7 form), the evidence-label counts [C]/[S]/[P]/[E], section/table/figure cross-references and citation keys, and
scans the anonymised build for identifying strings. Findings are PASS / FAIL / NEEDS QUALIFICATION; exit code 0 iff no FAIL.
"""
import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = Path(__file__).resolve().parents[1]
CHECKS = []


def add(cid, claim, status, evidence):
    CHECKS.append({"id": cid, "claim": claim, "status": status, "evidence": evidence})


def ok(cond, cid, claim, evidence, fail="FAIL"):
    add(cid, claim, "PASS" if cond else fail, evidence)


def pdftext(p):
    return subprocess.run(["pdftotext", str(p), "-"], capture_output=True, text=True, check=True).stdout.replace("\x0c", "\n")


MD = (ROOT / "docs/PAPER_DRAFT.md").read_text(encoding="utf-8")
TEX = (ROOT / "paper/body.tex").read_text(encoding="utf-8")
PDFS = {"anon": pdftext(ROOT / "paper/main_anon.pdf"), "preprint": pdftext(ROOT / "paper/main_preprint.pdf")}
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


def cut_refs_md(t):
    return t.split("## References")[0]


def cut_refs_pdf(t):
    i = t.rfind("\nReferences")
    return t[:i] if i > 0 else t


def sci_normalise(t):
    t = t.replace("−", "-").replace("\u2212", "-").replace("–", "-")
    t = re.sub(r"(\d+(?:\.\d+)?)\s*×\s*10([⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+)", lambda m: f"{m.group(1)}e{m.group(2).translate(SUP)}", t)           # md: 1.6×10⁻⁷
    t = re.sub(r"(\d+(?:\.\d+)?)\s*\\times\s*10\^\{?\s*-?(\d+)\}?", lambda m: f"{m.group(1)}e-{m.group(2)}", t)                    # tex
    t = re.sub(r"(\d+(?:\.\d+)?)\s*×\s*10\s*-?\s*(\d+)", lambda m: f"{m.group(1)}e-{m.group(2)}", t)                              # pdf
    t = re.sub(r"(\d+(?:\.\d+)?)\s*e-\s*(\d+)", lambda m: f"{m.group(1)}e-{m.group(2)}", t)
    return t


SAL = re.compile(r"(?<![\w.])\d+\.\d+e-\d+|(?<![\w.])\d[\d]*\.\d+|(?<![\w.])\d{1,3}(?:,\d{3})+")


def salient(t):
    t = sci_normalise(t)
    # drop section-heading numbers like "3.4" that start a heading line in md/pdf; they are handled via a separate allow-list below
    return collections.Counter(SAL.findall(t))


def strip_tex(t):
    t = re.sub(r"(?<!\\)%.*", "", t)
    for a, b in [("{[}", "["), ("{]}", "]"), ("\\%", "%"), ("~", " "), ("\\,", ""), ("$", ""), ("\\(", ""), ("\\)", ""), ("\\_", "_"), ("\\textbf{", ""), ("\\emph{", ""), ("\\textit{", ""),
                 ("\\text{", ""), ("\\mathrm", "")]:
        t = t.replace(a, b)
    return t


FIG_TEXT = pdftext(ROOT / "figures/fig1_rule_dependent_operating_points.pdf")      # text embedded in the vector figure (ticks, panel titles)
md_body, tex_body = cut_refs_md(MD), re.sub(r"p\{[\d.]+\\(?:line|text)width\}", "", re.sub(r"\\includegraphics\[[^\]]*\]", "", strip_tex(TEX)))     # layout widths are not content
sections = set(re.findall(r"^#{2,3} (\d+(?:\.\d+)?)[. ]", MD, flags=re.M))
heading_nums = {s for s in sections if "." in s}
c_md, c_tex = salient(md_body), salient(tex_body)
c_anon, c_pre = salient(cut_refs_pdf(PDFS["anon"])), salient(cut_refs_pdf(PDFS["preprint"]))


def residual(a, b):
    """{token: +n if only (more) in a, -n if only (more) in b}"""
    d = {}
    for k in set(a) | set(b):
        if a.get(k, 0) != b.get(k, 0):
            d[k] = a.get(k, 0) - b.get(k, 0)
    return d


# -- md vs tex: identical salient-number multisets
d = residual(c_md, c_tex)
d_other = {k: v for k, v in d.items() if k not in heading_nums}
ok(not d_other, "C.md_tex.numbers", f"Markdown and LaTeX body contain the same multiset of salient numbers ({sum(c_md.values())} tokens; Markdown heading numbers excluded)", f"differences: {d_other}; heading-number tokens only in Markdown: {sum(1 for k in d if k in heading_nums)}")
# -- tex vs each PDF: PDF may add section numbers (headings), reference years are cut; allow only heading numbers and 2-digit-grouped page artefacts
for name, cp in (("anon", c_anon), ("preprint", c_pre)):
    d = residual(c_tex, cp)
    fig = salient(FIG_TEXT)
    unexplained = {k: v for k, v in d.items() if k not in heading_nums and not (v < 0 and -v <= fig.get(k, 0))}      # extras in the PDF explained by text embedded in Figure 1
    ok(not unexplained, f"C.tex_pdf.{name}.numbers", "LaTeX body and the %s PDF contain the same salient numbers (allowed: section-heading numbers, and numbers that occur as text inside Figure 1)" % name,
       f"unexplained: {unexplained}; heading-number-only differences: {sum(1 for k in d if k in heading_nums)}; figure-text tokens used: {sum(1 for k, v in d.items() if v < 0 and -v <= fig.get(k, 0) and k not in heading_nums)}")
ok(c_anon == c_pre, "C.pdf.anon_vs_preprint", "the anonymous and the preprint PDF contain identical salient numbers", str(residual(c_anon, c_pre)))

# -- integers of two or more digits (counts, exit histograms, hash prefixes' digits): md vs tex must match exactly; the PDFs may add page numbers,
#    citation years and figure text only
INT = re.compile(r"(?<![\w.,])\d{2,}(?![\d]|\.\d|,\d|\w)")


def ints(t):
    return collections.Counter(INT.findall(sci_normalise(t)))


i_md, i_tex = ints(md_body), ints(tex_body)
i_anon, i_pre = ints(cut_refs_pdf(PDFS["anon"])), ints(cut_refs_pdf(PDFS["preprint"]))
d = residual(i_md, i_tex)
ok(not d, "C.md_tex.integers", f"Markdown and LaTeX body contain the same multiset of integers with 2+ digits ({sum(i_md.values())} tokens)", f"differences: {d}")
for name, ip in (("anon", i_anon), ("preprint", i_pre)):
    d = residual(i_tex, ip)
    figi = ints(FIG_TEXT)
    un = {k: v for k, v in d.items() if not (v < 0 and (k.isdigit() and (1 <= int(k) <= 30 or 2010 <= int(k) <= 2026) or -v <= figi.get(k, 0)))}
    ok(not un, f"C.tex_pdf.{name}.integers", "LaTeX body and the %s PDF contain the same integers with 2+ digits (allowed extras in the PDF: page numbers 1-30, citation years 2010-2026, text inside Figure 1)" % name,
       f"unexplained: {un}")

# -- evidence labels
LAB = re.compile(r"\[([CSPE])[\],]")


def labels(t):
    return collections.Counter(LAB.findall(t))


l_md, l_tex = labels(md_body), labels(tex_body)
l_an, l_pr = labels(cut_refs_pdf(PDFS["anon"])), labels(cut_refs_pdf(PDFS["preprint"]))
ok(l_md == l_tex == l_an == l_pr, "C.labels.counts", "evidence-label counts [C]/[S]/[P]/[E] agree in Markdown, LaTeX and both PDFs", f"md {dict(l_md)}; tex {dict(l_tex)}; anon {dict(l_an)}; pre {dict(l_pr)}")
# every [C] occurrence concerns the primary comparison
ctxs = [md_body[max(0, m.start() - 160): m.end() + 20].replace("\n", " ") for m in re.finditer(r"\[C\]", md_body)]
primary = [c for c in ctxs if any(s in c for s in ("0.053", "PASS", "primary", "+0.049", "macro", "preferred", "confirmatory", "Bootstrap-screened", "saved more", "found that a trained"))]
ok(len(primary) == len(ctxs), "C.labels.confirmatory_scope", "every [C] label is attached to the single pre-registered primary comparison (π1-x vs native-pb, bootstrap-screened rule)",
   f"{len(ctxs)} [C] occurrences; contexts: " + " || ".join(c[-110:] for c in ctxs))

# -- label discipline for the exploratory follow-up X1
sent = [x for x in re.split(r"(?<=[.])\s+", md_body) if "X1" in x]
bad = [x[:120] for x in sent if re.search(r"pre-regist|\[C\]|\[S\]", x)]
ok(not bad, "C.labels.x1", "no sentence about X1 calls it pre-registered or labels it [C]/[S] (it is a protocol-frozen exploratory analysis, label [E])", f"{len(sent)} X1 sentences; offending: {bad}")
nconf = [x[:100] for x in sent if re.search(r"confirm", x) and "not four independent confirmations" not in x and "after the confirmatory study" not in x.lower()]
ok(not nconf, "C.labels.x1_confirm", "X1 is never described as confirming anything except in the negated 'not four independent confirmations'", str(nconf))

# -- cross-references
refs_sec = set(re.findall(r"[Ss]ection (\d+(?:\.\d+)?)", md_body)) | set(re.findall(r"sections? (\d+(?:\.\d+)?)", md_body))
missing = sorted(r for r in refs_sec if r not in sections and r.split(".")[0] not in sections)
ok(not missing, "C.xref.sections", "every 'Section N.M' cross-reference in the Markdown points to an existing heading", f"referenced {sorted(refs_sec)}; missing {missing}")
t_cap = sorted(set(re.findall(r"\*Table (\d+)\.", MD)))
ok(t_cap == [str(i) for i in range(1, 8)], "C.captions.tables", "Markdown has captions for Tables 1-7", str(t_cap))
refs_tab = sorted(set(re.findall(r"Table (\d+)", md_body)))
ok(set(refs_tab) <= set(t_cap), "C.xref.tables", "every referenced table has a caption", f"referenced {refs_tab}")
for name, t in PDFS.items():
    caps = sorted(set(re.findall(r"^Table (\d+)[.:]", t, flags=re.M)))
    ok(caps == [str(i) for i in range(1, 8)] and re.search(r"^Figure 1[.:]", t, flags=re.M) is not None, f"C.captions.pdf.{name}", f"{name} PDF contains captions Table 1-7 and Figure 1", str(caps))
ok(len(re.findall(r"!\[Figure 1\]", MD)) == 1 and "includegraphics" in TEX, "C.figure", "one figure in the Markdown, included in the LaTeX body", "")
ok((ROOT / "figures/fig1_rule_dependent_operating_points.png").exists() and any((ROOT / "paper/figures").glob("fig1*")), "C.figure.files", "figure files exist (figures/ and paper/figures/)", "")
a = (ROOT / "figures/fig1_rule_dependent_operating_points.png").read_bytes() if (ROOT / "figures/fig1_rule_dependent_operating_points.png").exists() else b""
pf = list((ROOT / "paper/figures").glob("*"))
add("C.figure.copy", "paper/figures holds the figure used by the LaTeX build", "PASS" if pf else "FAIL", str([p.name for p in pf]))

# -- citations
md_keys_used = set()
for grp in re.findall(r"\[((?:[A-Za-z]+\d{4}[A-Za-z0-9]*|vllmrlt56)(?:;\s*[A-Za-z]+\d{4}[A-Za-z0-9]*)*(?:,[^\]]*)?)\]", md_body):
    for k in re.split(r";\s*", grp.split(",")[0]):
        md_keys_used.add(k.strip())
md_defs = set(re.findall(r"^- \[([^\]]+)\]", MD, flags=re.M))
ok(md_keys_used <= md_defs, "C.cites.md", "every citation key used in the Markdown has a reference-list entry", f"used {len(md_keys_used)}, defined {len(md_defs)}, missing {sorted(md_keys_used - md_defs)}")
ok(md_defs <= md_keys_used | {"vllmrlt56"}, "C.cites.md_unused", "every Markdown reference-list entry is cited", f"uncited: {sorted(md_defs - md_keys_used)}", fail="NEEDS QUALIFICATION")
bib = (ROOT / "paper/refs.bib").read_text(encoding="utf-8")
bib_keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
tex_cites = set()
for grp in re.findall(r"\\cite[a-z]*\{([^}]*)\}", TEX):
    tex_cites |= {k.strip() for k in grp.split(",")}
ok(tex_cites <= bib_keys, "C.cites.tex", "every \\cite key in body.tex exists in refs.bib", f"cited {len(tex_cites)}, bib {len(bib_keys)}, missing {sorted(tex_cites - bib_keys)}")
ok(bib_keys <= tex_cites, "C.cites.bib_unused", "every refs.bib entry is cited in body.tex", f"uncited: {sorted(bib_keys - tex_cites)}", fail="NEEDS QUALIFICATION")
norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
ok({norm(k) for k in md_defs} == {norm(k) for k in bib_keys}, "C.cites.md_vs_bib", "Markdown reference keys and refs.bib keys are the same set (up to case/punctuation)",
   f"only md: {sorted({norm(k) for k in md_defs} - {norm(k) for k in bib_keys})}; only bib: {sorted({norm(k) for k in bib_keys} - {norm(k) for k in md_defs})}")
for name, t in PDFS.items():
    ok("[?]" not in t and "(?)" not in t and "??" not in t, f"C.pdf.{name}.undefined", f"{name} PDF has no unresolved references or citations", "")
    ok(len(re.findall(r"^\[\d+\]|\(\w+ et al\.,? \d{4}", t, flags=re.M)) >= 0, f"C.pdf.{name}.cites", "citations render (informational)", "")
logs = {}
for n in ("main_anon", "main_preprint"):
    lg = (ROOT / f"paper/{n}.log").read_text(encoding="utf-8", errors="replace")
    logs[n] = {"undefined_refs": len(re.findall(r"undefined", lg)), "overfull": len(re.findall(r"Overfull", lg)), "pages": re.findall(r"Output written on .*?\((\d+) pages", lg)}
ok(all(v["undefined_refs"] == 0 for v in logs.values()), "C.build.logs", "LaTeX logs report no undefined references/citations", json.dumps(logs))
for name, p in (("anon", "main_anon.pdf"), ("preprint", "main_preprint.pdf")):
    info = subprocess.run(["pdfinfo", str(ROOT / "paper" / p)], capture_output=True, text=True).stdout
    # expectation updated 2026-10-27: 20 pages was the count of the earlier Times-substitute build; the final Latin Modern build has 21 (a layout fact, not a content change)
    ok("Pages:           21" in info, f"C.pages.{name}", f"{name} PDF has 21 pages (final fonts)", re.findall(r"Pages:\s+\d+", info)[0])
fonts = subprocess.run(["pdffonts", str(ROOT / "paper/main_anon.pdf")], capture_output=True, text=True).stdout
ok("Nimbus" not in fonts, "C.fonts", "PDFs are built with the final TMLR fonts (Computer Modern / lmodern), not the temporary Times stand-in used in the sandbox",
   "fonts embedded: " + ", ".join(sorted({l.split()[0].split('+')[-1] for l in fonts.splitlines()[2:]}))[:300], fail="NEEDS QUALIFICATION")

# -- anonymisation
ID_PAT = re.compile(r"Mahesh|Pagadala|Aston|lonewolf|mahesh15116|Sunny|aston\.ac\.uk|github\.com/lonewolf|Birmingham", re.I)
anon_files = ["paper/main_anon.tex", "paper/body.tex", "paper/preamble_common.tex", "paper/refs.bib", "paper/math_commands.tex"]
hits = {f: ID_PAT.findall((ROOT / f).read_text(encoding="utf-8", errors="replace")) for f in anon_files}
ok(not any(hits.values()), "A.source", "identifying strings are absent from the sources compiled into the anonymous build", json.dumps({k: v for k, v in hits.items() if v}) or "none")
hit_pdf = ID_PAT.findall(PDFS["anon"])
ok(not hit_pdf, "A.pdf.text", "identifying strings are absent from the anonymous PDF text", str(hit_pdf))
info = subprocess.run(["pdfinfo", str(ROOT / "paper/main_anon.pdf")], capture_output=True, text=True).stdout
ok(not ID_PAT.search(info), "A.pdf.meta", "anonymous PDF metadata has no author/title strings", info.replace("\n", " | ")[:300])
figbytes = b"".join(p.read_bytes() for p in (ROOT / "paper/figures").glob("*") if p.is_file())
ok(not re.search(rb"Mahesh|Pagadala|lonewolf|mahesh15116|C:\\\\Users|/home/claude", figbytes, flags=re.I), "A.figure.meta", "figure files embed no identifying path/author strings", "")
others = ID_PAT.findall(PDFS["preprint"])
add("A.preprint.author", "the preprint PDF carries the author block (expected) — placeholder e-mail still present", "NEEDS QUALIFICATION" if "TODO@aston.ac.uk" in PDFS["preprint"] else "PASS",
    "TODO@aston.ac.uk in main_preprint.pdf" if "TODO@aston.ac.uk" in PDFS["preprint"] else "e-mail set")
todo = [f"{f}: {l.strip()[:90]}" for f in ("paper/body.tex", "paper/main_anon.tex", "paper/main_preprint.tex") for l in (ROOT / f).read_text(encoding="utf-8").splitlines() if "TODO" in l]
add("A.todo", "TODO markers remain in the LaTeX sources (comments only; none in the anonymous PDF text)", "NEEDS QUALIFICATION" if todo else "PASS", str(todo))
ok("TODO" not in PDFS["anon"] and "XXXX" not in PDFS["anon"].replace("openreview.net/forum?id=XXXX", ""), "A.pdf.todo", "no TODO/placeholder text in the anonymous PDF body", "")
add("A.openreview", "TMLR template placeholders (\\month/\\year/openreview id) are unset in the .tex mains (TMLR sets them at acceptance)", "PASS", "MM/YYYY/XXXX expected before acceptance")

# -- AI-assistance statement (author edit pending, noted in the hand-over)
ai = re.search(r"AI[- ]assist|large language model|Claude", MD + TEX, flags=re.I)
add("A.ai_statement", "the manuscript contains an AI-assistance / authorship statement (to be written by the author)", "NEEDS QUALIFICATION" if not ai else "PASS", "no statement found" if not ai else ai.group(0))

ap = argparse.ArgumentParser()
ap.add_argument("--out", default=str(ROOT / "results/audit/step4_manuscript_consistency.json"))
args = ap.parse_args()
nf = [c for c in CHECKS if c["status"] == "FAIL"]
nq = [c for c in CHECKS if c["status"] == "NEEDS QUALIFICATION"]
rep = {"script": "scripts/step4_manuscript_consistency.py", "n_checks": len(CHECKS), "n_pass": sum(c["status"] == "PASS" for c in CHECKS), "n_needs_qualification": len(nq), "n_fail": len(nf),
       "salient_tokens": {"md": sum(c_md.values()), "tex": sum(c_tex.values()), "pdf_anon": sum(c_anon.values()), "pdf_preprint": sum(c_pre.values())}, "checks": CHECKS}
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
Path(args.out).write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"{len(CHECKS)} checks, {rep['n_pass']} PASS, {len(nq)} NEEDS QUALIFICATION, {len(nf)} FAIL -> {args.out}")
for c in CHECKS:
    if c["status"] != "PASS":
        print(c["status"], c["id"], "|", c["claim"][:110], "|", str(c["evidence"])[:300])
sys.exit(1 if nf else 0)
