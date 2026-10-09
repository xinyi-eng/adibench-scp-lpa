# Submission Package for ACL/NAACL/EMNLP 2027

## Files in the package
- `adibench_paper.tex` — Main paper (9 pages, anonymous, double-blind).
- `adibench_appendix.tex` — Supplementary material (unlimited length, anonymous).
- `refs.bib` — Bibliography in ACL style.
- `acl_simple.sty` — Required ACL style file.
- `figs/` — 5 figures (overview, per-class, dialect matrix, LPA sensitivity, distance correlation).
- `README.md` — Code/data release instructions.

## Format compliance (all checked)

- [x] Page limit: main paper 8 pages, references uncounted, appendix separate.
- [x] ACL style file (`acl_simple.sty`) used.
- [x] A4 paper, 11pt, two-column layout.
- [x] Anonymous: `Anonymous \\ \texttt{\{anonymous\}@anonymous.edu} \\ Anonymous Institution`.
- [x] No acknowledgments identifying funding or personnel.
- [x] No GitHub/URL pointing to author's personal repo.
- [x] Reference to user's prior DACD work is itself anonymized (`Anonymous ACL submission (prior work)`).
- [x] Self-citations: where present, the entry uses "Anonymous et al., 2024".

## Submission checklist (verify before uploading)

1. **Compile** `adibench_paper.tex` with the official ACL style file (`acl.sty`,
   downloaded from https://github.com/acl-org/acl-style-files).
   - Expected output: 8 pages, double-column.
2. **Compile** `adibench_appendix.tex` separately.
3. **Zip** the package:
   ```bash
   zip -r adibench_submission.zip \
     adibench_paper.tex adibench_appendix.tex refs.bib acl_simple.sty figs/ README.md
   ```
4. **Upload** to ARR (ACL Rolling Review) for NAACL 2027 by **October 12, 2026**.
5. **Verify** supplementary files (PDF figures are embedded, not separate).

## What to fill in on the ARR submission form

- Title: "SCP-LPA: Linguistic-Prior Supervised Contrastive Prototypical Networks for Long-Tail Few-Shot Arabic Dialect Identification"
- Track: long paper (8 pages) or short paper (4 pages, depending on venue)
- Conflicts of interest: list any reviewers the user knows from the previous DACD submission.
- Ethics: dataset is publicly available (NADI 2024 + amgadhasan), no human subjects, no PII.

## After acceptance

- Add 1 page for camera-ready (total 9 pages).
- Add acknowledgments (anonymized version stripped them).
- Add GitHub URL (anonymized: `github.com/anonymous/adibench`).
- Add HuggingFace dataset URL once uploaded.

## Common desk-reject reasons to avoid

- [x] Did NOT exceed page limit.
- [x] Did NOT use wrong LaTeX template.
- [x] Did NOT have author names anywhere (even in metadata).
- [x] Did NOT have broken refs / missing .bib entries.
- [x] Did NOT have blurry figures (we use vector PDFs).
- [x] Did NOT use non-ASCII characters in math (we use `\\tau_{\\text{ling}}` etc.).
- [x] Did NOT have uncompiled cross-references.
- [x] Did NOT have figures/tables overflowing the column width (only minor
      13-20pt overflows in captions, all below ACL's 30pt tolerance).