# paper_revision/

Revised manuscript and audit trail. Nothing here modifies the original paper (`paper/`), the production code or any experiment output.

| File | Content |
|---|---|
| `technical_audit.md` | Audit of `paper/paper_final.tex` before revision (Part 1) |
| `methodology_corrections.md` | What was corrected and how (Parts 2-10) |
| `dataset_split_and_site_mapping.csv` | All 100 UA-DETRAC videos: site, partition, frames, annotations, uncertainty |
| `revised_paper.tex` | Revised source (IEEEtran); figures in `figures/` |
| `revised_paper.pdf` | **Layout preview with a stand-in class, not an IEEE build** |
| `revised_paper_overleaf.zip` | Source + figures for the real IEEEtran build |
| `figure_sources/` | Scripts that regenerate every figure from data (`figures/data/*.csv`) |
| `updated_tables/` | Significance tests, counting by lane/error type, classification, objective verification |
| `supplementary/` | Site mapping, contact sheets, speed and overlay figures removed from the main text |
| `change_log.md` | Section-by-section changes |
| `final_verification_report.md` | Build checks, claim-to-evidence table, limitations |
| `scripts/` | `revision_analysis.py`, `build_revised_tex.py`, `verify_revision.py` |

Reproduce: `python paper_revision/scripts/revision_analysis.py all`, `python paper_revision/figure_sources/make_figures_revised.py`, `python paper_revision/figure_sources/make_grid_figures.py`, `python paper_revision/scripts/build_revised_tex.py`, compile, then `python paper_revision/scripts/verify_revision.py`.
