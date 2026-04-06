# EconMScThesis

This repository now keeps the active thesis code in a simpler layout.

## Structure

- `python/stata_inputs/`
  - Python scripts that build Stata-ready input datasets.
- `python/stata_output/`
  - Python scripts that analyze or visualize Stata output.
- `python/descriptive/`
  - Descriptive and standalone thesis figures/tables.
- `python/shared/`
  - Reusable shared Python helpers.
- `preprocessing/`
  - Shared preprocessing package used by the Python scripts.
- `stata/`
  - Active Stata `.do` files.
- `master data files/`
  - Core tracked input workbooks used by the project.

## Notes

- Generated outputs are intentionally ignored.
- Older legacy scripts have been removed from the main repo view.
- A pre-cleanup snapshot is preserved on branch `archive/pre-cleanup-2026-04-06`.
