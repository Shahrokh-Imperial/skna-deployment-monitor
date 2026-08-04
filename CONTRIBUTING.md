# Contributing

1. Open an issue describing the proposed change or bug.
2. Create a branch from `main`.
3. Add or update tests for scientific logic.
4. Run `pytest -q` and `python -m compileall src app scripts` where applicable.
5. Submit a pull request explaining whether the change affects numerical results or default parameters.

Changes to filtering, threshold estimation, burst occupancy, event timing, or replay logic must include a reproducible test and a changelog entry.
