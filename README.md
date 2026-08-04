# SKNA Deployment Monitor

Version **1.0.0** — first public release.

A Streamlit raw-to-results interface for ECG-derived SKNA preprocessing, multi-event definition, adaptive thresholding, burst analysis, and sequential replay. It accepts LabChart-style TXT/ZIP recordings and processed CSV files while preserving additional recorded channels.

## Manuscript-consistent defaults

- ECG preprocessing: 500-Hz fourth-order zero-phase Butterworth high-pass, rectification, 1-s envelope, 100-Hz output
- Threshold baseline: up to 60 s immediately before event onset; onset excluded
- Adaptive threshold: max(GMM intersection, q95, median + 6×MAD)
- Replay: 30-s trailing window, 10-s updates, >5% burst occupancy, two consecutive qualifying updates

All controls remain visible for exploratory use. Exported JSON records the exact settings used.

## Installation

```bash
git clone https://github.com/Shahrokh-Imperial/skna-deployment-monitor.git
cd skna-deployment-monitor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest -q
```

## Run

```bash
bash app/run.sh
```

The launcher applies Streamlit upload and message limits of 2 GB.

## Input modes

### Bundled example events
Three synthetic event-centred examples are included for testing. They contain no experimental data and do not reproduce manuscript results.

### Local recording file — recommended for large data
Enter the full path to a TXT, ZIP, or CSV file. The parsed recording is cached in memory, so later widget changes do not re-read the file.

### Browser upload
Suitable for smaller files and configured up to 2 GB, subject to browser and server memory constraints.

## Typical workflow

1. Load the recording.
2. Confirm the detected UAP channel and inspect the immediate preview.
3. Auto-detect, upload, or manually define one or more events.
4. Select ECG channels and run preprocessing when the input is raw.
5. Review median SKNA and adaptive-threshold components.
6. Inspect burst occupancy and the persistence-gated replay trigger.
7. Export analysed signals, replay CSV, event table, and configuration JSON.

See `docs/USER_GUIDE.md`, `docs/RAW_INPUT_FORMAT.md`, and `docs/LARGE_FILE_GUIDE.md`.

## Important behaviour

- Thresholds are recalculated separately for each selected event from its own pre-event baseline.
- No INAP or post-INAP samples are used for threshold calibration.
- Automatic UAP detection is a suggested starting point and must be visually confirmed.
- The software does not modify source files.

## Repository map

- `app/`: Streamlit monitor and launcher
- `src/skna_framework/`: synchronized scientific engine used by the monitor
- `.streamlit/config.toml`: large-file server configuration
- `examples/`: synthetic event examples
- `tools/`: data-checking and packaging utilities
- `docs/`: input, event, large-file, and troubleshooting guides
- `tests/`: scientific-default and import checks

## Privacy and limitations

Do not upload confidential data to public deployments. This software is intended for research and deployment-oriented replay; it is not a medical device and has not been prospectively validated for clinical decisions.

## Citation and licence

See `CITATION.cff`. Released under the MIT License.
