# Input data schema

## Raw LabChart-style TXT
Required: `Interval=`, `ChannelTitle=` and tab-delimited numeric data. At least one ECG channel is required for preprocessing. UAP is required only for automatic INAP detection.

## Processed CSV
Required: a time column plus `skna_med` or another identifiable median SKNA column. Optional: `compact_skna_1..3`, filtered ECG, UAP and any physiological channels.

## Event CSV
Any CSV with one start-like and one end-like column containing seconds from recording start. Multiple rows represent multiple events.


The default threshold baseline is up to 60 s immediately before event onset, excluding the onset sample.
