#!/usr/bin/env python3
from pathlib import Path
import sys, pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from skna_framework.core import load_event_folder, detect_time_column, detect_median_skna_column
folder=Path(sys.argv[1] if len(sys.argv)>1 else ROOT/'examples')
events,summary=load_event_folder(folder)
if not events: raise SystemExit('ERROR: no eventN_signals.csv found')
print(f'Found events: {events}')
for e in events:
 p=folder/f'event{e}_signals.csv'; df=pd.read_csv(p); t=detect_time_column(df); s=detect_median_skna_column(df)
 print(f'  event {e}: {p.name} | rows={len(df):,} | time={t} | median_SKNA={s}')
if summary.empty: print('WARNING: event_summary.csv not found; thresholds must be recalculated in GUI.')
else: print(f'event_summary.csv: {len(summary)} row(s), columns={len(summary.columns)}')
print('OK')
