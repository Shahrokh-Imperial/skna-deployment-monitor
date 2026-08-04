#!/usr/bin/env python3
"""Split one processed multi-event recording into eventN_signals.csv files.

Input CSV must already contain a time column and processed median SKNA. Event CSV must contain
one row per event with columns event, onset_s and offset_s. This utility does not perform raw ECG→SKNA extraction.
"""
import argparse
from pathlib import Path
import pandas as pd, numpy as np, sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from skna_framework.core import detect_time_column
p=argparse.ArgumentParser(); p.add_argument('--signals',required=True); p.add_argument('--events',required=True); p.add_argument('--out',required=True); p.add_argument('--pre',type=float,default=120); p.add_argument('--post',type=float,default=150); a=p.parse_args()
df=pd.read_csv(a.signals); tc=detect_time_column(df); ev=pd.read_csv(a.events); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
for _,r in ev.iterrows():
 e=int(r['event']); on=float(r['onset_s']); off=float(r['offset_s']); m=(df[tc]>=on-a.pre)&(df[tc]<=off+a.post)
 x=df.loc[m].copy(); x['time_relative_s']=pd.to_numeric(x[tc],errors='coerce')-on; x.to_csv(out/f'event{e}_signals.csv',index=False)
print(f'Wrote {len(ev)} event signal files to {out}')
