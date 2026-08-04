from __future__ import annotations
from pathlib import Path
import io, json, zipfile, tempfile, re
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from scipy.signal import butter, sosfiltfilt, resample_poly
from scipy.ndimage import uniform_filter1d

SKNA_NAME_HINTS=("skna_med","median_skna_uv","median_skna","skna median","median skna")
TIME_HINTS=("time_relative_s","time_s","time [s]","time")

def _pick_column(columns,hints,exclude=()):
    for h in hints:
        for c in columns:
            cl=str(c).lower()
            if h in cl and not any(e in cl for e in exclude): return c
    return None

def detect_time_column(df):
    c=_pick_column(df.columns,TIME_HINTS)
    if c is None: raise ValueError('Could not identify a time column.')
    return c

def detect_median_skna_column(df):
    c=_pick_column(df.columns,SKNA_NAME_HINTS,exclude=('threshold',))
    if c is None:
        cs=[x for x in df.columns if 'skna' in str(x).lower() and 'threshold' not in str(x).lower()]
        if cs: c=cs[0]
    if c is None: raise ValueError('Could not identify a median SKNA column.')
    return c

def detect_uap_column(df):
    for h in ('upper airway pressure','upper-airway pressure','uap'):
        c=_pick_column(df.columns,(h,))
        if c is not None: return c
    return None

def detect_ecg_columns(df):
    # Prefer columns explicitly named ECG and exclude derived/filter channels.
    cs=[c for c in df.columns if 'ecg' in str(c).lower() and not any(x in str(c).lower() for x in ('filt','filter','skna','rect','clip'))]
    return cs[:3]

def detect_recorded_channels(df):
    blocked=('skna','compact_','filt','filtered','rect','clip','burst','threshold','time_s','time_relative')
    return [c for c in df.columns if not any(b in str(c).lower() for b in blocked) and pd.api.types.is_numeric_dtype(df[c])]

def detect_processed_channels(df):
    cols=list(df.columns)
    return {'filtered_ecg':[c for c in cols if 'filt' in str(c).lower() and 'ecg' in str(c).lower()],
            'channel_skna':[c for c in cols if 'compact_skna' in str(c).lower() or ('skna' in str(c).lower() and 'median' not in str(c).lower() and 'skna_med' not in str(c).lower() and 'threshold' not in str(c).lower())]}

def infer_skna_scale_to_uv(series,column_name='',reference_median_uv=None):
    name=str(column_name).lower()
    if 'uv' in name or 'µv' in name: return 1.0,'Column name explicitly indicates µV.'
    x=pd.to_numeric(series,errors='coerce').to_numpy(float); x=x[np.isfinite(x)]
    med=float(np.nanmedian(x)) if x.size else np.nan
    if reference_median_uv is not None and np.isfinite(reference_median_uv) and np.isfinite(med) and med!=0:
        cand=[(1.0,'values treated as µV'),(1000.0,'values converted mV→µV'),(1e6,'values converted V→µV')]
        s,n=min(cand,key=lambda z:abs(med*z[0]-reference_median_uv)); return s,'Scale selected by baseline-reference check: '+n+'.'
    if 'skna_med' in name or 'compact_skna' in name: return 1000.0,'Processed cohort-style SKNA column assumed mV; converted ×1000 to µV.'
    return 1.0,'No unit conversion inferred; values treated as µV.'

def mad(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return float(np.median(np.abs(x-np.median(x)))) if x.size else np.nan

def weighted_gmm_intersection(x,random_state=0):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if x.size<20:return np.nan
    gm=GaussianMixture(n_components=2,random_state=random_state,n_init=10).fit(x.reshape(-1,1))
    means=gm.means_.ravel(); order=np.argsort(means); means=means[order]
    vars_=gm.covariances_.reshape(-1)[order]; weights=gm.weights_.ravel()[order]
    m1,m2=means; v1,v2=vars_; w1,w2=weights
    a=1/(2*v1)-1/(2*v2); b=m2/v2-m1/v1
    c=m1*m1/(2*v1)-m2*m2/(2*v2)+np.log((w2*np.sqrt(v1))/(w1*np.sqrt(v2)))
    roots=np.roots([a,b,c]) if abs(a)>1e-12 else np.array([-c/b])
    between=[float(r.real) for r in roots if abs(r.imag)<1e-8 and m1<r.real<m2]
    return between[0] if between else float((m1+m2)/2)

def adaptive_threshold(baseline_uv,percentile=95,k_mad=6.0):
    x=np.asarray(baseline_uv,float); x=x[np.isfinite(x)]
    if x.size<20: raise ValueError('At least 20 finite baseline samples are required.')
    g=weighted_gmm_intersection(x); pct=float(np.percentile(x,percentile)); med=float(np.median(x)); m=mad(x); robust=med+k_mad*m
    vals={'gmm_threshold_uV':g,'q95_threshold_uV':pct,'median_plus_6mad_uV':robust}
    finite={k:v for k,v in vals.items() if np.isfinite(v)}
    vals['adaptive_threshold_uV']=max(finite.values()); vals['selected_source']=max(finite,key=finite.get)
    vals.update({'baseline_median_uV':med,'baseline_mad_uV':m,'percentile':percentile,'k_mad':k_mad})
    return vals

def persistence_gate(candidate,consecutive=2):
    c=np.asarray(candidate,bool); out=np.zeros(c.size,bool); run=0
    for i,v in enumerate(c): run=run+1 if v else 0; out[i]=run>=consecutive
    return out

def sequential_replay(time_relative_s,skna_uv,point_threshold_uv,window_s=30.0,hop_s=10.0,occupancy_threshold_pct=5.0,persistence_windows=2):
    t=np.asarray(time_relative_s,float); y=np.asarray(skna_uv,float); g=np.isfinite(t)&np.isfinite(y); t=t[g]; y=y[g]
    o=np.argsort(t); t=t[o]; y=y[o]
    if not t.size: raise ValueError('No finite time/SKNA samples.')
    first=np.ceil((t.min()+window_s)/hop_s)*hop_s; ends=np.arange(first,t.max()+1e-9,hop_s)
    occ=[]
    for end in ends:
        m=(t>end-window_s)&(t<=end); occ.append(100*np.mean(y[m]>point_threshold_uv) if m.any() else np.nan)
    occ=np.asarray(occ); cand=occ>occupancy_threshold_pct; trig=persistence_gate(cand,persistence_windows)
    return pd.DataFrame({'window_end_relative_s':ends,'burst_occupancy_pct':occ,'candidate_on':cand.astype(int),'trigger_on':trig.astype(int)})

def replay_event_metrics(replay,onset_s=0,offset_s=75,hop_s=10):
    t=replay.window_end_relative_s.to_numpy(float); tr=replay.trigger_on.to_numpy(bool); ev=(t>=onset_s)&(t<=offset_s); idx=np.flatnonzero(tr&ev)
    return {'trigger_success':bool(idx.size),'latency_s':float(t[idx[0]]-onset_s) if idx.size else np.nan,
            'trigger_duration_in_event_s':float(np.sum(tr&ev)*hop_s),'mean_event_occupancy_pct':float(replay.loc[ev,'burst_occupancy_pct'].mean()) if ev.any() else np.nan,
            'pre_event_trigger_updates':int(np.sum(tr&(t<onset_s))),'post_event_trigger_updates':int(np.sum(tr&(t>offset_s)))}

# ---------- LabChart TXT/ZIP ----------
def _decimal_float(s):
    try:return float(str(s).replace(',','.'))
    except:return np.nan

def inspect_labchart_header(path_or_bytes):
    if isinstance(path_or_bytes,(str,Path)):
        p=Path(path_or_bytes)
        if p.suffix.lower()=='.zip':
            with zipfile.ZipFile(p) as z:
                txt=[n for n in z.namelist() if n.lower().endswith('.txt')]
                if not txt: raise ValueError('ZIP contains no .txt file.')
                data=z.read(txt[0])
        else:data=p.read_bytes()
    else:data=path_or_bytes
    text=data[:200000].decode('utf-8',errors='replace'); lines=text.splitlines()
    meta={}; header_lines=0
    for i,line in enumerate(lines):
        if line.startswith('Interval='):
            meta['interval_s']=_decimal_float(line.split('\t')[1].split()[0])
        elif line.startswith('ChannelTitle='):
            meta['channels']=line.split('\t')[1:]
        elif line.startswith('UnitName='):
            meta['units']=line.split('\t')[1:]
        if re.match(r'^[-+0-9]+[,.][0-9]+\t',line): header_lines=i; break
    meta['header_lines']=header_lines
    return meta

def _open_labchart_stream(path_or_bytes):
    """Return (binary stream, cleanup callable) for TXT/ZIP/bytes without expanding a ZIP in memory."""
    if isinstance(path_or_bytes,(str,Path)):
        p=Path(path_or_bytes)
        if p.suffix.lower()=='.zip':
            z=zipfile.ZipFile(p)
            txt=[n for n in z.namelist() if n.lower().endswith('.txt')]
            if not txt:
                z.close(); raise ValueError('ZIP contains no .txt file.')
            fh=z.open(txt[0],'r')
            return fh, lambda: (fh.close(), z.close())
        fh=p.open('rb')
        return fh, fh.close
    return io.BytesIO(path_or_bytes), lambda: None


def _repair_bad_labchart_row(fields, expected_fields):
    """Repair occasional LabChart rows with surplus tab-delimited fields.

    Most observed malformed rows are caused by one or more trailing tab characters.
    We first remove only surplus empty fields.  If a row still has too many fields,
    it is truncated at the expected width rather than aborting the whole recording.
    A short row is padded with empty values so channel alignment remains explicit.
    """
    fields=list(fields)
    while len(fields)>expected_fields and fields and fields[-1]=='':
        fields.pop()
    if len(fields)>expected_fields:
        # Prefer removing surplus empty tokens from right to left before truncation.
        extra=len(fields)-expected_fields
        for i in range(len(fields)-1,-1,-1):
            if extra<=0: break
            if fields[i]=='':
                fields.pop(i); extra-=1
        if len(fields)>expected_fields:
            fields=fields[:expected_fields]
    elif len(fields)<expected_fields:
        fields.extend(['']*(expected_fields-len(fields)))
    return fields


def load_labchart_txt(path_or_bytes, max_rows=None):
    """Load LabChart text exported as TXT or ZIP.

    Fast path uses pandas' C parser.  If LabChart contains an occasional irregular
    row (for example an extra trailing tab), loading automatically retries with the
    Python parser and repairs only malformed rows instead of failing the recording.
    ZIP members are streamed; they are not fully decompressed into RAM first.
    """
    # Read only a small header prefix for metadata.
    stream, cleanup=_open_labchart_stream(path_or_bytes)
    try:
        prefix=stream.read(200000)
    finally:
        cleanup()
    meta=inspect_labchart_header(prefix)
    channels=meta.get('channels',[]); skip=meta.get('header_lines',0)
    if not channels:
        raise ValueError('Could not find ChannelTitle= in the LabChart header.')
    names=['_time_abs']+channels
    expected=len(names)

    def _read(engine='c'):
        fh, close=_open_labchart_stream(path_or_bytes)
        try:
            kwargs=dict(sep='\t',skiprows=skip,header=None,names=names,decimal=',',
                        na_values=['NaN',''],nrows=max_rows,engine=engine)
            if engine=='python':
                kwargs['on_bad_lines']=lambda row: _repair_bad_labchart_row(row,expected)
            return pd.read_csv(fh,**kwargs)
        finally:
            close()

    try:
        df=_read('c')
        meta['parser_mode']='fast_c'
        meta['repaired_irregular_rows']=False
    except pd.errors.ParserError as exc:
        df=_read('python')
        meta['parser_mode']='robust_python_fallback'
        meta['repaired_irregular_rows']=True
        meta['parser_warning']=('The LabChart file contained at least one row with an irregular tab count. '
                                'The loader repaired surplus/trailing fields and continued. Original parser error: '+str(exc))

    df=df.dropna(how='all')
    t=pd.to_numeric(df['_time_abs'],errors='coerce').to_numpy(float)
    if np.isfinite(t).any():
        t=t-t[np.flatnonzero(np.isfinite(t))[0]]
    else:
        dt=float(meta.get('interval_s',np.nan)); t=np.arange(len(df))*dt
    df.insert(0,'time_s',t); df=df.drop(columns=['_time_abs'])
    for c in channels: df[c]=pd.to_numeric(df[c],errors='coerce')
    meta['source_fs_hz']=1.0/meta['interval_s'] if meta.get('interval_s') else np.nan
    meta['n_rows_loaded']=int(len(df)); meta['n_channels_loaded']=int(len(channels))
    return df,meta

def preprocess_raw_recording(raw_df,fs_hz,ecg_columns=None,fs_out_hz=100.0,hp_hz=500.0,envelope_s=1.0,
                              ecg_display_low_hz=0.5,ecg_display_high_hz=45.0,mad_clip_k=None):
    if ecg_columns is None: ecg_columns=detect_ecg_columns(raw_df)
    if len(ecg_columns)<1: raise ValueError('No ECG channels detected. Select ECG columns manually.')
    if hp_hz>=fs_hz/2: raise ValueError(f'High-pass cutoff {hp_hz} Hz must be below Nyquist ({fs_hz/2:g} Hz).')
    # output time grid via integer decimation when possible; otherwise polyphase resampling
    q=max(1,int(round(fs_hz/fs_out_hz))); actual_out=fs_hz/q
    sos_hp=butter(4,hp_hz,btype='highpass',fs=fs_hz,output='sos')
    sos_ecg=butter(4,[ecg_display_low_hz,ecg_display_high_hz],btype='bandpass',fs=fs_hz,output='sos') if ecg_display_high_hz<fs_hz/2 else None
    nwin=max(1,int(round(envelope_s*fs_hz)))
    out={}
    t=raw_df['time_s'].to_numpy(float); out['time_s']=t[::q]
    # preserve recorded channels at output rate
    for c in raw_df.columns:
        if c=='time_s':continue
        out[c]=pd.to_numeric(raw_df[c],errors='coerce').to_numpy(float)[::q]
    sknas=[]
    for i,c in enumerate(ecg_columns,1):
        x=pd.to_numeric(raw_df[c],errors='coerce').to_numpy(float)
        # fill isolated NaN for filtering
        s=pd.Series(x).interpolate(limit_direction='both').to_numpy(float)
        hp=sosfiltfilt(sos_hp,s)
        rect=np.abs(hp)
        if mad_clip_k is not None and mad_clip_k>0:
            med=np.median(rect); m=np.median(np.abs(rect-med)); cap=med+mad_clip_k*m; rect=np.minimum(rect,cap)
        env=uniform_filter1d(rect,size=nwin,mode='nearest')
        ecg=sosfiltfilt(sos_ecg,s) if sos_ecg is not None else s
        out[f'ecg_filt__{c}']=ecg[::q]
        out[f'compact_skna_{i}']=env[::q]
        sknas.append(env[::q])
    out['skna_med']=np.nanmedian(np.vstack(sknas),axis=0)
    return pd.DataFrame(out),{'source_fs_hz':fs_hz,'fs_out_hz':actual_out,'ecg_sources':list(ecg_columns),'hp_hz':hp_hz,'envelope_s':envelope_s,
                              'mad_clip_k':mad_clip_k,'note':'SKNA values remain in the ECG input amplitude unit; LabChart ECG channels in the supplied format are mV.'}

def detect_uap_events(time_s,uap,threshold=None,min_duration_s=5.0,merge_gap_s=2.0):
    t=np.asarray(time_s,float); y=np.asarray(uap,float); good=np.isfinite(t)&np.isfinite(y); t=t[good]; y=y[good]
    if t.size<2:return pd.DataFrame(columns=['event','start_s','end_s','duration_s'])
    if threshold is None:
        # For typical UAP recordings baseline is near 0 mbar and obstruction is strongly negative.
        base=np.nanmedian(y[:max(10,int(.1*len(y)))]); m=mad(y[:max(10,int(.1*len(y)))])
        threshold=min(-10.0,base-6*m) if np.nanmin(y)<-10 else base-6*m
    active=y<threshold
    idx=np.flatnonzero(np.diff(np.r_[False,active,False].astype(int)))
    seg=[]
    for a,b in idx.reshape(-1,2):
        s=t[a]; e=t[b-1]
        if e-s>=min_duration_s: seg.append([s,e])
    merged=[]
    for s,e in seg:
        if merged and s-merged[-1][1]<=merge_gap_s: merged[-1][1]=e
        else: merged.append([s,e])
    return pd.DataFrame([{'event':i+1,'start_s':s,'end_s':e,'duration_s':e-s} for i,(s,e) in enumerate(merged)]),float(threshold)

def load_any_recording(path):
    p=Path(path); suf=p.suffix.lower()
    if suf in ('.txt','.zip'):
        # zip may contain CSV or LabChart TXT
        if suf=='.zip':
            with zipfile.ZipFile(p) as z:
                csvs=[n for n in z.namelist() if n.lower().endswith('.csv')]
                txts=[n for n in z.namelist() if n.lower().endswith('.txt')]
                if txts:
                    df,meta=load_labchart_txt(p); return df,meta,'raw'
                if csvs:
                    df=pd.read_csv(io.BytesIO(z.read(csvs[0]))); return df,{},('processed' if any('skna' in str(c).lower() for c in df.columns) else 'raw_csv')
        df,meta=load_labchart_txt(p); return df,meta,'raw'
    df=pd.read_csv(p)
    return df,{},('processed' if any('skna' in str(c).lower() for c in df.columns) else 'raw_csv')


def load_event_folder(folder):
    folder=Path(folder); summary_path=folder/'event_summary.csv'; summary=pd.read_csv(summary_path) if summary_path.exists() else pd.DataFrame(); events=[]
    for p in sorted(folder.glob('event*_signals.csv')):
        d=''.join(ch for ch in p.stem if ch.isdigit()); events.append(int(d)) if d else None
    return sorted(set(events)),summary


def suggest_uap_threshold(time_s,uap):
    """Suggest a data-driven UAP threshold for negative-pressure events.

    The estimate is intentionally descriptive rather than a physiological gold standard:
    baseline is represented by the recording-wide median/IQR, while the candidate threshold
    is placed between the baseline centre and the lower tail of the UAP distribution.
    """
    y=np.asarray(uap,float); y=y[np.isfinite(y)]
    if y.size<20:
        raise ValueError('At least 20 finite UAP samples are required for automatic threshold suggestion.')
    q01,q05,q10,q25,q50,q75,q95,q99=np.percentile(y,[1,5,10,25,50,75,95,99])
    # Negative-pressure events occupy the lower tail. Estimate a baseline distribution
    # from the upper 75% of samples so a long negative event does not dominate it.
    baseline_pool=y[y>=q25]
    if baseline_pool.size<20: baseline_pool=y
    b25,b50,b75=np.percentile(baseline_pool,[25,50,75])
    biqr=max(b75-b25,1e-12)
    tail=q05
    excursion=b50-tail
    if excursion<=0:
        suggested=b50-1.5*biqr
    else:
        suggested=(b50+tail)/2.0
        # Keep the proposal clearly below baseline but within the observed lower tail range.
        suggested=min(suggested,b50-max(1.5*biqr,0.05*excursion))
        suggested=max(suggested,q01)
    return {
        'baseline_median':float(b50),'baseline_q25':float(b25),'baseline_q75':float(b75),
        'q01':float(q01),'q05':float(q05),'q10':float(q10),'q95':float(q95),'q99':float(q99),
        'suggested_threshold':float(suggested),'iqr':float(biqr)
    }


def diagnose_uap_detection(time_s,uap,threshold,min_duration_s=5.0):
    """Explain why a threshold produced no accepted UAP events."""
    t=np.asarray(time_s,float); y=np.asarray(uap,float)
    g=np.isfinite(t)&np.isfinite(y); t=t[g]; y=y[g]
    if t.size<2:
        return {'status':'insufficient_data','message':'Too few finite UAP samples to detect events.'}
    below=y<threshold
    frac=float(np.mean(below))
    if not np.any(below):
        return {'status':'always_above','fraction_below':frac,
                'message':f'UAP never falls below the current threshold ({threshold:.3g}). Raise the threshold toward the baseline or use manual/event-CSV timing.'}
    if np.all(below):
        return {'status':'always_below','fraction_below':frac,
                'message':f'UAP is below the current threshold for the entire recording. Lower the threshold (make it more negative) so baseline samples are excluded.'}
    idx=np.flatnonzero(np.diff(np.r_[False,below,False].astype(int))).reshape(-1,2)
    durs=[]
    for a,b in idx:
        durs.append(float(t[b-1]-t[a]))
    maxdur=max(durs) if durs else 0.0
    if maxdur<min_duration_s:
        return {'status':'segments_too_short','fraction_below':frac,'max_segment_s':maxdur,
                'message':f'UAP crosses the threshold, but the longest below-threshold segment is only {maxdur:.2f} s (< minimum event duration {min_duration_s:.2f} s). Reduce the minimum duration or adjust the threshold.'}
    return {'status':'crossings_present','fraction_below':frac,'max_segment_s':maxdur,
            'message':'Threshold crossings are present; if no events are accepted, review merge/minimum-duration settings.'}
