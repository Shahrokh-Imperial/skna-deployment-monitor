from pathlib import Path
import sys, json, tempfile, zipfile, io, re
import numpy as np, pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
APP_DIR=Path(__file__).resolve().parent; ROOT=APP_DIR.parent; sys.path.insert(0,str(ROOT/'src'))
from skna_framework.core import *

# -------------------------------------------------------------------------
# Performance cache
# -------------------------------------------------------------------------
# Streamlit reruns the whole script whenever a widget changes. Large LabChart
# TXT/ZIP files must therefore NOT be reparsed on every click. The functions
# below cache loaded recordings by a cheap source signature (path/size/mtime
# for local files; upload name/size for browser uploads). The parsed DataFrame
# stays in server memory and is reused on subsequent widget interactions.

@st.cache_resource(show_spinner=False)
def _cached_load_local_recording(path_str: str, size_bytes: int, mtime_ns: int):
    return load_any_recording(Path(path_str))

def _upload_signature(uploaded_file):
    return (
        str(getattr(uploaded_file, "name", "")),
        int(getattr(uploaded_file, "size", 0) or 0),
        str(getattr(uploaded_file, "type", "") or ""),
    )

def _get_cached_uploaded_recording(uploaded_file):
    sig = _upload_signature(uploaded_file)
    cache = st.session_state.get("_skna_uploaded_recording_cache")
    if cache is not None and cache.get("signature") == sig:
        return cache["df"], dict(cache["meta"]), cache["source_kind"], cache["temp_path"], True

    suffix = Path(uploaded_file.name).suffix
    # Include size in the filename so a same-named replacement does not
    # accidentally reuse the previous temporary file.
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(uploaded_file.name).stem)
    tmp = Path(tempfile.gettempdir()) / f"skna_upload_{safe_name}_{sig[1]}{suffix}"

    uploaded_file.seek(0)
    with tmp.open("wb") as fh:
        while True:
            chunk = uploaded_file.read(16 * 1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    uploaded_file.seek(0)

    df, meta, source_kind = load_any_recording(tmp)
    st.session_state["_skna_uploaded_recording_cache"] = {
        "signature": sig,
        "df": df,
        "meta": dict(meta),
        "source_kind": source_kind,
        "temp_path": str(tmp),
    }
    # A new source invalidates preprocessing from the previous recording.
    for key in (
        "processed_raw_cache",
        "processed_raw_meta",
        "processed_raw_key",
    ):
        st.session_state.pop(key, None)
    return df, dict(meta), source_kind, str(tmp), False

def _clear_recording_caches():
    st.session_state.pop("_skna_uploaded_recording_cache", None)
    st.session_state.pop("_preview_array_cache", None)
    st.session_state.pop("_local_loaded_once", None)
    for key in ("processed_raw_cache", "processed_raw_meta", "processed_raw_key"):
        st.session_state.pop(key, None)
    _cached_load_local_recording.clear()


st.set_page_config(page_title='SKNA Deployment Monitor',page_icon='⚡',layout='wide')
st.title('SKNA Deployment / Replay Monitor')
st.caption('Raw-to-results deployment interface for ECG-derived SKNA. Load a LabChart-style TXT/ZIP or an already processed CSV; extra channels are preserved but the analysis automatically focuses on ECG and upper-airway pressure.')

mode=st.sidebar.radio('Input mode',['Bundled example events','Local recording file','Upload recording'])
if st.sidebar.button('Clear loaded recording cache', help='Use this only when you intentionally want to reload the same file from disk/browser.'):
    _clear_recording_caches()
    st.rerun()
processed=None; raw=None; meta={}; source_kind=''; source_name=''; events=None; bundled_mode=False; bundled_summary_row=None

if mode=='Bundled example events':
    bundled_mode=True
    folder=ROOT/'examples'; evs,_=load_event_folder(folder) if 'load_event_folder' in globals() else ([],pd.DataFrame())
    ev=st.sidebar.selectbox('Bundled event',evs,key='bundled_event_selector')
    p=folder/f'event{ev}_signals.csv'; processed=pd.read_csv(p); source_kind='processed'; source_name=str(p)
    tc=detect_time_column(processed); processed=processed.rename(columns={tc:'time_s'}) if tc!='time_s' else processed
    sm=pd.read_csv(folder/'event_summary.csv') if (folder/'event_summary.csv').exists() else pd.DataFrame()
    row=sm.loc[sm['event'].astype(str)==str(ev)].iloc[0] if (not sm.empty and 'event' in sm.columns and (sm['event'].astype(str)==str(ev)).any()) else pd.Series(dtype=object)
    bundled_summary_row=row.copy()
    dur=float(row.get('event_duration_s',row.get('duration_s',75.0))); events=pd.DataFrame([{'event':int(ev),'start_s':0.0,'end_s':dur,'duration_s':dur}])
elif mode=='Local recording file':
    path=Path(st.sidebar.text_input('Recording TXT / ZIP / CSV path','')).expanduser()
    if not str(path) or not path.exists(): st.info('Enter a valid recording path in the sidebar.'); st.stop()
    source_name=str(path)
    try:
        stat=path.stat()
        with st.spinner('Loading recording…' if not st.session_state.get('_local_loaded_once') else 'Using cached recording…'):
            df,meta,source_kind=_cached_load_local_recording(str(path.resolve()),int(stat.st_size),int(stat.st_mtime_ns))
        st.session_state['_local_loaded_once']=True
    except Exception as e: st.error(f'Could not load recording: {e}'); st.stop()
    if source_kind=='processed': processed=df
    else: raw=df
else:
    up=st.sidebar.file_uploader(
        'Recording file',type=['txt','zip','csv'],
        help='Maximum upload size is configured to 2 GB. The recording is parsed only once and then kept in server memory while you change event or analysis controls.'
    )
    if up is None:
        st.info('Upload a recording TXT, ZIP, or CSV.'); st.stop()
    source_name=up.name
    try:
        with st.spinner('Loading and parsing recording once…'):
            df,meta,source_kind,tmp_path,from_cache=_get_cached_uploaded_recording(up)
    except Exception as e:
        st.error(f'Could not load recording: {e}'); st.stop()
    if from_cache:
        st.sidebar.caption('✓ Recording already loaded — using in-memory cache')
    else:
        st.sidebar.success('Recording loaded and cached. Further widget changes will not re-read the large file.')
    if source_kind=='processed': processed=df
    else: raw=df

# ---------- event preview BEFORE preprocessing ----------
preview_df = raw if raw is not None else processed
if preview_df is None:
    st.error('No recording data are available.'); st.stop()

preview_key=(source_name, source_kind, len(preview_df), tuple(map(str, preview_df.columns)))
pcache=st.session_state.get('_preview_array_cache')
if pcache is not None and pcache.get('key')==preview_key:
    preview_t=pcache['time']
    uap_preview_col=pcache['uap_col']
    uv_cached=pcache.get('uap')
else:
    if 'time_s' in preview_df.columns:
        preview_t = pd.to_numeric(preview_df['time_s'],errors='coerce').to_numpy(float)
    else:
        ptc=detect_time_column(preview_df); preview_t=pd.to_numeric(preview_df[ptc],errors='coerce').to_numpy(float)
    if (not bundled_mode) and np.isfinite(preview_t).any():
        first_t=preview_t[np.flatnonzero(np.isfinite(preview_t))[0]]; preview_t=preview_t-first_t
    uap_preview_col=detect_uap_column(preview_df)
    uv_cached=(pd.to_numeric(preview_df[uap_preview_col],errors='coerce').to_numpy(float) if uap_preview_col else None)
    st.session_state['_preview_array_cache']={'key':preview_key,'time':preview_t,'uap_col':uap_preview_col,'uap':uv_cached}

st.sidebar.markdown('### Event definition')
if bundled_mode:
    event_source='Bundled synthetic event'
    st.sidebar.info(f'Using annotated timing for bundled synthetic Event {int(events.iloc[0].event)}.')
else:
    event_source=st.sidebar.radio('INAP events',['Auto-detect from UAP','Upload event CSV','Manual event'],index=0 if uap_preview_col else 2)
    if event_source=='Upload event CSV':
        evup=st.sidebar.file_uploader('Event CSV',type=['csv'],key='events')
        if evup is None:
            events=pd.DataFrame(columns=['event','start_s','end_s','duration_s'])
        else:
            e=pd.read_csv(evup); sc=next((c for c in e.columns if 'start' in c.lower()),None); ec=next((c for c in e.columns if 'end' in c.lower()),None)
            if not sc or not ec: st.error('Event CSV needs start and end columns.'); st.stop()
            events=pd.DataFrame({'event':np.arange(1,len(e)+1),'start_s':pd.to_numeric(e[sc],errors='coerce'),'end_s':pd.to_numeric(e[ec],errors='coerce')}).dropna(); events['duration_s']=events.end_s-events.start_s
    elif event_source=='Manual event':
        finite_t=preview_t[np.isfinite(preview_t)]
        if finite_t.size<2: st.error('Could not determine recording time range.'); st.stop()
        lo=float(np.nanmin(finite_t)); hi=float(np.nanmax(finite_t)); default_s=lo+max(30.0,(hi-lo)*0.2)
        ss=float(st.sidebar.number_input('Event start [s]',min_value=lo,max_value=hi,value=min(default_s,hi)))
        ee=float(st.sidebar.number_input('Event end [s]',min_value=ss,max_value=hi,value=min(hi,ss+75.0)))
        events=pd.DataFrame([{'event':1,'start_s':ss,'end_s':ee,'duration_s':ee-ss}])
    else:
        if not uap_preview_col:
            st.error('Upper-airway pressure channel was not detected; choose Manual event or Upload event CSV.'); st.stop()
        uv_preview=uv_cached if uv_cached is not None else pd.to_numeric(preview_df[uap_preview_col],errors='coerce').to_numpy(float)
        ustats=suggest_uap_threshold(preview_t,uv_preview)
        mind=float(st.sidebar.number_input('Minimum event duration [s]',1.0,120.0,5.0,1.0)); merge_gap=float(st.sidebar.number_input('Merge gaps shorter than [s]',0.0,30.0,2.0,0.5))
        finite_u=uv_preview[np.isfinite(uv_preview)]; qlo=float(np.nanpercentile(finite_u,0.5)); qhi=float(np.nanpercentile(finite_u,99.5)); full_lo=float(np.nanmin(finite_u)); full_hi=float(np.nanmax(finite_u)); span=max(qhi-qlo,full_hi-full_lo,1e-6)
        slider_lo=min(full_lo,qlo-0.05*span); slider_hi=max(full_hi,qhi+0.05*span); step=max((slider_hi-slider_lo)/500.0,1e-4); suggested=float(np.clip(ustats['suggested_threshold'],slider_lo,slider_hi))
        uth=float(st.sidebar.slider('UAP event threshold',min_value=float(slider_lo),max_value=float(slider_hi),value=suggested,step=float(step)))
        events,_=detect_uap_events(preview_t,uv_preview,uth,mind,merge_gap); diag=diagnose_uap_detection(preview_t,uv_preview,uth,mind)

# ---------- immediate UAP/event preview ----------
st.subheader('UAP event-detection preview')
if uap_preview_col:
    uv_preview=uv_cached if uv_cached is not None else pd.to_numeric(preview_df[uap_preview_col],errors='coerce').to_numpy(float)
    n=len(preview_t); stride=max(1,int(np.ceil(n/12000)))
    px=preview_t[::stride]; py=uv_preview[::stride]
    fig_preview=go.Figure()
    fig_preview.add_trace(go.Scatter(x=px,y=py,name=str(uap_preview_col),line=dict(width=1)))
    if event_source=='Auto-detect from UAP':
        fig_preview.add_hline(y=uth,line_dash='dash',line_color='crimson',annotation_text=f'Detection threshold {uth:.3g}',annotation_position='top right')
        fig_preview.add_hline(y=ustats['baseline_median'],line_dash='dot',line_color='gray',annotation_text=f'Baseline median {ustats["baseline_median"]:.3g}',annotation_position='bottom right')
        fig_preview.add_hrect(y0=ustats['baseline_q25'],y1=ustats['baseline_q75'],fillcolor='gray',opacity=.08,line_width=0)
    if events is not None and not events.empty:
        for _,er in events.iterrows():
            fig_preview.add_vrect(x0=float(er.start_s),x1=float(er.end_s),fillcolor='gold',opacity=.16,line_width=0)
            fig_preview.add_vline(x=float(er.start_s),line_dash='dash',line_color='seagreen')
            fig_preview.add_vline(x=float(er.end_s),line_dash='dash',line_color='darkorange')
    fig_preview.update_layout(height=420,xaxis_title='Time from recording start [s]',yaxis_title=str(uap_preview_col),legend_orientation='h')
    st.plotly_chart(fig_preview,use_container_width=True)
else:
    st.info('No UAP channel was detected in this recording, so an automatic UAP preview is unavailable.')

if event_source=='Auto-detect from UAP':
    cprev=st.columns(4)
    cprev[0].metric('Baseline median',f"{ustats['baseline_median']:.3g}")
    cprev[1].metric('Baseline IQR',f"{ustats['baseline_q25']:.3g} to {ustats['baseline_q75']:.3g}")
    cprev[2].metric('Suggested threshold',f"{ustats['suggested_threshold']:.3g}")
    cprev[3].metric('Detected events',str(len(events)))
    st.caption('The suggested threshold is an automatic starting point based on the recording UAP distribution; it is not treated as a physiological gold standard. Confirm the detected intervals visually before analysis.')
    if events.empty:
        if diag['status'] in ('always_above','segments_too_short'): st.warning(diag['message'])
        elif diag['status']=='always_below': st.error(diag['message'])
        else: st.info(diag['message'])
    else:
        st.success(f'Detected {len(events)} candidate event(s). Start/end markers are shown above. Adjust the threshold if any interval is incorrect.')

if events is not None and not events.empty:
    st.dataframe(events,hide_index=True,use_container_width=True)
else:
    st.info('Define at least one event before continuing to SKNA analysis.')

# ---------- preprocessing ----------
if processed is None:
    st.sidebar.markdown('### Raw preprocessing')
    fs=float(meta.get('source_fs_hz',np.nan))
    if not np.isfinite(fs):
        tc=detect_time_column(raw); tt=pd.to_numeric(raw[tc],errors='coerce').to_numpy(float); fs=1/np.nanmedian(np.diff(tt[np.isfinite(tt)]))
        if tc!='time_s': raw=raw.rename(columns={tc:'time_s'})
    st.sidebar.write(f'Detected sampling rate: **{fs:g} Hz**')
    auto_ecg=detect_ecg_columns(raw)
    ecg_cols=st.sidebar.multiselect('ECG channels used for SKNA',list(raw.columns),default=auto_ecg)
    fsout=float(st.sidebar.number_input('Output sampling [Hz]',10.0,500.0,100.0,10.0))
    hp=float(st.sidebar.number_input('SKNA high-pass [Hz]',100.0,float(fs/2-1),500.0,50.0))
    env=float(st.sidebar.number_input('Envelope moving average [s]',0.1,5.0,1.0,0.1))
    clip_on=st.sidebar.checkbox('Apply optional MAD clipping before envelope',False)
    clipk=float(st.sidebar.number_input('MAD clipping multiplier',1.0,100.0,20.0,1.0)) if clip_on else None
    if st.sidebar.button('Run preprocessing',type='primary') or 'processed_raw_cache' in st.session_state:
        key=(source_name,tuple(ecg_cols),fsout,hp,env,clipk)
        if st.session_state.get('processed_raw_key')!=key:
            with st.spinner('Filtering ECG and generating SKNA envelopes…'):
                try: st.session_state.processed_raw_cache,st.session_state.processed_raw_meta=preprocess_raw_recording(raw,fs,ecg_cols,fsout,hp,env,mad_clip_k=clipk)
                except Exception as e: st.error(f'Preprocessing failed: {e}'); st.stop()
            st.session_state.processed_raw_key=key
        processed=st.session_state.processed_raw_cache; meta.update(st.session_state.processed_raw_meta)
    else:
        st.info('Event timing can be reviewed above before preprocessing. When the UAP intervals are correct, confirm the ECG channels and click **Run preprocessing** in the sidebar.'); st.stop()

# normalize processed time
if 'time_s' not in processed.columns:
    tc=detect_time_column(processed); processed=processed.rename(columns={tc:'time_s'})
t=pd.to_numeric(processed.time_s,errors='coerce').to_numpy(float)
if (not bundled_mode) and np.isfinite(t).any() and np.nanmin(t)!=0: processed['time_s']=t-np.nanmin(t); t=processed.time_s.to_numpy(float)

# Event definitions were established from the recorded UAP before preprocessing.
if events is None or events.empty:
    st.stop()
uap_col=detect_uap_column(processed)
st.sidebar.write(f'Events found: **{len(events)}**')
evnum=int(events.iloc[0].event) if bundled_mode else st.sidebar.selectbox('Selected event',events.event.tolist(),key='analysis_event_selector')
evrow=events.loc[events.event==evnum].iloc[0]; onset=float(evrow.start_s); offset=float(evrow.end_s); duration=offset-onset

# event-centred time and analysis window
baseline_duration=float(st.sidebar.number_input('Pre-event baseline duration [s]',10.0,300.0,60.0,10.0))
post_s=float(st.sidebar.number_input('Post-event display [s]',10.0,300.0,120.0,10.0))
view=(t>=max(float(np.nanmin(t)),onset-baseline_duration))&(t<=offset+post_s)
df=processed.loc[view].copy(); df['time_relative_s']=df.time_s-onset; tr=df.time_relative_s.to_numpy(float)

skcol=detect_median_skna_column(df); scale,note=infer_skna_scale_to_uv(df[skcol],skcol,None); y=pd.to_numeric(df[skcol],errors='coerce').to_numpy(float)*scale
base=(tr>=-baseline_duration)&(tr<0); inap=(tr>=0)&(tr<=duration); post=tr>duration
if base.sum()<20: st.error('Insufficient pre-event baseline samples.'); st.stop()
recalc_th=adaptive_threshold(y[base],95,6.0)
th=recalc_th; thr=float(th['adaptive_threshold_uV'])

st.sidebar.markdown('### Sequential replay')
win=float(st.sidebar.number_input('Trailing window [s]',5.0,120.0,30.0,5.0)); hop=float(st.sidebar.number_input('Update interval [s]',1.0,60.0,10.0,1.0)); occ=float(st.sidebar.number_input('Occupancy criterion [%]',0.1,100.0,5.0,0.5)); persist=int(st.sidebar.number_input('Persistence updates',1,10,2,1))
rep=sequential_replay(tr,y,thr,win,hop,occ,persist); met=replay_event_metrics(rep,0,duration,hop)
bm=float(np.nanmean(y[base])); im=float(np.nanmean(y[inap])); change=100*(im-bm)/bm if bm else np.nan

# top metrics
m=st.columns(6); m[0].metric('Event',str(evnum)); m[1].metric('Duration',f'{duration:.1f} s'); m[2].metric('Threshold',f'{thr:.3f} µV'); m[3].metric('SKNA change',f'{change:+.1f}%'); m[4].metric('Replay','Triggered' if met['trigger_success'] else 'No trigger'); m[5].metric('Latency','—' if not met['trigger_success'] else f"{met['latency_s']:.1f} s")
st.caption(f'Source: {source_name} · {note}')
if meta.get('repaired_irregular_rows'):
    st.warning('Input QC: one or more LabChart rows had an irregular number of tab-separated fields. The robust parser repaired those rows and continued. See Preprocessing → provenance for details.')

def shade(fig):
    fig.add_vrect(x0=0,x1=duration,fillcolor='gold',opacity=.12,line_width=0,annotation_text='INAP',annotation_position='top left'); fig.add_vline(x=0,line_dash='dash',line_color='gray'); fig.add_vline(x=duration,line_dash='dash',line_color='darkorange')

tabs=st.tabs(['Overview','Recorded/raw channels','Processed SKNA','Preprocessing','Adaptive threshold','Burst analysis','Sequential replay','Event results','Export'])
with tabs[0]:
    st.subheader(f'Event {evnum} overview')
    f=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.4,.6],vertical_spacing=.06)
    if uap_col: f.add_trace(go.Scatter(x=tr,y=pd.to_numeric(df[uap_col],errors='coerce'),name='UAP',line=dict(width=1)),row=1,col=1)
    f.add_trace(go.Scatter(x=tr,y=y,name='Median SKNA',line=dict(width=1.2)),row=2,col=1); f.add_hline(y=thr,line_dash='dash',line_color='crimson',row=2,col=1)
    f.add_vrect(x0=0,x1=duration,fillcolor='gold',opacity=.1,line_width=0,row='all',col=1); f.update_layout(height=620); f.update_xaxes(title_text='Time relative to INAP onset [s]',row=2,col=1); st.plotly_chart(f,use_container_width=True)
with tabs[1]:
    st.subheader('Recorded/raw channels preserved in the input')
    rec=detect_recorded_channels(df); default=[c for c in rec if any(k in str(c).lower() for k in ('ecg','upper airway','spo2','heart rate'))][:6]
    sel=st.multiselect('Channels',rec,default=default or rec[:4])
    if sel:
        f=make_subplots(rows=len(sel),cols=1,shared_xaxes=True,vertical_spacing=.015)
        for i,c in enumerate(sel,1): f.add_trace(go.Scatter(x=tr,y=pd.to_numeric(df[c],errors='coerce'),name=c,line=dict(width=.8)),row=i,col=1)
        f.add_vrect(x0=0,x1=duration,fillcolor='gold',opacity=.08,line_width=0,row='all',col=1); f.update_layout(height=max(450,150*len(sel)),showlegend=False); st.plotly_chart(f,use_container_width=True)
with tabs[2]:
    st.subheader('Processed ECG-derived SKNA')
    pc=detect_processed_channels(df); f=go.Figure()
    for c in pc['channel_skna']:
        sc,_=infer_skna_scale_to_uv(df[c],c,None); f.add_trace(go.Scatter(x=tr,y=pd.to_numeric(df[c],errors='coerce')*sc,name=c,line=dict(width=.7)))
    f.add_trace(go.Scatter(x=tr,y=y,name='Median SKNA',line=dict(width=2))); shade(f); f.update_layout(height=480,yaxis_title='SKNA [µV]',xaxis_title='Time relative to onset [s]'); st.plotly_chart(f,use_container_width=True)
with tabs[3]:
    st.subheader('Preprocessing provenance')
    st.json({'input_kind':source_kind,'source':source_name,**meta})
    if source_kind!='processed': st.info('Raw pipeline: ECG selection → 500-Hz high-pass Butterworth → rectification → optional MAD clipping → 1-s moving-average envelope → resampling → median across ECG channels. A separate 0.5–45 Hz ECG trace is generated for display.')
    else: st.info('Processed SKNA was supplied, so raw preprocessing was not repeated.')
with tabs[4]:
    st.subheader('Adaptive threshold')
    st.caption(f'Threshold recalculated from up to {baseline_duration:g} s immediately before event onset; the onset sample is excluded.')
    c=st.columns(4); c[0].metric('Final',f'{thr:.3f} µV'); c[1].metric('GMM',f"{th['gmm_threshold_uV']:.3f} µV"); c[2].metric('95th percentile',f"{th['q95_threshold_uV']:.3f} µV"); c[3].metric('Median + 6×MAD',f"{th['median_plus_6mad_uV']:.3f} µV")
    f=go.Figure(go.Scatter(x=tr,y=y,name='Median SKNA')); f.add_hline(y=thr,line_dash='dash',line_color='crimson'); shade(f); f.update_layout(height=440,yaxis_title='SKNA [µV]'); st.plotly_chart(f,use_container_width=True)
with tabs[5]:
    st.subheader('Burst analysis'); above=y>thr
    f=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.7,.3]); f.add_trace(go.Scatter(x=tr,y=y,name='Median SKNA'),row=1,col=1); f.add_hline(y=thr,line_dash='dash',line_color='crimson',row=1,col=1); f.add_trace(go.Scatter(x=tr,y=above.astype(int),line_shape='hv',name='Burst state'),row=2,col=1); f.add_vrect(x0=0,x1=duration,fillcolor='gold',opacity=.1,line_width=0,row='all',col=1); f.update_layout(height=600); st.plotly_chart(f,use_container_width=True)
    st.dataframe(pd.DataFrame({'Phase':['Baseline','INAP','Post-INAP'],'Burst-active samples [%]':[100*above[base].mean(),100*above[inap].mean(),100*above[post].mean() if post.any() else np.nan]}),hide_index=True,use_container_width=True)
with tabs[6]:
    st.subheader('Sequential replay under real-time information constraints')
    f=make_subplots(rows=3,cols=1,shared_xaxes=True,row_heights=[.45,.3,.25]); f.add_trace(go.Scatter(x=tr,y=y,name='Median SKNA'),row=1,col=1); f.add_hline(y=thr,line_dash='dash',line_color='crimson',row=1,col=1); f.add_trace(go.Scatter(x=rep.window_end_relative_s,y=rep.burst_occupancy_pct,mode='lines+markers',line_shape='hv',name='Occupancy'),row=2,col=1); f.add_hline(y=occ,line_dash='dash',line_color='crimson',row=2,col=1); f.add_trace(go.Scatter(x=rep.window_end_relative_s,y=rep.trigger_on,line_shape='hv',name='Trigger'),row=3,col=1); f.add_vrect(x0=0,x1=duration,fillcolor='gold',opacity=.1,line_width=0,row='all',col=1); f.update_layout(height=720); st.plotly_chart(f,use_container_width=True); st.dataframe(rep,hide_index=True,use_container_width=True)
with tabs[7]:
    st.subheader('Event results')
    result={'event':int(evnum),'start_s':onset,'end_s':offset,'duration_s':duration,'baseline_mean_uV':bm,'inap_mean_uV':im,'skna_change_pct':change,'adaptive_threshold_uV':thr,**th,**met}
    st.json(result); st.dataframe(events,hide_index=True,use_container_width=True)
with tabs[8]:
    st.subheader('Reproducible export')
    analysed=df.copy(); analysed['median_skna_uV_gui']=y; analysed['above_threshold_gui']=(y>thr)
    config={'version':'1.0.0','source':source_name,'source_kind':source_kind,'event':int(evnum),'event_start_s':onset,'event_end_s':offset,'baseline_duration_before_onset_s':baseline_duration,'threshold':th,'replay':{'window_s':win,'hop_s':hop,'occupancy_threshold_pct':occ,'persistence_windows':persist},'preprocessing':meta}
    st.download_button('Download analysed event CSV',analysed.to_csv(index=False),f'event{evnum}_analysed.csv','text/csv'); st.download_button('Download replay CSV',rep.to_csv(index=False),f'event{evnum}_replay.csv','text/csv'); st.download_button('Download configuration JSON',json.dumps(config,indent=2,default=str),f'event{evnum}_config.json','application/json'); st.download_button('Download detected event table',events.to_csv(index=False),'events.csv','text/csv')
