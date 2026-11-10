#!/usr/bin/env python3
"""Vinfotech traffic-intersection project: desktop/VS Code compatible.

Reuse the included all_vehicle_trajectories.csv to reproduce the traffic analytics
without rerunning YOLO. To process a new video, provide --video and --infer.

The geometry is calibrated ONLY for the 2017 Bellevue NE8th 1920x1080 video.
All counts are provisional; no population-level accuracy is claimed.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARMS = ['NORTH', 'EAST', 'SOUTH', 'WEST']
BASE_W, BASE_H = 1920, 1080
# Coordinates correspond to the final corrected Cell 3 and East-out adjustment.
GATES_BASE = {
    'NORTH_IN': {'a':(750,285),'b':(970,259),'ref':(840,165)},
    'NORTH_OUT':{'a':(978,254),'b':(1145,282),'ref':(1060,390)},
    'EAST_IN':  {'a':(1365,342),'b':(1490,475),'ref':(1620,365)},
    'EAST_OUT': {'a':(1370,490),'b':(1495,700),'ref':(1190,590)},
    'WEST_IN':  {'a':(440,500),'b':(445,735),'ref':(230,625)},
    'WEST_OUT': {'a':(510,375),'b':(515,520),'ref':(735,425)},
    'SOUTH_IN': {'a':(1005,910),'b':(1450,755),'ref':(1280,1005)},
    'SOUTH_OUT':{'a':(620,715),'b':(1000,855),'ref':(920,615)},
}
QUEUE_ZONES_BASE = {
    'NORTH':[(690,75),(990,60),(1000,250),(720,290)],
    'EAST': [(1380,320),(1875,320),(1870,485),(1460,505)],
    'WEST': [(0,490),(435,465),(450,750),(0,815)],
    'SOUTH':[(950,875),(1470,760),(1665,1045),(1120,1070)],
}

def geometry(width:int, height:int):
    def point(p):return (int(round(p[0]*width/BASE_W)), int(round(p[1]*height/BASE_H)))
    gates = {k:{q:point(v) for q,v in g.items()} for k,g in GATES_BASE.items()}
    zones = {k:np.array([point(p) for p in points], dtype=np.int32)
             for k,points in QUEUE_ZONES_BASE.items()}
    return gates,zones


def load_road_geometry(config_path:Path):
    """Load all road-dependent coordinates from a human-editable JSON file.

    IMPORTANT: when changing camera viewpoint, the new gates/zones MUST be
    calibrated from a reference frame. Resizing is not camera calibration.
    """
    global BASE_W, BASE_H, GATES_BASE, QUEUE_ZONES_BASE
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    resolution=config['reference_resolution']
    BASE_W,BASE_H=map(int,resolution)
    expected={f'{arm}_{suffix}' for arm in ARMS for suffix in ['IN','OUT']}
    if set(config['gates'])!=expected:
        raise ValueError(f'Geometry must have exactly these gates: {sorted(expected)}')
    if set(config['queue_zones'])!=set(ARMS):
        raise ValueError('Geometry must contain NORTH/EAST/SOUTH/WEST queue zones')
    for name,g in config['gates'].items():
        if set(g)!={'a','b','ref'}:
            raise ValueError(f'Bad gate definition {name}: use a,b,ref')
        if any(len(g[key])!=2 for key in ['a','b','ref']):
            raise ValueError(f'Bad gate endpoints for {name}')
    for k,polygon in config['queue_zones'].items():
        if len(polygon)<3: raise ValueError(f'Queue polygon {k} needs 3+ points')
    GATES_BASE=config['gates']
    QUEUE_ZONES_BASE=config['queue_zones']
    return config


def calibration_preview(video:Path, outdir:Path, timestamp:float=60.0):
    fps,w,h,total,duration=video_metadata(video)
    cap=cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC,min(max(0,timestamp),max(0,duration-.5))*1000)
    ret,frame=cap.read();cap.release()
    if not ret:raise RuntimeError('Cannot read calibration preview frame')
    gates,zones=geometry(w,h)
    for arm,poly in zones.items():
        cv2.polylines(frame,[poly],True,(30,225,45),3)
        mid=np.rint(poly.mean(axis=0)).astype(int)
        cv2.putText(frame,arm+'_QUEUE',tuple(mid),cv2.FONT_HERSHEY_SIMPLEX,.85,(30,225,45),2)
    for name,g in gates.items():
        color=(0,165,255) if name.endswith('_IN') else (255,255,0)
        cv2.line(frame,g['a'],g['b'],color,4)
        cv2.putText(frame,name,g['a'],cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
        cv2.circle(frame,g['ref'],5,color,-1)
    dest=outdir/'calibration_preview.jpg'
    cv2.imwrite(str(dest),frame)
    print('Calibration preview:',dest)
    return dest

def video_metadata(video:Path):
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise RuntimeError(f'Cannot open video: {video}')
    fps=float(cap.get(cv2.CAP_PROP_FPS))
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if fps<=0 or width<=0 or height<=0: raise RuntimeError('Invalid video metadata')
    return fps,width,height,total,total/fps

def infer_tracks(video:Path, outdir:Path, *, model_name='yolo11s.pt',
                 stride=2, imgsz=1280, conf=0.20):
    """Run YOLO + BoT-SORT. Only needed when no saved trajectory CSV exists.

    PyTorch GPU installation is platform-dependent; see README_VSCODE.md.
    """
    import torch
    import yaml
    import ultralytics
    from ultralytics import YOLO
    fps,w,h,total,dur=video_metadata(video)
    model=YOLO(model_name)
    gpu=torch.cuda.is_available()
    cfg_path=Path(ultralytics.__file__).resolve().parent/'cfg'/'trackers'/'botsort.yaml'
    with open(cfg_path,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    cfg.update(track_buffer=60,track_low_thresh=0.08,gmc_method='none',with_reid=False)
    tracker_path=outdir/'fixed_cam_botsort.yaml'
    with open(tracker_path,'w',encoding='utf-8') as f:yaml.safe_dump(cfg,f,sort_keys=False)
    fields=['frame','time_s','track_id','class_id','confidence','x1','y1','x2','y2','x','y']
    path=outdir/'all_vehicle_trajectories.csv'
    cap=cv2.VideoCapture(str(video))
    try:
        with open(path,'w',newline='',encoding='utf-8') as file:
            wr=csv.DictWriter(file,fieldnames=fields);wr.writeheader()
            frame_no=0
            while frame_no<total:
                if frame_no%stride:
                    if not cap.grab():break
                    frame_no+=1;continue
                ok,frame=cap.read()
                if not ok:break
                result=model.track(frame,persist=True,tracker=str(tracker_path),
                    classes=[2,3,5,7],conf=conf,imgsz=imgsz,
                    device=0 if gpu else 'cpu',verbose=False)[0]
                if result.boxes is not None and result.boxes.id is not None:
                    ids=result.boxes.id.int().cpu().tolist()
                    boxes=result.boxes.xyxy.cpu().numpy()
                    labels=result.boxes.cls.int().cpu().tolist()
                    scores=result.boxes.conf.cpu().tolist()
                    for tid,b,cls,score in zip(ids,boxes,labels,scores):
                        x1,y1,x2,y2=map(float,b)
                        wr.writerow(dict(frame=frame_no,time_s=round(frame_no/fps,4),
                            track_id=int(tid),class_id=int(cls),confidence=round(float(score),4),
                            x1=round(x1,1),y1=round(y1,1),x2=round(x2,1),y2=round(y2,1),
                            x=round((x1+x2)/2,1),y=round(y2,1)))
                frame_no+=1
                if frame_no%int(fps*30)<stride:
                    print(f'Inference: video second {frame_no/fps:.0f}/{dur:.0f}',flush=True)
    finally:cap.release()
    return path

def annotated_video(video:Path, trajectory_csv:Path, outdir:Path,
                    *, stride=2, min_conf=0.0, exclude_json=None, trail_seconds=2.0,
                    queue_unavailable=None, source_note=None):
    """Annotate every sampled frame with boxes, IDs, trails, gates, IN/OUT and route labels.

    The overlay consumes saved tracking and analytics CSVs. It does not run YOLO.
    Exclusion polygons suppress DISPLAY ONLY; CSV and metrics are unchanged.
    """
    fps,w,h,total,duration=video_metadata(video)
    stride=max(1,int(stride))
    gates,zones=geometry(w,h)
    tracks=pd.read_csv(trajectory_csv,usecols=['frame','time_s','track_id',
        'confidence','x1','y1','x2','y2','x','y'])
    tracks=tracks.sort_values(['frame','track_id'])
    boxes=tracks if min_conf<=0 else tracks[tracks.confidence>=min_conf]
    polygons=[]
    if exclude_json:
        config=json.loads(Path(exclude_json).read_text(encoding='utf-8'))
        polygons=[np.asarray(poly,dtype=np.float32) for poly in config.get('exclude_polygons',[])]
    by_frame={int(k):g for k,g in boxes.groupby('frame')}
    gate_df_path=outdir/'gate_crossing_events_deduplicated.csv'
    gate_df=pd.read_csv(gate_df_path) if gate_df_path.exists() else pd.DataFrame()
    if not gate_df.empty: gate_df=gate_df.sort_values('time_s')
    pair_path=outdir/'turning_movements_matched.csv'
    pair_df=pd.read_csv(pair_path) if pair_path.exists() else pd.DataFrame()
    route_by_id={}
    if not pair_df.empty:
        for r in pair_df.sort_values('Exit_s').itertuples():
            route_by_id[int(r.track_id)]=(str(r.Entry),str(r.Exit),float(r.Exit_s))
    queues_path=outdir/'queue_per_second.csv'
    queues=pd.read_csv(queues_path) if queues_path.exists() else pd.DataFrame()
    queue_lookup={(int(r.Second),str(r.Approach)):int(r.Stationary_tracks)
                 for r in queues.itertuples(index=False)} if not queues.empty else {}
    cap=cv2.VideoCapture(str(video))
    out_w=1280;out_h=round(h*out_w/w)
    sx,sy=out_w/w,out_h/h
    path=outdir/'annotated_traffic_desktop.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps/stride,(out_w,out_h))
    if not writer.isOpened():cap.release();raise RuntimeError('VideoWriter failed')
    count=0
    unavailable=set(queue_unavailable or [])
    from collections import defaultdict, deque
    last_trail_time={}
    trails=defaultdict(lambda:deque(maxlen=max(6,int(trail_seconds*fps/stride))))
    crossings=gate_df.to_dict('records') if not gate_df.empty else []
    event_idx=0
    entry_counts={a:0 for a in ARMS};exit_counts={a:0 for a in ARMS}
    try:
        f=0
        while f<total:
            if f%stride:
                ok=cap.grab();f+=1
                if not ok:break
                continue
            ok,frame=cap.read()
            if not ok:break
            now=f/fps;canvas=cv2.resize(frame,(out_w,out_h))
            while event_idx<len(crossings) and crossings[event_idx]['time_s']<=now:
                ev=crossings[event_idx];arm=ev['arm']
                if ev['kind']=='IN': entry_counts[arm]+=1
                else: exit_counts[arm]+=1
                event_idx+=1
            grp=by_frame.get(f)
            if grp is not None:
                for r in grp.itertuples(index=False):
                    if any(cv2.pointPolygonTest(p,(float(r.x),float(r.y)),False)>=0
                           for p in polygons): continue
                    tid=int(r.track_id)
                    xy=(int(r.x*sx),int(r.y*sy))
                    if now-last_trail_time.get(tid,now)>1.0:trails[tid].clear()
                    trails[tid].append(xy)
                    last_trail_time[tid]=now
                    a=(int(r.x1*sx),int(r.y1*sy)); b=(int(r.x2*sx),int(r.y2*sy))
                    cv2.rectangle(canvas,a,b,(0,255,60),2)
                    label=f'{tid}'
                    route=route_by_id.get(tid)
                    if route and now>=route[2]:label+=f' {route[0]}->{route[1]}'
                    cv2.putText(canvas,label,(a[0],max(16,a[1]-4)),
                                cv2.FONT_HERSHEY_SIMPLEX,.45,(0,255,60),1)
                    trail=list(trails[tid])
                    if len(trail)>1:
                        cv2.polylines(canvas,[np.array(trail,dtype=np.int32)],
                                      False,(60,210,255),2)
            for name,g in gates.items():
                color=(0,165,255) if name.endswith('_IN') else (255,255,0)
                a=(int(g['a'][0]*sx),int(g['a'][1]*sy))
                b=(int(g['b'][0]*sx),int(g['b'][1]*sy))
                cv2.line(canvas,a,b,color,2)
                label_xy=(max(2,min(out_w-170,a[0])),max(18,min(out_h-15,a[1]-5)))
                cv2.putText(canvas,name,label_xy,cv2.FONT_HERSHEY_SIMPLEX,.49,color,1)
            # Dynamic IN and OUT counts (gate candidate events; not ground truth).
            bar_height=152
            cv2.rectangle(canvas,(8,8),(435,bar_height),(20,20,20),-1)
            cv2.putText(canvas,f'Video {now:06.1f}s  |  YOLO11s + BoT-SORT',
                        (18,30),cv2.FONT_HERSHEY_SIMPLEX,.56,(255,255,255),1)
            cv2.putText(canvas,'Approach      IN     OUT    Queue proxy',
                        (18,55),cv2.FONT_HERSHEY_SIMPLEX,.49,(255,255,255),1)
            for j,arm in enumerate(ARMS):
                q=queue_lookup.get((int(now),arm))
                qtext='N/A' if arm in unavailable else (str(q) if q is not None else 'N/A')
                cv2.putText(canvas,f'{arm:<8}     {entry_counts[arm]:>3}     {exit_counts[arm]:>3}        {qtext}',
                            (18,76+j*19),cv2.FONT_HERSHEY_SIMPLEX,.49,(240,240,240),1)
            writer.write(canvas);f+=1;count+=1
            if count%1500==0:print(f'Annotated frames: {count}',flush=True)
    finally:
        writer.release();cap.release()
    if count==0:raise RuntimeError('No frames were written to annotated video')
    print('Annotated output:',path,'frames:',count)
    return path

def analyze_tracks(trajectory_csv:Path, outdir:Path, *, video=None,
                   fps=30.0, duration=None, stride=2, reviewed_audit=False,
                   manual_validation_csv=None, model_name='yolo11s.pt',
                   queue_unavailable=None, source_note=None):
    ROOT=outdir
    TRACK_CSV=trajectory_csv
    if video is not None:
        FPS,W,H,TOTAL,DURATION=video_metadata(Path(video))
        VIDEO=Path(video)
    else:
        FPS,W,H,TOTAL=float(fps),1920,1080,0
        if duration is None:
            probe=pd.read_csv(TRACK_CSV,usecols=['time_s'])
            duration=float(probe.time_s.max())+1.0/FPS
        DURATION=float(duration)
        VIDEO=Path('precomputed_trajectory_csv')
    FRAME_STRIDE=int(stride)
    MODEL_NAME=str(model_name)
    GATES,QUEUE_ZONES=geometry(W,H)
    from collections import defaultdict
    if not TRACK_CSV.exists():
        raise FileNotFoundError('Run tracking Cell 4 first; trajectory CSV missing')
    df=pd.read_csv(TRACK_CSV)
    if df.empty: raise ValueError('No trajectories saved')
    for col in ['track_id','frame','class_id']:
        df[col]=pd.to_numeric(df[col],errors='coerce').astype('Int64')
    df=df.dropna(subset=['track_id','frame','time_s','x','y'])
    df[['track_id','frame']]=df[['track_id','frame']].astype(int)
    df=df.sort_values(['track_id','frame']).reset_index(drop=True)
    print('Trajectory rows:',len(df),'IDs:',df.track_id.nunique())

    # Accurate directed crossing of a finite segment (in pixels, not raw cross-products).
    def directed_cross(g, p, q):
        a=np.array(g['a'],float);b=np.array(g['b'],float)
        ref=np.array(g['ref'],float)
        edge=b-a
        den=np.linalg.norm(edge)
        if den<1:return None
        def cross2(u,v):return u[0]*v[1]-u[1]*v[0]
        sign=np.sign(cross2(edge,ref-a))
        if sign==0:return None
        p=np.array(p,float);q=np.array(q,float)
        dp=cross2(edge,p-a)/den*sign
        dq=cross2(edge,q-a)/den*sign
        if not (dp>0.15 and dq<=0.15):return None
        if np.linalg.norm(q-p)<0.8:return None
        frac=dp/(dp-dq)
        cross=p+frac*(q-p)
        u=np.dot(cross-a,edge)/np.dot(edge,edge)
        if u < -0.02 or u > 1.02:return None
        return float(frac),float(cross[0]),float(cross[1])

    gate_events=[]
    # Read every pair of points from same tracked ID and count only first valid pass/gate.
    for tid,part in df.groupby('track_id',sort=False):
        part=part.sort_values('time_s').copy()
        part['gx']=part.x.rolling(5,center=True,min_periods=1).median()
        part['gy']=part.y.rolling(5,center=True,min_periods=1).median()
        vals=part[['frame','time_s','gx','gy']].to_numpy(float)
        previous_event=defaultdict(lambda:-1e9)
        for i in range(1,len(vals)):
            f0,t0,x0,y0=vals[i-1]
            f1,t1,x1,y1=vals[i]
            if t1-t0 > 0.55 or t1<=t0:continue
            for name,g in GATES.items():
                hit=directed_cross(g,(x0,y0),(x1,y1))
                if hit is None:continue
                fraction,cx,cy=hit
                t=t0+fraction*(t1-t0)
                if t-previous_event[name]<3:continue
                previous_event[name]=t
                gate_events.append({'track_id':int(tid),'time_s':round(t,3),
                    'gate':name,'arm':name.split('_')[0],
                    'kind':name.split('_')[1],
                    'x':round(cx,1),'y':round(cy,1)})

    events=pd.DataFrame(gate_events,columns=['track_id','time_s','gate','arm','kind','x','y'])
    events.to_csv(ROOT/'gate_crossing_events.csv',index=False)

    # For audit only: near-exact spatial/time duplicates likely indicate track fragmentation.
    def dedup_gate(df_events,time_tol=0.45,dist_tol=19):
        if df_events.empty:return df_events.copy()
        rows=[]
        for gate,gp in df_events.groupby('gate'):
            keep=[]
            for r in gp.sort_values('time_s').to_dict('records'):
                duplicate=any(r['track_id']!=k['track_id'] and
                    abs(r['time_s']-k['time_s'])<=time_tol and
                    math.hypot(r['x']-k['x'],r['y']-k['y'])<=dist_tol for k in keep)
                if not duplicate:keep.append(r)
            rows.extend(keep)
        return pd.DataFrame(rows,columns=df_events.columns)

    dedup_events=dedup_gate(events)
    dedup_events.to_csv(ROOT/'gate_crossing_events_deduplicated.csv',index=False)

    ARMS=['NORTH','EAST','SOUTH','WEST']
    entries=dedup_events[dedup_events.kind=='IN']
    counts=entries.groupby('arm').size().reindex(ARMS,fill_value=0)
    entry_summary=pd.DataFrame({'Approach':ARMS,'Incoming_crossing_candidates':counts.values})
    entry_summary.to_csv(ROOT/'approach_entry_counts.csv',index=False)
    exits=dedup_events[dedup_events.kind=='OUT']
    exit_counts=exits.groupby('arm').size().reindex(ARMS,fill_value=0)
    pd.DataFrame({'Approach':ARMS,'Outgoing_crossing_candidates':exit_counts.values}).to_csv(
        ROOT/'approach_exit_counts.csv',index=False)
    pd.DataFrame({'Approach':ARMS,'IN_candidates':counts.values,
       'OUT_candidates':exit_counts.values}).to_csv(ROOT/'approach_in_out_counts.csv',index=False)


    # Match crossing event after entry for SAME track ID; unmatched are not guessed.
    pairs=[];matched_ids=set()
    for tid,part in events.groupby('track_id'):
        seq=part.sort_values('time_s').to_dict('records')
        for i,start in enumerate(seq):
            if start['kind']!='IN':continue
            exits=[r for r in seq[i+1:] if r['kind']=='OUT' and
                   r['arm']!=start['arm'] and 0.5<=r['time_s']-start['time_s']<=150]
            if not exits:continue
            finish=exits[0]
            pairs.append({'track_id':int(tid),'Entry':start['arm'],
                 'Exit':finish['arm'],'Entry_s':start['time_s'],
                 'Exit_s':finish['time_s'],
                 'Traverse_s':round(finish['time_s']-start['time_s'],2)})
            matched_ids.add(int(tid))
            break
    pairs_df=pd.DataFrame(pairs,columns=['track_id','Entry','Exit','Entry_s','Exit_s','Traverse_s'])
    pairs_df.to_csv(ROOT/'turning_movements_matched.csv',index=False)
    pairs_df.to_csv(ROOT/'vehicle_routes.csv',index=False)  # entry->exit paths for matched tracks
    # Random movements for fast human verification; leave verdicts empty until examined.
    audit_sample=pairs_df.sample(n=min(20,len(pairs_df)),random_state=42).copy()
    audit_sample['Verified_same_vehicle']=''
    audit_sample['Verified_entry_exit']=''
    audit_sample['Notes']=''
    audit_sample.to_csv(ROOT/'manual_validation_sample.csv',index=False)
    mat=pd.crosstab(pairs_df['Entry'],pairs_df['Exit']).reindex(index=ARMS,columns=ARMS,fill_value=0)
    mat.index.name='Entry / Exit'
    mat.to_csv(ROOT/'turning_matrix_4x4.csv')

    unmatched=entries[~entries.track_id.isin(matched_ids)].copy()
    unmatched.to_csv(ROOT/'unmatched_entries.csv',index=False)

    # Estimate stopped episodes using smoothed contact positions and consecutive slow measurements.
    # Pixel-speed is perspective-dependent: these are provisional proxy measurements.
    stop_episodes=[]
    for tid,part in df.groupby('track_id',sort=False):
        part=part.sort_values('time_s').copy()
        if len(part)<10:continue
        
        # More stable stationary-vehicle speed estimate
        # Works with the actual sampling rate of the tracking CSV

        processed_fps = FPS / FRAME_STRIDE

        # Smooth noisy bounding-box contact positions
        smooth_window = max(5, int(round(processed_fps * 0.6)))
        if smooth_window % 2 == 0:
            smooth_window += 1

        part["xs"] = (
            part.x.rolling(
                smooth_window,
                center=True,
                min_periods=1
            ).median()
        )

        part["ys"] = (
            part.y.rolling(
                 smooth_window,
                 center=True,
                min_periods=1
            ).median()
        ) 

        # Measure displacement across approximately one second
        lag = max(2, int(round(processed_fps * 1.0)))

        dx = part["xs"].diff(lag)
        dy = part["ys"].diff(lag)
        dt = part.time_s.diff(lag)

        part["speed"] = (
            np.hypot(dx, dy) / dt.where(dt > 0)
        )

        pvals=part[['time_s','xs','ys','speed']].to_numpy(float)
        FIRST= float(part.time_s.iloc[0])
        for arm,poly in QUEUE_ZONES.items():
            inside=np.array([cv2.pointPolygonTest(poly,(float(x),float(y)),False)>=0
                             for t,x,y,s in pvals])
            slow=inside & np.isfinite(pvals[:,3]) & (pvals[:,3]<10)
            active=False; start=0
            for i in range(len(slow)+1):
                status=bool(slow[i]) if i<len(slow) else False
                gap=False
                if i>0 and i<len(slow):gap=pvals[i,0]-pvals[i-1,0]>0.55
                if active and (not status or gap or i==len(slow)):
                    e=i-1
                    if e>=start and pvals[e,0]-pvals[start,0]>=1.8:
                        s_time=float(pvals[start,0]);end_time=float(pvals[e,0])
                        # Detect departure within the next 3 seconds
                        departure = False
                        last_time = end_time
                        moving_samples = 0

                        vehicle_height = float(
                            part["y2"].iloc[e] - part["y1"].iloc[e]
                        )

                        min_move_px = max(
                            6.0, 0.12 * vehicle_height
                        )

                        for j in range(e + 1, len(pvals)):
                            current_time = float(pvals[j, 0])

                            if current_time - end_time > 3.0:
                                break

                            if current_time - last_time > 0.55:
                                break

                            speed_j = float(pvals[j, 3])

                            distance_j = float(
                                np.hypot(
                                    pvals[j, 1] - pvals[e, 1],
                                    pvals[j, 2] - pvals[e, 2]
                                )
                            )

                            moving = (
                                np.isfinite(speed_j)
                                and speed_j > 11.0
                                and distance_j >= min_move_px
                            )

                            if moving:
                                moving_samples += 1

                                if moving_samples >= 3:
                                    departure = True
                                    break
                            else:
                                moving_samples = 0

                            last_time = current_time
  
                        before_stop=s_time-FIRST>=1.5
                        stop_episodes.append({'track_id':int(tid),'Approach':arm,
                            'Stop_start_s':round(s_time,2),'Stop_end_s':round(end_time,2),
                            'Stationary_s':round(end_time-s_time,2),
                            'Departure_seen':bool(departure),
                            'Approach_observed':bool(before_stop),
                            'x':round(float(pvals[start,1]),1),
                            'y':round(float(pvals[start,2]),1)})
                    active=False
                if status and not active:active=True;start=i

    episodes=pd.DataFrame(stop_episodes,columns=['track_id','Approach','Stop_start_s','Stop_end_s',
        'Stationary_s','Departure_seen','Approach_observed','x','y'])
    episodes.to_csv(ROOT/'stopped_vehicle_episodes.csv',index=False)

    # Waiting times: report the well-observed ones; others flagged for audit.
    if not episodes.empty:
        episodes['Complete_wait'] = episodes.Departure_seen & episodes.Approach_observed
    else:episodes['Complete_wait']=pd.Series(dtype=bool)
    complete=episodes[episodes.Complete_wait].copy()
    complete.to_csv(ROOT/'completed_stationary_waits.csv',index=False)

    # Queue proxy by second (counts IDs whose stopped-episode window includes timestamp).
    seconds=np.arange(0,int(np.ceil(DURATION)))
    queue_records=[]
    for second in seconds:
        for arm in ARMS:
            if episodes.empty:n=0
            else:
                active=episodes[(episodes.Approach==arm) &
                                (episodes.Stop_start_s<=second) &
                                (episodes.Stop_end_s>=second)]
                n=int(active.track_id.nunique())
            queue_records.append({'Second':int(second),'Approach':arm,'Stationary_tracks':n})
    queue_ts=pd.DataFrame(queue_records)
    queue_ts.to_csv(ROOT/'queue_per_second.csv',index=False)

    summ=[]
    for arm in ARMS:
        q=queue_ts[queue_ts.Approach==arm].Stationary_tracks
        c=complete[complete.Approach==arm]
        summ.append({'Approach':arm,
          'Incoming_candidate_count':int(counts.loc[arm]),
          'Max_observed_stationary_tracks':int(q.max()) if len(q) else 0,
          'Complete_wait_samples':len(c),
          'Average_stationary_wait_s':round(float(c.Stationary_s.mean()),2) if len(c) else None,
          'Maximum_stationary_wait_s':round(float(c.Stationary_s.max()),2) if len(c) else None})
    summary=pd.DataFrame(summ)
    if queue_unavailable:
        mask=summary.Approach.isin(queue_unavailable)
        for col in ['Max_observed_stationary_tracks','Average_stationary_wait_s',
                    'Maximum_stationary_wait_s']:
            summary.loc[mask,col]=float('nan')
        summary.loc[mask,'Complete_wait_samples']=0
    summary.to_csv(ROOT/'intersection_summary.csv',index=False)
    print('GATE EVENTS:',len(events),'| MATCHED TURNING MOVEMENTS:',len(pairs_df),
          '| UNMATCHED ENTRIES:',len(unmatched))
    print('\n4 x 4 TURNING MATRIX (PROVISIONAL)'); print(mat.to_string())
    print('\nAPPROACH SUMMARY (PROVISIONAL)'); print(summary.to_string(index=False))
    # Basic plots
    fig,ax=plt.subplots(figsize=(8,5))
    ax.imshow(mat.values,cmap='Blues')
    ax.set_xticks(range(4),ARMS);ax.set_yticks(range(4),ARMS)
    ax.set_xlabel('Exit approach');ax.set_ylabel('Entry approach')
    ax.set_title('Candidate turning movements (matched track IDs)')
    for i in range(4):
     for j in range(4):ax.text(j,i,str(int(mat.iloc[i,j])),ha='center',va='center')
    fig.tight_layout();fig.savefig(ROOT/'turning_matrix_heatmap.png',dpi=170);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5))
    for arm in ARMS:
        q=queue_ts[queue_ts.Approach==arm]
        ax.plot(q.Second,q.Stationary_tracks,label=arm,linewidth=1.4)
    ax.set(xlabel='Video time (seconds)',ylabel='Observed nearly-stationary track IDs',
           title='Queue proxy by approach — not physical ground-truth queue')
    ax.legend();fig.tight_layout();fig.savefig(ROOT/'queue_timeline.png',dpi=170)
    plt.close(fig)

    report={
      'source':os.path.basename(VIDEO),
      'source_note':'Input source must be documented by submitter. The provided Bellevue dataset is archived 2017 footage, not a current live recording.',
      'video_duration_s':round(DURATION,3),'source_fps':FPS,
      'processed_fps':round(FPS/FRAME_STRIDE,3),
      'detector':MODEL_NAME,'tracker':'BoT-SORT; ReID disabled',
      'methods':'Directed finite-segment crossings, track-ID movement matching, stationary-track queue proxy',
      'entry_counts':{r.Approach:int(r.Incoming_candidate_count) for r in summary.itertuples()},
      'turning_matrix':{a:{b:int(mat.loc[a,b]) for b in ARMS} for a in ARMS},
      'approach_statistics':summary.replace({np.nan:None}).to_dict('records'),
      'matched_movements':int(len(pairs_df)),
      'unmatched_entry_events':int(len(unmatched)),
      'warnings':['Provisional counts; gates are manually estimated and must be visually audited.',
                  'Camera shadows/occlusions and track fragmentation can affect counts.',
                  'South queue/exit may be partly obscured.',
                  'Waiting time here means stationary duration, not total travel delay.',
                  'Accuracy percentage requires manual ground-truth validation.']
    }
    with open(ROOT/'results.json','w') as f:json.dump(report,f,indent=2)

    readme=f"""# Traffic Intersection Video Analytics — Vinfotech Case Study

    Source: {os.path.basename(VIDEO)}
    Video duration: {DURATION:.1f} seconds, {W}×{H}, {FPS:.1f} source FPS.
    Processed approximately {FPS/FRAME_STRIDE:.1f} video FPS with {MODEL_NAME} and BoT-SORT (appearance ReID disabled).

    ## Outputs
    - annotated_traffic_desktop.mp4: tracked IDs, gates, path trails, and counts (generated when video supplied)\n- traffic_pipeline.py: VS Code / terminal runnable Python source\n- requirements.txt: Python environment dependencies
    - all_vehicle_trajectories.csv: raw frame, ID, bbox, class, confidence, bottom-center contact point
    - approach_entry_counts.csv: directed crossing candidates per entering approach
    - gate_crossing_events.csv: gate events (raw); gate_crossing_events_deduplicated.csv: conservative candidate dedup
    - turning_matrix_4x4.csv, turning_movements_matched.csv, unmatched_entries.csv
    - stopped_vehicle_episodes.csv, completed_stationary_waits.csv, queue_per_second.csv
    - intersection_summary.csv, results.json, manual_validation_sample.csv, figures and calibration_preview.jpg

    ## Methods and limitations
    The eight geometric gates and queue zones were manually approximated from a reference frame.
    A crossing must be in the specified direction, through the finite gate segment, within a continuous track.
    Turning entries are paired with exits only for the same track ID; movements without confirmed exits remain unmatched, not guessed.
    Queue is the number of tracked near-stationary vehicles in a queue polygon, NOT a measured physical queue length.
    Stationary time is computed only on observed multi-frame slow-motion episodes; censored waits are not treated as zero.
    YOLO does not guarantee perfect detection, and BoT-SORT does not guarantee uninterrupted IDs.
    The South camera view is limited. Validate all four directional measurements manually.

    ## Important disclosure
    This is historical Bellevue recording (2017), not a newly captured live camera, so it does not fully meet the live-source instruction.
    No measured accuracy (e.g., 70%) is claimed absent manually labelled ground truth.
    The camera-side coordinate configuration is in traffic_pipeline.py and should be verified by visual inspection.
    """
    (ROOT/'README.md').write_text(readme)

    writeup=f"""A fixed elevated video of a signalised intersection was analysed with {MODEL_NAME} for vehicle detection and BoT-SORT for identity association. To balance inference cost against temporal resolution, frames were processed at approximately {FPS/FRAME_STRIDE:.1f} frames per second while all events retained original video timestamps. The pipeline captured bounding boxes, confidence scores, classes, and vehicle contact points for offline analysis. Entry and exit counting used manually defined directional gates. A 4×4 movement matrix was calculated only when an entering vehicle and a later outgoing crossing shared a track ID. The program identified {len(pairs_df)} candidate completed turning movements, while {len(unmatched)} incoming gate events remained unmatched. Unmatched movements were not assigned destinations. Stationary-vehicle episodes were estimated using smoothed contact-point speeds inside four approach regions, producing queue proxies and available waiting-time samples. Results were exported as CSV, JSON, figures and an annotated video. Important limitations include partial occlusion of one approach, small distant vehicles, compression artefacts, and ID fragmentation. Gate positions and counting samples require manual video verification before final accuracy claims. Source note: {source_note or 'Recording source/date must be confirmed by submitter'}. The reported values are provisional computer-vision estimates rather than certified ground-truth traffic counts."""
    (ROOT/'case_study_200_word_summary.txt').write_text(writeup)
    print('Saved JSON, charts, README and summary.')
    print('Summary words:',len(writeup.split()))
    # Audit corrections must remain separate from raw detector output.
    # The reviewed figures only apply to this particular Bellevue video/CSV.
    if reviewed_audit:
        raw=ROOT/'intersection_summary_raw.csv'
        if not raw.exists():shutil.copy2(ROOT/'intersection_summary.csv',raw)
        audited={'NORTH':10,'EAST':9,'SOUTH':np.nan,'WEST':9}
        status={'NORTH':'Provisional','EAST':'Duplicate-filtered estimate',
                'SOUTH':'N/A - insufficient queue visibility','WEST':'Provisional'}
        summary['Max_observed_stationary_tracks']=summary.Approach.map(audited)
        summary['Queue_audit_status']=summary.Approach.map(status)
        summary.to_csv(ROOT/'intersection_summary.csv',index=False)
        summary[['Approach','Max_observed_stationary_tracks','Queue_audit_status']].to_csv(
            ROOT/'queue_peak_audit.csv',index=False)
        fig,ax=plt.subplots(figsize=(8,5))
        valid=summary.dropna(subset=['Max_observed_stationary_tracks'])
        ax.bar(valid.Approach,valid.Max_observed_stationary_tracks)
        ax.set_title('Audited Maximum Queue Candidates')
        ax.set_ylabel('Observed stationary vehicles (estimate)')
        ax.text(.5,-.16,'South: N/A (not reliably measured)',ha='center',transform=ax.transAxes)
        fig.tight_layout();fig.savefig(ROOT/'audited_queue_peaks.png',dpi=170,bbox_inches='tight')
        plt.close(fig)
        report['approach_statistics']=json.loads(summary.to_json(orient='records'))
        report['queue_peak_audit']={k:(None if pd.isna(v) else int(v)) for k,v in audited.items()}
        report['queue_note']='Original queue_per_second.csv and queue_timeline.png are raw stationary-ID proxies.'
        report['warnings'].append('South queue and stationary wait N/A due to insufficient visibility.')
        if manual_validation_csv:
            src=Path(manual_validation_csv)
            if src.exists():
                validation=pd.read_csv(src)
                n=len(validation)
                review_col='Manual_review' if 'Manual_review' in validation.columns else 'Visual_review'
                correct=int((validation[review_col].astype(str).str.lower()=='correct').sum())
                validation.to_csv(ROOT/'manual_validation_reviewed_12.csv',index=False)
                report['manual_validation']={
                    'sample_size':n,'correct_in_selected_sample':correct,
                    'note':'Selected-sample visual review, NOT an overall accuracy rate.'}
        (ROOT/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        with open(ROOT/'README.md','a',encoding='utf-8') as f:
            f.write("\n## Audited corrections\nNorth queue 10 (provisional), East 9 (duplicate-filtered), West 9 (provisional), South N/A.\nThe timeline and raw summary remain diagnostic, not physical queue ground truth.\nThe manual validation sample is not a population-wide accuracy measurement.\n")
    print('Analysis complete: outputs in',ROOT)
    return ROOT


def main():
    parser=argparse.ArgumentParser(description='Run YOLO + BoT-SORT, IN/OUT, turning, queues and annotated video',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video',type=Path,help='Path to MP4, e.g. C:\\Videos\\Traffic.mp4. Runs full inference by default.')
    parser.add_argument('--csv',type=Path,help='Reuse saved trajectory CSV: skip YOLO inference')
    parser.add_argument('--geometry',type=Path,default=Path(__file__).with_name('geometry_bellevue.json'),
                        help='Edit for NEW camera viewpoints before counting')
    parser.add_argument('--out',type=Path,default=Path('results'),help='Destination folder for all outputs')
    parser.add_argument('--model',default='yolo11s.pt',help='YOLO weights (downloaded on first use)')
    parser.add_argument('--imgsz',type=int,default=1280,help='YOLO inference resolution')
    parser.add_argument('--conf',type=float,default=.20,help='YOLO confidence threshold')
    parser.add_argument('--stride',type=int,default=2,help='Process every Nth original video frame')
    parser.add_argument('--skip-video',action='store_true',help='Do not render annotated MP4')
    parser.add_argument('--preview-only',action='store_true',help='Draw and save gates/queue zones, then exit')
    parser.add_argument('--preview-time',type=float,default=60.0)
    parser.add_argument('--min-display-confidence',type=float,default=0.0)
    parser.add_argument('--exclude-json',type=Path,help='Visual-only masking of false-positive display regions')
    parser.add_argument('--fps',type=float,default=30.0,help='FPS when no video supplied')
    parser.add_argument('--duration',type=float,default=None,help='Duration when no video supplied')
    parser.add_argument('--use-reviewed-audit',action='store_true',help='Use ONLY on the included, reviewed Bellevue recording')
    parser.add_argument('--validation-csv',type=Path,help='Manually reviewed vehicle samples')
    args=parser.parse_args()
    if args.stride<1:parser.error('--stride must be >=1')
    if not args.video and not args.csv: parser.error('Provide --video for full run, or --csv for offline reproduction')
    if args.video and not args.video.exists():parser.error(f'Video not found: {args.video}')
    if not args.geometry.exists():parser.error(f'Geometry JSON not found: {args.geometry}')
    args.out.mkdir(parents=True,exist_ok=True)
    config=load_road_geometry(args.geometry)
    shutil.copy2(args.geometry,args.out/'geometry_used.json')
    if args.video:
        fps,w,h,total,duration=video_metadata(args.video)
        print(f'Video: {args.video} | {w}x{h}, {fps:.2f}fps, {duration:.2f}s')
        calibration_preview(args.video,args.out,args.preview_time)
        print('IMPORTANT: Check the calibration preview. New camera angles require editing the geometry JSON.')
    if args.preview_only:
        if not args.video:parser.error('--preview-only requires --video')
        return
    if args.csv:
        track_csv=args.csv
        if not track_csv.exists():parser.error(f'Existing CSV not found: {track_csv}')
        local_csv=args.out/'all_vehicle_trajectories.csv'
        if track_csv.resolve()!=local_csv.resolve():shutil.copy2(track_csv,local_csv)
        track_csv=local_csv
        print('REUSE MODE: loading existing tracks; YOLO inference skipped.')
    else:
        print(f'FULL MODE: YOLO({args.model}) + BoT-SORT on {args.video}')
        track_csv=infer_tracks(args.video,args.out,model_name=args.model,
                               stride=args.stride,imgsz=args.imgsz,conf=args.conf)
    if not track_csv.exists():parser.error(f'Trajectory CSV missing: {track_csv}')
    analyze_tracks(track_csv,args.out,video=args.video,fps=args.fps,duration=args.duration,
                   stride=args.stride,reviewed_audit=args.use_reviewed_audit,
                   manual_validation_csv=args.validation_csv,model_name=args.model,
                   queue_unavailable=config.get('queue_unavailable',[]),
                   source_note=config.get('source_note'))
    if args.video and not args.skip_video:
        annotated_video(args.video,track_csv,args.out,stride=args.stride,
                        min_conf=args.min_display_confidence,exclude_json=args.exclude_json,
                        queue_unavailable=config.get('queue_unavailable',[]),
                   source_note=config.get('source_note'))
    print('\nDONE. Results directory:',args.out.resolve())
    print('Main files: all_vehicle_trajectories.csv (generated for full runs), annotated_traffic_desktop.mp4,')
    print('approach_in_out_counts.csv, turning_matrix_4x4.csv, intersection_summary.csv,')
    print('completed_stationary_waits.csv, queue_per_second.csv, vehicle_routes.csv')

if __name__=='__main__':
    main()
