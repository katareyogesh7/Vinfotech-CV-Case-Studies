"""Rule-based gear assembly verification using OpenCV (CPU only).

Commands (from project folder):
  python assembly_check.py --folder candidate_pack/test_images
  python assembly_check.py --image candidate_pack/test_images/test_01.png

The detection thresholds are calibrated from the three known-good images.
This is not a trained AI model. See README.md for limitations.
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

# Every output image is re-created on each run; source images are untouched.
PROJECT = Path(__file__).resolve().parent
OUTPUT = PROJECT / 'output'
PLATE_W, PLATE_H = 1000, 700

# Approximate positions in the STRAIGHTENED blue-plate coordinate system.
# Define once; do not define separate positions for the six test images.
CORNER_SCREWS = {
    'Top Left': (45, 45), 'Top Right': (955, 45),
    'Bottom Left': (45, 655), 'Bottom Right': (955, 655),
}
BEARING_SCREWS = {
    'Upper': (550, 149), 'Left': (475, 326),
    'Right': (687, 403), 'Lower': (592, 542),
}
MOTOR_SCREWS = {
    'Upper': (718, 177), 'Lower': (879, 270),
}
MOTOR_REGIONS = {
    'Cable Connector': (790, 90, 860, 145),
    'Motor Pinion': (785, 150, 880, 245),
    'Red Cable': (770, 0, 925, 140),
}
GREEN, RED = (0, 220, 0), (0, 0, 235)


def read_image(path):
    """Read an image, including Windows paths with non-English characters."""
    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f'Cannot decode image: {path}')
    return image


def save_image(path, image):
    """Save PNG without OpenCV's Windows Unicode-path limitation."""
    success, encoded = cv2.imencode('.png', image)
    if not success:
        raise RuntimeError(f'Cannot encode image: {path}')
    encoded.tofile(str(path))


def data_directory():
    """Accept a normal extracted ZIP OR the extra candidate_pack nesting."""
    options = [PROJECT/'candidate_pack', PROJECT/'candidate_pack'/'candidate_pack']
    for folder in options:
        if (folder/'reference_good').is_dir() and (folder/'test_images').is_dir():
            return folder
    raise FileNotFoundError('candidate_pack must contain reference_good and test_images')


def find_golden_gear(image):
    """Detect the brass gear from its HSV colour, then find its centre."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gold = cv2.inRange(hsv, np.array([12,35,70]), np.array([38,255,255]))
    gold = cv2.morphologyEx(gold, cv2.MORPH_OPEN, np.ones((5,5),np.uint8))
    contours, _ = cv2.findContours(gold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Golden output gear not found')
    gear = max(contours, key=cv2.contourArea)
    if cv2.contourArea(gear) < 20000:
        raise ValueError('Golden output gear uncertain')
    moments = cv2.moments(gear)
    if moments['m00'] == 0:
        raise ValueError('Invalid output gear contour')
    cx = int(moments['m10']/moments['m00'])
    cy = int(moments['m01']/moments['m00'])
    _, _, w, h = cv2.boundingRect(gear)
    return cx, cy, min(w, h)


def straighten_plate(image):
    """Locate blue plate and warp it to a fixed 1000x700 plane."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    blue = cv2.inRange(hsv, np.array([95,40,40]), np.array([115,160,220]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13,13))
    blue = cv2.morphologyEx(blue, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError('Blue base plate not found')
    plate = max(contours, key=cv2.contourArea)
    if cv2.contourArea(plate) < 100000:
        raise ValueError('Blue base plate uncertain')

    corners = cv2.boxPoints(cv2.minAreaRect(plate)).astype(np.float32)
    top = corners[np.argsort(corners[:,1])[:2]]
    bottom = corners[np.argsort(corners[:,1])[2:]]
    tl,tr = top[np.argsort(top[:,0])]
    bl,br = bottom[np.argsort(bottom[:,0])]
    source = np.float32([tl,tr,br,bl])
    target = np.float32([[0,0],[PLATE_W-1,0],
                         [PLATE_W-1,PLATE_H-1],[0,PLATE_H-1]])
    transform = cv2.getPerspectiveTransform(source, target)
    return cv2.warpPerspective(image, transform, (PLATE_W,PLATE_H))


def measure(image):
    """Return raw colour/edge scores. NO test-image labels are used here."""
    cx,cy,size = find_golden_gear(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Output nut: look for golden pixels in a ring around expected nut.
    half = int(size*.17)
    nut_rect = (max(0,cx-half), max(0,cy-half),
                min(image.shape[1],cx+half), min(image.shape[0],cy+half))
    x1,y1,x2,y2 = nut_rect
    nut_hsv = hsv[y1:y2,x1:x2]
    nh,nw = nut_hsv.shape[:2]
    yy,xx = np.ogrid[:nh,:nw]
    radius = np.sqrt((xx-nw/2)**2+(yy-nh/2)**2)
    small = min(nw,nh)/2
    ring = (radius > .32*small)&(radius < .63*small)
    if not ring.any():
        raise ValueError('Output nut region is too small')
    nut_gold = cv2.inRange(nut_hsv, np.array([16,45,0]),np.array([37,255,255]))
    nut_score = float(100*np.mean(nut_gold[ring]>0))

    # Idler: expected centre relative to detected output gear.
    gy,gx = np.ogrid[:image.shape[0],:image.shape[1]]
    ix,iy = int(cx+.79*size),cy
    id2 = (gx-ix)**2+(gy-iy)**2
    grey = (hsv[:,:,1]<45)&(hsv[:,:,2]>60)
    idler_gear = float(100*np.mean(grey[id2<(.28*size)**2]))
    idler_ring = (id2>(.055*size)**2)&(id2<(.12*size)**2)
    dark_blue = ((hsv[:,:,0]>90)&(hsv[:,:,0]<135)&
                 (hsv[:,:,1]>30)&(hsv[:,:,2]<135))
    idler_clip = float(100*np.mean(dark_blue[idler_ring]))

    # Compound: a grey ring and its separate dark-blue circlip blob.
    px,py = int(cx+1.06*size),int(cy-.46*size)
    d2 = (gx-px)**2+(gy-py)**2
    compound_ring = (d2>(.11*size)**2)&(d2<(.25*size)**2)
    compound_grey = ((hsv[:,:,1]<35)&(hsv[:,:,2]>110)&(hsv[:,:,2]<200))
    compound_gear = float(100*np.mean(compound_grey[compound_ring]))
    clip_pixels = cv2.inRange(hsv,np.array([108,40,50]),np.array([130,170,115]))
    clip_pixels = cv2.morphologyEx(clip_pixels,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    near_compound = (d2<(.25*size)**2).astype(np.uint8)*255
    clip_pixels = cv2.bitwise_and(clip_pixels,near_compound)
    n, labels, stats, centers = cv2.connectedComponentsWithStats(clip_pixels,8)
    largest_area = 0
    compound_clip_center = (px,py)
    for label in range(1,n):
        area = int(stats[label,cv2.CC_STAT_AREA])
        if area>largest_area:
            largest_area = area
            compound_clip_center = tuple(np.rint(centers[label]).astype(int))
    compound_clip = float(100*largest_area/(size*size))

    # Hardware uses the rectified blue plate, so screw locations are fixed.
    plate = straighten_plate(image)
    phsv = cv2.cvtColor(plate,cv2.COLOR_BGR2HSV)
    yy,xx = np.ogrid[:PLATE_H,:PLATE_W]
    screw_grey = ((phsv[:,:,1]<45)&(phsv[:,:,2]>40)&(phsv[:,:,2]<220))
    corner_scores = {}
    for name,(sx,sy) in CORNER_SCREWS.items():
        circle = (xx-sx)**2+(yy-sy)**2<20**2
        corner_scores[name] = float(100*np.mean(screw_grey[circle]))

    gray = cv2.cvtColor(plate,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray,(3,3),0),40,80)
    def edge_scores(positions,half):
        return {name: int(cv2.countNonZero(edges[y-half:y+half+1,
                                                   x-half:x+half+1]))
                for name,(x,y) in positions.items()}
    bearing_scores = edge_scores(BEARING_SCREWS,25)
    motor_screw_scores = edge_scores(MOTOR_SCREWS,20)

    h,s,v = cv2.split(phsv)
    masks = {
        'Cable Connector':(s<40)&(v>170),
        'Motor Pinion':(s<72)&(v>25)&(v<110),
        'Red Cable':((h<12)|(h>172))&(s>85)&(v>115),
    }
    motor_scores = {}
    for name,(a,b,c,d) in MOTOR_REGIONS.items():
        motor_scores[name] = float(100*np.mean(masks[name][b:d,a:c]))

    return dict(nut=nut_score, nut_rect=nut_rect, gold_center=(cx,cy),
                idler_gear=idler_gear, idler_clip=idler_clip,
                idler_circle=(ix,iy,int(.28*size)),
                compound_gear=compound_gear, compound_clip=compound_clip,
                compound_center=compound_clip_center,
                compound_gear_center=(px,py),compound_radius=int(.25*size),
                plate=plate,corner=corner_scores,bearing=bearing_scores,
                motor_screws=motor_screw_scores,motor=motor_scores)


def calibrate(reference_files):
    """Choose trial thresholds ONLY from known-good images."""
    measures = [measure(read_image(p)) for p in reference_files]
    return {
        'nut_max':max(m['nut'] for m in measures)+5,
        'idler_gear_min':min(m['idler_gear'] for m in measures)-15,
        'idler_clip_min':min(m['idler_clip'] for m in measures)-12,
        'compound_gear_min':min(m['compound_gear'] for m in measures)-12,
        'compound_clip_min':min(m['compound_clip'] for m in measures)*.70,
        'corner_min':min(x for m in measures for x in m['corner'].values())*.60,
        'bearing_min':{n:int(min(m['bearing'][n] for m in measures)*.65)
                       for n in BEARING_SCREWS},
        'motor_screw_min':{n:int(min(m['motor_screws'][n] for m in measures)*.60)
                           for n in MOTOR_SCREWS},
        'motor_min':{n:min(m['motor'][n] for m in measures)*ratio
                     for n,ratio in {'Cable Connector':.70,
                                     'Motor Pinion':.80,'Red Cable':.60}.items()},
    }


def classify(m,thresholds):
    """Return results for each inspected part, not just a single image score."""
    checks=[]
    def add(component,passed,score,limit,comparison,description):
        checks.append(dict(component=component,status='PASS' if passed else 'FAIL',
                           score=score,threshold=limit,rule=comparison,
                           issue='' if passed else description))
    # Locating a blue plate is required before measuring any screw.
    add('Blue base plate',True,None,None,'shape','Blue base plate not found')
    add('Output gear',True,None,None,'shape','Output gear not found')
    add('Output gear nut/washer',m['nut']<=thresholds['nut_max'],
        m['nut'],thresholds['nut_max'],'<=','Output gear nut/washer suspected missing')
    add('Idler gear',m['idler_gear']>=thresholds['idler_gear_min'],
        m['idler_gear'],thresholds['idler_gear_min'],'>=','Idler gear suspected missing')
    idler_present=checks[-1]['status']=='PASS'
    if idler_present:
        add('Idler circlip',m['idler_clip']>=thresholds['idler_clip_min'],
            m['idler_clip'],thresholds['idler_clip_min'],'>=','Idler circlip suspected missing')
    else:
        checks.append(dict(component='Idler circlip',status='SKIPPED',score=m['idler_clip'],
                           threshold=thresholds['idler_clip_min'],rule='>=',issue='Cannot inspect without idler gear'))
    add('Compound gear',m['compound_gear']>=thresholds['compound_gear_min'],
        m['compound_gear'],thresholds['compound_gear_min'],'>=','Compound gear: review')
    compound_present=checks[-1]['status']=='PASS'
    if compound_present:
        add('Compound circlip',m['compound_clip']>=thresholds['compound_clip_min'],
            m['compound_clip'],thresholds['compound_clip_min'],'>=','Compound circlip: review')
    else:
        checks.append(dict(component='Compound circlip',status='SKIPPED',score=m['compound_clip'],
                           threshold=thresholds['compound_clip_min'],rule='>=',issue='Cannot inspect without compound gear'))
    for name,score in m['corner'].items():
        add('Corner bolt: '+name, score>=thresholds['corner_min'],score,
            thresholds['corner_min'],'>=','Corner bolt '+name+' suspected missing')
    for name,score in m['bearing'].items():
        add('Bearing screw: '+name,score>=thresholds['bearing_min'][name],score,
            thresholds['bearing_min'][name],'>=','Bearing screw '+name+' suspected missing')
    for name,score in m['motor_screws'].items():
        add('Motor mounting screw: '+name,score>=thresholds['motor_screw_min'][name],score,
            thresholds['motor_screw_min'][name],'>=','Motor mounting screw '+name+' suspected missing')
    for name,score in m['motor'].items():
        add(name,score>=thresholds['motor_min'][name],score,
            thresholds['motor_min'][name],'>=',name+' needs inspection')
    return checks


def annotate(image,m,checks):
    """Make an annotated original, hardware view, and side-by-side view."""
    lookup = {c['component']:c['status'] for c in checks}
    ok = lambda name: lookup.get(name) == 'PASS'
    problems = [c['issue'] for c in checks if c['status']=='FAIL']
    verdict = 'NOT OK' if problems else 'OK'
    color = RED if problems else GREEN

    original = image.copy()
    x1,y1,x2,y2=m['nut_rect']
    cv2.rectangle(original,(x1,y1),(x2,y2),GREEN if ok('Output gear nut/washer') else RED,3)
    ix,iy,ir=m['idler_circle']
    cv2.circle(original,(ix,iy),ir,GREEN if ok('Idler gear') else RED,3)
    # Small inner circle marks the idler circlip independently.
    clip_color = GREEN if ok('Idler circlip') else RED
    cv2.circle(original,(ix,iy),int(ir*.42),clip_color,2)
    px,py=m['compound_gear_center']
    cv2.circle(original,(px,py),m['compound_radius'],GREEN if ok('Compound gear') else RED,3)
    ccx,ccy=m['compound_center']
    cv2.circle(original,(ccx,ccy),int(m['compound_radius']*.4),
               GREEN if ok('Compound circlip') else RED,2)
    cv2.putText(original,verdict,(25,43),cv2.FONT_HERSHEY_SIMPLEX,1.0,color,2)
    for i,text in enumerate(problems[:4]):
        cv2.putText(original,text[:65],(25,78+i*30),
                    cv2.FONT_HERSHEY_SIMPLEX,.65,RED,2)

    hardware=m['plate'].copy()
    # Blue base plate presence was checked by the automatic rectification.
    cv2.rectangle(hardware,(0,0),(PLATE_W-1,PLATE_H-1),GREEN,3)
    for name,(x,y) in CORNER_SCREWS.items():
        cv2.circle(hardware,(x,y),27,GREEN if ok('Corner bolt: '+name) else RED,3)
    for name,(x,y) in BEARING_SCREWS.items():
        cv2.rectangle(hardware,(x-25,y-25),(x+25,y+25),
                      GREEN if ok('Bearing screw: '+name) else RED,2)
    for name,(x,y) in MOTOR_SCREWS.items():
        cv2.circle(hardware,(x,y),24,GREEN if ok('Motor mounting screw: '+name) else RED,2)
    for name,(a,b,c,d) in MOTOR_REGIONS.items():
        cv2.rectangle(hardware,(a,b),(c,d),GREEN if ok(name) else RED,2)
    hardware_checks = [c for c in checks if c['component'].startswith(('Corner bolt:', 'Bearing screw:', 'Motor mounting screw:')) or c['component'] in MOTOR_REGIONS]
    hardware_color = RED if any(c['status']=='FAIL' for c in hardware_checks) else GREEN
    cv2.putText(hardware,'HARDWARE CHECKS',(280,39),cv2.FONT_HERSHEY_SIMPLEX,
                .7,hardware_color,2)

    # Put both views next to each other; no GUI windows are required.
    left=cv2.resize(original,(800,600))
    right=cv2.resize(hardware,(800,600))
    both=np.full((650,1600,3),255,np.uint8)
    both[50:650,0:800]=left
    both[50:650,800:1600]=right
    cv2.putText(both,'GEARS + NUT + CIRCLIPS',(200,34),
                cv2.FONT_HERSHEY_SIMPLEX,.8,(30,30,30),2)
    cv2.putText(both,'SCREWS + CONNECTOR + MOTOR',(880,34),
                cv2.FONT_HERSHEY_SIMPLEX,.8,(30,30,30),2)
    return verdict,problems,original,hardware,both


def find_images(folder):
    """Case-insensitive search for PNG/JPG images in one folder."""
    return sorted([p for p in folder.iterdir() if p.is_file() and
                   p.suffix.lower() in ('.png','.jpg','.jpeg')])


def main():
    parser=argparse.ArgumentParser(description='Gear assembly inspection (classical OpenCV).')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--image',help='Inspect one image (path to PNG/JPG).')
    group.add_argument('--folder',help='Inspect every PNG/JPG in a folder.')
    args=parser.parse_args()

    data=data_directory()
    reference_files=find_images(data/'reference_good')
    if len(reference_files)!=3:
        parser.error('Expected exactly 3 reference images in candidate_pack/reference_good')

    # Relative paths are accepted from either current terminal location or project folder.
    def resolve(raw):
        p=Path(raw)
        if p.exists():return p.resolve()
        p=PROJECT/raw
        if p.exists():return p.resolve()
        raise FileNotFoundError(raw)

    if args.image:
        images=[resolve(args.image)]
        if images[0].suffix.lower() not in ('.png','.jpg','.jpeg'):
            parser.error('Image must be PNG or JPG')
    else:
        folder=resolve(args.folder) if args.folder else data/'test_images'
        if not folder.is_dir():
            parser.error('Test folder does not exist')
        images=find_images(folder)
    if not images:
        parser.error('No test images found')

    print('Calibrating from 3 known-good reference images...')
    thresholds=calibrate(reference_files)
    # Dedicated output only: clear old generated files on every run.
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    OUTPUT.mkdir(parents=True)
    (OUTPUT/'thresholds.json').write_text(json.dumps(thresholds,indent=2),encoding='utf-8')

    summary=[]
    component_rows=[]
    thumbnails=[]
    print('\nIMAGE                 VERDICT     FLAGGED COMPONENTS')
    print('-'*84)
    for file in images:
        try:
            image=read_image(file)
            measures=measure(image)
            checks=classify(measures,thresholds)
            verdict,problems,assembly,hardware,both=annotate(image,measures,checks)
            stem=file.stem
            save_image(OUTPUT/(stem+'_assembly.png'),assembly)
            save_image(OUTPUT/(stem+'_hardware.png'),hardware)
            save_image(OUTPUT/(stem+'_all_checks.png'),both)
            thumbs=cv2.resize(both,(800,325))
            thumbnails.append(thumbs)
            summary.append([file.name,verdict,'; '.join(problems) or 'No detected issues in checked parts'])
            for c in checks:
                component_rows.append([file.name,c['component'],c['status'],
                    '' if c['score'] is None else round(c['score'],3),
                    '' if c['threshold'] is None else round(c['threshold'],3),
                    c['rule'],c['issue']])
        except (ValueError, OSError, cv2.error, ZeroDivisionError) as exc:
            verdict='NOT OK'
            note='Inspection could not be completed: '+str(exc)
            summary.append([file.name,verdict,note])
            print('WARNING:',file.name,note)
        print(f'{file.name:<22}{verdict:<12}{summary[-1][2]}')

    with (OUTPUT/'inspection_report.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        writer=csv.writer(fp)
        writer.writerow(['Image','Verdict','Flagged components / notes'])
        writer.writerows(summary)
    with (OUTPUT/'component_details.csv').open('w',newline='',encoding='utf-8-sig') as fp:
        writer=csv.writer(fp)
        writer.writerow(['Image','Component','Status','Measured score','Trial threshold','Rule','Reason'])
        writer.writerows(component_rows)
    # A grid of all inspected images; generated automatically even in a terminal.
    if thumbnails:
        columns=2
        rows=(len(thumbnails)+columns-1)//columns
        grid=np.full((rows*325,columns*800,3),240,np.uint8)
        for i,thumb in enumerate(thumbnails):
            y,x=divmod(i,columns)
            grid[y*325:(y+1)*325,x*800:(x+1)*800]=thumb
        save_image(OUTPUT/'summary_grid.png',grid)
    print('\nSaved results to:',OUTPUT)
    print('Rerun = old output folder is replaced.')
    print('NOTE: Rule-based prototype; see README.md for coverage and limitations.')


if __name__=='__main__':
    main()
