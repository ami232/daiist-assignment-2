"""Standalone reconstruction from supplied aggregate counts, not original project code.

Python 3 standard library only. Run: python3 recompute.py
Writes the full 24-threshold / 300-band sweep, summary, and sensitivity outputs.
All monetary figures are EUR; rates are fractions of incoming real objects.
"""
from pathlib import Path
import csv
import json
import math

ROOT = Path(__file__).resolve().parent
with (ROOT / 'threshold_counts.csv').open() as f:
    COUNTS = [{k: float(v) if k == 'threshold' else int(v)
               for k, v in row.items()} for row in csv.DictReader(f)]

# Same inputs as costing.py (sources in REPORT.md). The first-pass scenario used
# cd=.77, cs=19.8875, pd=.015, hf=.03; those results are recorded in REPORT.md's history.
BASE = dict(cd=.05*(3.85+22.5*1.1665), cs=.05*(3.85+45*1.1665+.75*47.9), fp=.06,
            cr=47.9*3/(3600*.8),
            pd=.010, ps=.0005, hm=.15, hf=.10, n=30000., hours=8.,
            seconds=3., utilisation=.8, reviewers=1, ctp=0., ctn=0.)
PAIRS = [(i,j) for i in range(len(COUNTS)) for j in range(i,len(COUNTS))]

def policy(pair, p=BASE):
    i,j = pair
    low, high = COUNTS[i], COUNTS[j]
    ad, ass = high['defects']/31, high['stones']/93
    ag = (high['good']+high['strays'])/2922
    vd = (low['defects']-high['defects'])/31
    vs = (low['stones']-high['stones'])/93
    vg = (low['good']+low['strays']-high['good']-high['strays'])/2922
    # Strays follow the supplied convention: extra good-equivalent events.
    pg = 1-p['pd']-p['ps']
    md, ms = 1-ad-vd+vd*p['hm'], 1-ass-vs+vs*p['hm']
    fg = ag+vg*p['hf']
    q = p['pd']*vd+p['ps']*vs+pg*vg
    defect = 1000*p['pd']*md*p['cd']
    stone = 1000*p['ps']*ms*p['cs']
    false = 1000*pg*fg*p['fp']
    review = 1000*q*p['cr']
    # Optional action-specific sensitivity only; zero in all base results.
    real_good_false = (high['good']+(low['good']-high['good'])*p['hf'])/2922
    correct = 1000*((p['pd']*(1-md)+p['ps']*(1-ms))*p['ctp']
                    +pg*(1-real_good_false)*p['ctn'])
    cost = defect+stone+false+review+correct
    people = q*p['n']*p['seconds']/(3600*p['utilisation'])
    return dict(low=low['threshold'], high=high['threshold'], cost=cost,
                cost_day=cost*p['n']*p['hours']/1000,
                review_share=q, reviews_hour=q*p['n'], people=people,
                feasible=people <= p['reviewers']+1e-10,
                defect_cost=defect, stone_cost=stone, false_cost=false,
                review_cost=review, correct_cost=correct,
                missed_defects_per_1000=1000*p['pd']*md,
                missed_stones_per_1000=1000*p['ps']*ms,
                false_events_per_1000=1000*pg*fg)

def sweep(p=BASE, mode='capped', strict=False):
    candidates = PAIRS if mode!='auto' else [(i,i) for i in range(len(COUNTS))]
    rows = [policy(pair,p) for pair in candidates]
    if mode=='capped': rows=[r for r in rows if r['feasible']]
    if strict: rows=[r for r in rows if r['review_share']>0]
    return min(rows,key=lambda r:(r['cost'],r['review_share'],r['low'],r['high']))

def write_csv(name, rows):
    with (ROOT/name).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def baseline_alternatives(p):
    pg=1-p['pd']-p['ps']
    nothing=1000*(p['pd']*p['cd']+p['ps']*p['cs'])
    human=1000*(p['pd']*p['hm']*p['cd']+p['ps']*p['hm']*p['cs']
                +pg*p['hf']*p['fp']+p['cr'])
    return dict(do_nothing=nothing,human_all=human,
                human_people=p['n']*p['seconds']/(3600*p['utilisation']))

def stable_interval(key, maximum, mode):
    """Exact local stability interval of base optimum for affine scalar inputs.

    For each candidate, find its cost crossing against the base winner and its
    capacity crossing. Partition at every such crossing; test interval interiors.
    All listed scalar inputs make cost and capacity affine when varied alone.
    """
    winner=sweep(BASE,mode)
    pair0=next(pair for pair in PAIRS if (COUNTS[pair[0]]['threshold'],
         COUNTS[pair[1]]['threshold'])==(winner['low'],winner['high']))
    pairs=PAIRS if mode!='auto' else [(i,i) for i in range(len(COUNTS))]
    p0=dict(BASE,**{key:0});p1=dict(BASE,**{key:1})
    c0=policy(pair0,p0)['cost'];c1=policy(pair0,p1)['cost']
    points={0.,float(maximum)}
    for pair in pairs:
        r0,r1=policy(pair,p0),policy(pair,p1)
        intercept=r0['cost']-c0
        slope=(r1['cost']-c1)-intercept
        if abs(slope)>1e-12:
            x=-intercept/slope
            if 0<x<maximum:points.add(x)
        if mode=='capped':
            a=r0['people']-BASE['reviewers'];b=r1['people']-r0['people']
            if abs(b)>1e-12:
                x=-a/b
                if 0<x<maximum:points.add(x)
    pts=sorted(points)
    intervals=[]
    for a,b in zip(pts,pts[1:]):
        p=dict(BASE,**{key:(a+b)/2})
        r=sweep(p,mode)
        same=(r['low'],r['high'])==(winner['low'],winner['high'])
        intervals.append((a,b,same,r))
    idx=next(i for i,(a,b,_,_) in enumerate(intervals) if a-1e-12<=BASE[key]<=b+1e-12)
    start=end=idx
    while start>0 and intervals[start-1][2]:start-=1
    while end+1<len(intervals) and intervals[end+1][2]:end+=1
    def label(r):return f"[{r['low']:g},{r['high']:g})"
    return dict(input=key,mode=mode,base=BASE[key],lower=intervals[start][0],
        upper=intervals[end][1],tested_maximum=maximum,
        below=label(intervals[start-1][3]) if start else 'none within domain',
        above=label(intervals[end+1][3]) if end+1<len(intervals) else 'none within domain')

def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n
    mid=(p+z*z/(2*n))/d
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [mid-half,mid+half]

def main():
    assert len(COUNTS)==24 and len(PAIRS)==300
    for key in ('defects','stones','good','strays'):
        assert all(a[key]>=b[key] for a,b in zip(COUNTS,COUNTS[1:]))
    # Independent reference targets provided in the uploaded instructions.
    placeholder=dict(BASE,cd=1.5,cs=25,fp=.08,cr=.025,pd=.01,ps=.002,
                     hm=.1,hf=.02,n=50000,hours=16,seconds=3,utilisation=.75,reviewers=2)
    pa,pb=sweep(placeholder,'auto'),sweep(placeholder)
    assert pa['low']==.03 and abs(pa['cost']-7.302276)<1e-6
    assert (pb['low'],pb['high'])==(.02,.06) and abs(pb['cost']-6.606594)<1e-6
    active=dict(BASE,cr=47.9*3/3600)
    strict=sweep(BASE,strict=True)
    dedicated=[]
    for pair in PAIRS:
        r=policy(pair)
        if r['review_share']>0 and r['feasible']:
            r=dict(r);r['cost']=r['cost']-r['review_cost']+1000*47.9/BASE['n']
            r['cost_day']=r['cost']*BASE['n']*BASE['hours']/1000
            dedicated.append(r)
    # Dedicated result's review_cost below remains allocation figure; expose
    # explicit paid-shift cost separately so users cannot misread components.
    dedicated_best=min(dedicated,key=lambda r:r['cost'])
    dedicated_best['review_cost']=1000*47.9/BASE['n']
    summary=dict(assumptions=BASE,auto=sweep(BASE,'auto'),capped=sweep(),
        uncapped=sweep(BASE,'uncapped'),strict_nonempty=strict,
        active_time_best=sweep(active),dedicated_nonempty=dedicated_best,
        alternatives=baseline_alternatives(BASE),
        active_time_alternatives=baseline_alternatives(active),
        placeholder_check=dict(auto=pa,band=pb),
        wilson=dict(defect=wilson(25,31),stone=wilson(90,93)))
    write_csv('model_sweep.csv',[policy((i,i)) for i in range(24)])
    write_csv('band_sweep.csv',[policy(pair) for pair in PAIRS])
    limits=dict(cd=100,cs=10000,fp=10,cr=1,pd=1-BASE['ps'],ps=1-BASE['pd'],
                hm=1,hf=1,ctp=100,ctn=10)
    sensitivities=[stable_interval(k,v,m) for m in ['auto','capped'] for k,v in limits.items()]
    write_csv('sensitivity.csv',sensitivities)
    tests=[]
    for key,values in dict(cd=[.385,.77,1.54],cs=[5,19.8875,100,200],
        fp=[.03,.052,.06,.068,.12],cr=[.02,47.9*3/3600,BASE['cr'],.08],
        pd=[.005,.015,.03],ps=[.0001,.0005,.002,.01],hm=[0,.15,.3],
        hf=[0,.03,.1],reviewers=[0,1,2,4],n=[15000,30000,60000],
        hours=[4,8,16],ctp=[0,.01,.1,1],ctn=[0,.005,.01,.03]).items():
        for val in values:
            p=dict(BASE,**{key:val});r=sweep(p)
            tests.append(dict(input=key,value=val,**r))
    # Coupled labour/capacity sweeps recompute Cr when duration or utilisation changes.
    for key,values in dict(seconds=[1.5,2,2.47518493,3,4.5],
                           utilisation=[.6,.8,.97,1.]).items():
        for val in values:
            p=dict(BASE,**{key:val});p['cr']=47.9*p['seconds']/(3600*p['utilisation'])
            tests.append(dict(input=key,value=val,**sweep(p)))
    write_csv('stress_cases.csv',tests)
    (ROOT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    (ROOT/'assumptions.json').write_text(json.dumps(BASE,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['auto','strict_nonempty','active_time_best',
          'dedicated_nonempty','alternatives','wilson']},indent=2))

if __name__=='__main__':main()
