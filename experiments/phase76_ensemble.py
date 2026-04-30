"""
Phase 76: Multi-Strategy Ensemble
====================================
Portfolio (P60) x Model Soup (P69) ultimate combination.
3 strategies x K=3 soups = 9 total adapted models.
Test majority vote across all 9.
3 seeds.
"""
import os,sys,json,time,copy,random
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from collections import deque, Counter
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','training')
RESULTS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
FIGURES_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'figures')
os.makedirs(RESULTS_DIR,exist_ok=True);os.makedirs(FIGURES_DIR,exist_ok=True)
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10

def extract_objects(grid):
    arr=np.array(grid);h,w=arr.shape;vis=np.zeros_like(arr,dtype=bool)
    bg=int(np.bincount(arr.flatten()).argmax());objs=[]
    for r in range(h):
        for c in range(w):
            if not vis[r,c] and arr[r,c]!=bg:
                col=int(arr[r,c]);px=[];q=deque([(r,c)]);vis[r,c]=True
                while q:
                    cr,cc=q.popleft();px.append((cr,cc))
                    for dr,dc in[(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not vis[nr,nc] and arr[nr,nc]==col:vis[nr,nc]=True;q.append((nr,nc))
                rs=[p[0]for p in px];cs=[p[1]for p in px]
                objs.append({'color':col,'area':len(px),'center':(np.mean(rs),np.mean(cs)),'bbox':(min(rs),min(cs),max(rs),max(cs))})
    return objs,bg

def obj_feats(o,h,w):
    f=[0.]*10;f[min(o['color'],9)]=1.;cr,cc=o['center']
    f.extend([cr/max(h,1),cc/max(w,1),o['area']/max(h*w,1)])
    r0,c0,r1,c1=o['bbox'];f.extend([(r1-r0+1)/max(h,1),(c1-c0+1)/max(w,1),(c1-c0+1)/max(r1-r0+1,1)])
    return f[:NODE_FEAT_DIM]

def extract_op(inp,out):
    ia,oa=np.array(inp),np.array(out);objs,bg=extract_objects(inp)
    if ia.shape==oa.shape and np.array_equal(ia,oa):return 1,0,0,0
    if oa.size>0 and len(np.unique(oa))==1:return 2,0,int(np.unique(oa)[0]),0
    if ia.shape==oa.shape:
        d=ia!=oa
        if d.any():
            oc,nc=set(ia[d].tolist()),set(oa[d].tolist())
            if len(oc)==1 and len(nc)==1:
                o=int(list(oc)[0]);p=0
                for j,obj in enumerate(objs):
                    if obj['color']==o:p=j;break
                return 5,min(p,MAX_OBJECTS-1),o,int(list(nc)[0])
        if np.array_equal(np.flipud(ia),oa):return 6,0,0,0
        if np.array_equal(np.fliplr(ia),oa):return 7,0,0,0
    return 3,0,0,0

def prep(inp,out):
    objs,bg=extract_objects(inp);h,w=len(inp),len(inp[0])
    if not objs:return None
    nf=np.zeros((MAX_OBJECTS,NODE_FEAT_DIM),dtype=np.float32)
    n=min(len(objs),MAX_OBJECTS)
    for j in range(n):f=obj_feats(objs[j],h,w);nf[j,:len(f)]=f[:NODE_FEAT_DIM]
    op,ptr,c1,c2=extract_op(inp,out)
    return{'nf':torch.tensor(nf).unsqueeze(0).float(),'nn':torch.tensor([n]),
           'op':torch.tensor([op]),'c1':torch.tensor([c1]),'c2':torch.tensor([c2]),
           'ptr':torch.tensor([min(ptr,max(n-1,0))])}

def augment_pair(inp,out):
    ia,oa=np.array(inp),np.array(out);aug=[(inp,out)]
    if ia.shape==oa.shape:
        for k in[1,2,3]:aug.append((np.rot90(ia,k).tolist(),np.rot90(oa,k).tolist()))
        aug.append((np.flipud(ia).tolist(),np.flipud(oa).tolist()))
        aug.append((np.fliplr(ia).tolist(),np.fliplr(oa).tolist()))
    for _ in range(2):
        perm=list(range(10));random.shuffle(perm)
        ai=np.vectorize(lambda x:perm[x] if x<10 else x)(ia)
        ao=np.vectorize(lambda x:perm[x] if x<10 else x)(oa)
        aug.append((ai.tolist(),ao.tolist()))
    return aug

class Agent(nn.Module):
    def __init__(s,hid=64):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)
    def forward(s,nf,nn_c):
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1);h=s.ne(nf)
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g2(torch.cat([h,msg.expand_as(h)],-1));h=s.n2(h)*mf
        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        pl=((s.pq(g).unsqueeze(1))*s.pk(h)).sum(-1).masked_fill(~mask,-1e9)
        return s.oh(g),s.c1h(g),s.c2h(g),pl

def compute_loss(model,s):
    ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
    return(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
           F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))

def adapt_model(model,task_samples,steps=100,lr=1e-2):
    if not task_samples:return model
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am

def ablate_l2_random(model,rate,task_samples,seed=0):
    torch.manual_seed(seed)
    am=copy.deepcopy(model);am.train()
    total_loss=torch.tensor(0.0,device=DEVICE)
    for s in task_samples[:8]:total_loss=total_loss+compute_loss(am,s)
    total_loss=total_loss/max(len(task_samples[:8]),1);total_loss.backward()
    l2_params=set()
    for p in am.g2.parameters():l2_params.add(id(p))
    for p in am.n2.parameters():l2_params.add(id(p))
    with torch.no_grad():
        for p in am.parameters():
            if id(p) not in l2_params:continue
            if p.grad is not None:
                imp=p.grad.abs();noise=torch.rand_like(imp)*imp.mean()*0.5
                ni=imp+noise;thr=torch.quantile(ni.flatten(),rate)
                p.mul_((ni>thr).float())
    am.eval();return am

def model_soup(models):
    soup=copy.deepcopy(models[0])
    with torch.no_grad():
        for p in soup.parameters():p.zero_()
        for m in models:
            for ps,pm in zip(soup.parameters(),m.parameters()):ps.add_(pm/len(models))
    soup.eval();return soup

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 76: Multi-Strategy Ensemble")
    print("Portfolio x Model Soup = ultimate ensemble")
    print("=" * 60)
    t0 = time.time()
    tasks = load_arc_tasks(DATA_DIR)
    split = int(len(tasks) * 0.8)
    test_tasks = tasks[split:]

    # 3 strategies from P60
    strategies = {
        'fortress': {'l2_rate': 0.10, 'lr': 0.01, 'steps': 50},   # Low risk
        'standard': {'l2_rate': 0.20, 'lr': 0.1, 'steps': 100},    # Optimal
        'gambler':  {'l2_rate': 0.30, 'lr': 0.1, 'steps': 150},    # High risk
    }
    K = 3  # Soups per strategy
    N_SEEDS = 3

    methods = {'single_best': [], 'per_strat_soup': [], 'cross_strat_vote': [], 'full_9way': []}

    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        random.seed(seed*1000); np.random.seed(seed*1000); torch.manual_seed(seed*1000)

        all_samples = []
        for task in tasks:
            for p in task.get('train', []):
                s = prep(p['input'], p['output'])
                if s: all_samples.append(s)

        model = Agent().to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for ep in range(80):
            model.train(); random.shuffle(all_samples)
            for s in all_samples:
                loss = compute_loss(model, s)
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()

        single_ok, soup_ok, vote_ok, full_ok, total = 0, 0, 0, 0, 0

        for task in test_tasks:
            demos = task.get('train', [])
            aug_pairs = []
            for p in demos: aug_pairs.extend(augment_pair(p['input'], p['output']))
            aug_samples = [prep(ai, ao) for ai, ao in aug_pairs]
            aug_samples = [s for s in aug_samples if s]

            all_preds = []  # All 9 predictions
            strat_soups = []  # 3 soup predictions

            for sname, scfg in strategies.items():
                strat_models = []
                strat_preds = []
                for k in range(K):
                    ablated = ablate_l2_random(model, scfg['l2_rate'], aug_samples, seed=seed*100+k)
                    adapted = adapt_model(ablated, aug_samples, steps=scfg['steps'], lr=scfg['lr'])
                    strat_models.append(adapted)
                    # Get prediction
                    for tp in task.get('test', []):
                        s = prep(tp['input'], tp['output'])
                        if s is None: continue
                        with torch.no_grad():
                            ol,cl1,cl2,pl = adapted(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                        pred = (ol.argmax(1).item(), cl1.argmax(1).item(), cl2.argmax(1).item(), pl.argmax(1).item())
                        strat_preds.append(pred)
                        all_preds.append(pred)
                        break  # First test pair

                # Per-strategy soup
                souped = model_soup(strat_models)
                for tp in task.get('test', []):
                    s = prep(tp['input'], tp['output'])
                    if s is None: continue
                    with torch.no_grad():
                        ol,cl1,cl2,pl = souped(s['nf'].to(DEVICE), s['nn'].to(DEVICE))
                    pred = (ol.argmax(1).item(), cl1.argmax(1).item(), cl2.argmax(1).item(), pl.argmax(1).item())
                    strat_soups.append(pred)
                    break

            # Ground truth
            for tp in task.get('test', []):
                s_gt = prep(tp['input'], tp['output'])
                if s_gt is None: continue
                gt = (s_gt['op'].item(), s_gt['c1'].item(), s_gt['c2'].item(), s_gt['ptr'].item())
                total += 1

                # Single best (standard strategy, first model)
                if len(all_preds) >= 4:
                    if all_preds[3] == gt: single_ok += 1  # Standard strat, model 0

                # Per-strategy soup vote
                if strat_soups:
                    soup_majority = Counter(strat_soups).most_common(1)[0][0]
                    if soup_majority == gt: soup_ok += 1

                # Cross-strategy vote (one model per strat)
                cross_preds = [all_preds[i*K] for i in range(min(3, len(all_preds)//K))]
                if cross_preds:
                    cross_majority = Counter(cross_preds).most_common(1)[0][0]
                    if cross_majority == gt: vote_ok += 1

                # Full 9-way vote
                if all_preds:
                    full_majority = Counter(all_preds).most_common(1)[0][0]
                    if full_majority == gt: full_ok += 1
                break

        for mname, ok_count in [('single_best', single_ok), ('per_strat_soup', soup_ok),
                                 ('cross_strat_vote', vote_ok), ('full_9way', full_ok)]:
            acc = ok_count / max(total, 1)
            methods[mname].append(acc)
            print(f"    {mname:<20}: {acc:.1%}")

    results = {}
    print(f"\n  === MULTI-STRATEGY ENSEMBLE ({N_SEEDS} seeds) ===")
    for mname in methods:
        vals = np.array(methods[mname])
        results[f'{mname}_mean'] = float(vals.mean())
        results[f'{mname}_std'] = float(vals.std())
        results[f'{mname}_all'] = [float(v) for v in vals]
        print(f"  {mname:<20}: {vals.mean():.1%} +/- {vals.std():.1%}")

    results['n_seeds'] = N_SEEDS
    results['elapsed'] = time.time() - t0
    results['timestamp'] = time.strftime('%Y-%m-%dT%H:%M:%S')

    with open(os.path.join(RESULTS_DIR, 'phase76_ensemble.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    labels = list(methods.keys())
    means = [results[f'{m}_mean'] for m in labels]
    stds = [results[f'{m}_std'] for m in labels]
    colors = ['#9E9E9E', '#4CAF50', '#FF9800', '#E91E63']
    bars = ax.bar(range(4), means, yerr=stds, color=colors, alpha=0.85, capsize=4)
    ax.set_xticks(range(4))
    ax.set_xticklabels(['Single\nBest', 'Per-Strategy\nSoup', 'Cross-Strategy\nVote', 'Full 9-Way\nVote'], fontsize=9)
    ax.set_ylabel('Full Match'); ax.set_title('Phase 76: Multi-Strategy Ensemble')
    ax.set_ylim(0.7, 1.0); ax.grid(True, alpha=0.3, axis='y')
    for b, m in zip(bars, means):
        ax.text(b.get_x()+b.get_width()/2, m+0.02, f'{m:.1%}', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'phase76_ensemble.png'), dpi=150)
    plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__ == '__main__': main()
