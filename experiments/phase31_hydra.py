"""
Phase 31: Controlled Hydra Self-Repair
=========================================
The Hydra Effect: destroy parts of the brain, then measure
if Adaptation can repair it. First-ever quantification of
AI self-healing via "Hydra Recovery Rate".

Ablation rates: 10%, 20%, 50% of neurons zeroed out.
Then run Augmented Adaptation (50 steps) on the damaged model.
Recovery Rate = (post_adapt - post_ablation) / (original - post_ablation)
"""
import os,sys,json,time,copy,random
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from collections import deque
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

def ablate_model(model,rate):
    """Zero out `rate` fraction of parameters randomly."""
    am=copy.deepcopy(model)
    with torch.no_grad():
        for p in am.parameters():
            mask=torch.rand_like(p)>rate
            p.mul_(mask.float())
    return am

def eval_model(model,test_tasks):
    model.eval();ok,total=0,0
    for task in test_tasks:
        for tp in task.get('test',[]):
            s=prep(tp['input'],tp['output'])
            if s is None:continue
            total+=1
            with torch.no_grad():
                ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            if(ol.argmax(1).item()==s['op'].item()and cl1.argmax(1).item()==s['c1'].item()and
               cl2.argmax(1).item()==s['c2'].item()and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok/max(total,1),ok,total

def adapt_model(model,task,steps=50):
    """Augmented adaptation on a single task's demos."""
    demos=task.get('train',[])
    aug_pairs=[]
    for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
    aug_samples=[prep(ai,ao)for ai,ao in aug_pairs]
    aug_samples=[s for s in aug_samples if s]
    if not aug_samples:return model
    am=copy.deepcopy(model);ao=torch.optim.SGD(am.parameters(),lr=1e-2);am.train()
    for _ in range(steps):
        batch=random.sample(aug_samples,min(8,len(aug_samples)))
        tl=torch.tensor(0.0,device=DEVICE)
        for d in batch:
            ol,cl1,cl2,pl=am(d['nf'].to(DEVICE),d['nn'].to(DEVICE))
            tl=tl+(F.cross_entropy(ol,d['op'].to(DEVICE))+F.cross_entropy(cl1,d['c1'].to(DEVICE))+
                    F.cross_entropy(cl2,d['c2'].to(DEVICE))+F.cross_entropy(pl,d['ptr'].to(DEVICE)))
        tl=tl/len(batch);ao.zero_grad();tl.backward()
        torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);ao.step()
    am.eval();return am

def eval_with_adapt(model,test_tasks,adapt_steps=50):
    ok,total=0,0
    for task in test_tasks:
        tests=task.get('test',[])
        if not tests:continue
        am=adapt_model(model,task,adapt_steps)
        for tp in tests:
            s=prep(tp['input'],tp['output'])
            if s is None:continue
            total+=1
            with torch.no_grad():
                ol,cl1,cl2,pl=am(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            if(ol.argmax(1).item()==s['op'].item()and cl1.argmax(1).item()==s['c1'].item()and
               cl2.argmax(1).item()==s['c2'].item()and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok/max(total,1),ok,total

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8')as fp:t.append({'id':f[:-5],**json.load(fp)})
    return t

def main():
    print("="*60);print("Phase 31: Controlled Hydra Self-Repair");print("="*60)
    t0=time.time();tasks=load_arc_tasks(DATA_DIR);print(f"Loaded {len(tasks)} tasks")

    # Train base model
    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)
    model=Agent().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(80):
        model.train();random.shuffle(all_samples);el,nb=0,0
        for s in all_samples:
            ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            loss=(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
                  F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))
            opt.zero_grad();loss.backward();opt.step();el+=loss.item();nb+=1
        if(ep+1)%20==0:print(f"  Epoch {ep+1}/80: loss={el/nb:.4f}")

    split=int(len(tasks)*0.8);test_tasks=tasks[split:]

    # Baseline performance
    base_r,_,_=eval_model(model,test_tasks)
    base_adapt_r,_,_=eval_with_adapt(model,test_tasks,50)
    print(f"\n  Baseline (no ablation): greedy={base_r:.1%}, adapted={base_adapt_r:.1%}")

    # Ablation experiments
    ablation_rates=[0.1,0.2,0.3,0.5]
    results={'baseline_greedy':base_r,'baseline_adapted':base_adapt_r}

    print("\n--- Hydra Self-Repair Experiment ---")
    print(f"{'Rate':>6} | {'Post-Ablation':>14} | {'Post-Adapt':>11} | {'Recovery Rate':>14} | {'Verdict'}")
    print("-"*75)

    for rate in ablation_rates:
        # Average over 3 random ablation seeds
        drops,recoveries=[],[]
        for seed in range(3):
            torch.manual_seed(seed)
            ablated=ablate_model(model,rate)
            abl_r,_,_=eval_model(ablated,test_tasks)
            rec_r,_,_=eval_with_adapt(ablated,test_tasks,50)
            drops.append(abl_r)
            recoveries.append(rec_r)

        avg_drop=np.mean(drops)
        avg_rec=np.mean(recoveries)
        # Hydra Recovery Rate
        if base_r-avg_drop>0.001:
            hydra_rate=(avg_rec-avg_drop)/(base_r-avg_drop)
        else:
            hydra_rate=1.0

        verdict="FULL REPAIR" if hydra_rate>0.9 else("PARTIAL" if hydra_rate>0.5 else "FAILED")
        print(f"  {rate:>4.0%} | {avg_drop:>13.1%} | {avg_rec:>10.1%} | {hydra_rate:>13.1%} | {verdict}")
        results[f'ablation_{int(rate*100)}']={
            'post_ablation':avg_drop,'post_adapt':avg_rec,
            'hydra_recovery_rate':hydra_rate,'verdict':verdict}

    elapsed=time.time()-t0
    results['elapsed']=elapsed;results['timestamp']=time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(os.path.join(RESULTS_DIR,'phase31_hydra.json'),'w',encoding='utf-8')as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    # Plot
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    rates_pct=[int(r*100)for r in ablation_rates]
    post_abl=[results[f'ablation_{r}']['post_ablation']for r in rates_pct]
    post_adp=[results[f'ablation_{r}']['post_adapt']for r in rates_pct]
    hydra_rates=[results[f'ablation_{r}']['hydra_recovery_rate']for r in rates_pct]

    ax=axes[0]
    x=np.arange(len(rates_pct));w=0.3
    ax.bar(x-w/2,post_abl,w,label='Post-Ablation',color='#F44336',alpha=0.8)
    ax.bar(x+w/2,post_adp,w,label='Post-Adaptation',color='#4CAF50',alpha=0.8)
    ax.axhline(y=base_r,color='#2196F3',linestyle='--',label=f'Baseline ({base_r:.1%})')
    ax.set_xticks(x);ax.set_xticklabels([f'{r}%'for r in rates_pct])
    ax.set_xlabel('Ablation Rate');ax.set_ylabel('Full Match')
    ax.set_title('Hydra Effect: Destruction vs Recovery');ax.legend();ax.set_ylim(0,1)

    ax=axes[1]
    ax.bar(range(len(rates_pct)),hydra_rates,color=['#4CAF50'if h>0.9 else('#FF9800'if h>0.5 else'#F44336')for h in hydra_rates])
    ax.axhline(y=1.0,color='gray',linestyle='--',alpha=0.5)
    ax.set_xticks(range(len(rates_pct)));ax.set_xticklabels([f'{r}%'for r in rates_pct])
    ax.set_xlabel('Ablation Rate');ax.set_ylabel('Hydra Recovery Rate')
    ax.set_title('Self-Repair Capacity');ax.set_ylim(0,1.5)

    plt.tight_layout();plt.savefig(os.path.join(FIGURES_DIR,'phase31_hydra.png'),dpi=150);plt.close()
    print(f"\nElapsed: {elapsed:.1f}s");return results

if __name__=='__main__':main()
