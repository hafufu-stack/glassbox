"""
Phase 80: Early Stopping for TTT
===================================
Instead of always running 100 adaptation steps,
monitor loss and stop when it plateaus.
Compare: fixed 100 vs early stopping (patience=10).
3 seeds.
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

def compute_loss(model,s):
    ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
    return(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
           F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))

def adapt_fixed(model, task_samples, steps=100, lr=0.1):
    if not task_samples: return model, steps
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am, steps

def adapt_early_stop(model, task_samples, max_steps=200, lr=0.1, patience=10):
    """Adapt with early stopping based on loss plateau."""
    if not task_samples: return model, 0
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    best_loss=float('inf');wait=0;actual_steps=0
    for step in range(max_steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
        actual_steps+=1
        curr_loss=tl.item()
        if curr_loss<best_loss-1e-4:
            best_loss=curr_loss;wait=0
        else:
            wait+=1
        if wait>=patience:break
    am.eval();return am, actual_steps

def ablate_l2(model,rate,task_samples):
    am=copy.deepcopy(model);am.train()
    tl=torch.tensor(0.0,device=DEVICE)
    for s in task_samples[:8]:tl=tl+compute_loss(am,s)
    tl=tl/max(len(task_samples[:8]),1);tl.backward()
    l2p=set()
    for p in am.g2.parameters():l2p.add(id(p))
    for p in am.n2.parameters():l2p.add(id(p))
    with torch.no_grad():
        for p in am.parameters():
            if id(p) not in l2p:continue
            if p.grad is not None:
                imp=p.grad.abs();thr=torch.quantile(imp.flatten(),rate)
                p.mul_((imp>thr).float())
    am.eval();return am

def eval_task(model,task):
    ok,total=0,0
    for tp in task.get('test',[]):
        s=prep(tp['input'],tp['output'])
        if s is None:continue
        total+=1
        with torch.no_grad():
            ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        if(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
           cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok,total

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 80: Early Stopping for TTT")
    print("Fixed 100 steps vs adaptive early stopping")
    print("=" * 60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8);test_tasks=tasks[split:]
    N_SEEDS=3

    fixed_accs,es_accs=[],[]
    fixed_times,es_times=[],[]
    avg_steps_list=[]

    for seed in range(N_SEEDS):
        print(f"\n  --- Seed {seed+1}/{N_SEEDS} ---")
        random.seed(seed*1000);np.random.seed(seed*1000);torch.manual_seed(seed*1000)

        all_samples=[]
        for task in tasks:
            for p in task.get('train',[]):
                s=prep(p['input'],p['output'])
                if s:all_samples.append(s)
        model=Agent().to(DEVICE)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3)
        for ep in range(80):
            model.train();random.shuffle(all_samples)
            for s in all_samples:
                loss=compute_loss(model,s);opt.zero_grad();loss.backward();opt.step()
        model.eval()

        # Fixed 100 steps
        f_ok,f_total,f_time=0,0,0
        es_ok,es_total,es_time,total_steps=0,0,0,0

        for task in test_tasks:
            demos=task.get('train',[]);aug_pairs=[]
            for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
            aug_samples=[prep(ai,ao)for ai,ao in aug_pairs]
            aug_samples=[s for s in aug_samples if s]

            ablated=ablate_l2(model,0.20,aug_samples)

            t1=time.time()
            adapted_f,_=adapt_fixed(ablated,aug_samples,steps=100,lr=0.1)
            f_time+=time.time()-t1
            ok,tot=eval_task(adapted_f,task);f_ok+=ok;f_total+=tot

            t2=time.time()
            adapted_es,n_steps=adapt_early_stop(ablated,aug_samples,max_steps=200,lr=0.1,patience=10)
            es_time+=time.time()-t2
            ok,tot=eval_task(adapted_es,task);es_ok+=ok;es_total+=tot
            total_steps+=n_steps

        f_acc=f_ok/max(f_total,1);es_acc=es_ok/max(es_total,1)
        avg_steps=total_steps/max(len(test_tasks),1)
        fixed_accs.append(f_acc);es_accs.append(es_acc)
        fixed_times.append(f_time);es_times.append(es_time)
        avg_steps_list.append(avg_steps)
        print(f"    Fixed 100: {f_acc:.1%} ({f_time:.1f}s)")
        print(f"    Early Stop: {es_acc:.1%} ({es_time:.1f}s, avg {avg_steps:.0f} steps)")

    results={
        'fixed_mean':float(np.mean(fixed_accs)),'fixed_std':float(np.std(fixed_accs)),
        'es_mean':float(np.mean(es_accs)),'es_std':float(np.std(es_accs)),
        'fixed_time_mean':float(np.mean(fixed_times)),
        'es_time_mean':float(np.mean(es_times)),
        'avg_steps':float(np.mean(avg_steps_list)),
        'speedup':float(np.mean(fixed_times)/max(np.mean(es_times),0.001)),
        'elapsed':time.time()-t0,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
    }

    print(f"\n  === EARLY STOPPING ({N_SEEDS} seeds) ===")
    print(f"  Fixed 100:   {results['fixed_mean']:.1%} +/- {results['fixed_std']:.1%}")
    print(f"  Early Stop:  {results['es_mean']:.1%} +/- {results['es_std']:.1%}")
    print(f"  Avg steps:   {results['avg_steps']:.0f}")
    print(f"  Speedup:     {results['speedup']:.2f}x")

    with open(os.path.join(RESULTS_DIR,'phase80_earlystop.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    fig,axes=plt.subplots(1,2,figsize=(14,6))
    ax=axes[0]
    bars=ax.bar(['Fixed\n(100 steps)','Early Stop\n(patience=10)'],
                [results['fixed_mean'],results['es_mean']],
                yerr=[results['fixed_std'],results['es_std']],
                color=['#9E9E9E','#4CAF50'],alpha=0.85,capsize=4)
    ax.set_ylabel('Accuracy');ax.set_title('Accuracy');ax.set_ylim(0.7,1.0);ax.grid(True,alpha=0.3,axis='y')
    ax=axes[1]
    ax.bar(['Fixed','Early Stop'],[results['fixed_time_mean'],results['es_time_mean']],
           color=['#9E9E9E','#4CAF50'],alpha=0.85)
    ax.set_ylabel('Time (s)');ax.set_title(f"Time (avg {results['avg_steps']:.0f} steps)")
    ax.grid(True,alpha=0.3,axis='y')
    plt.suptitle('Phase 80: Early Stopping for TTT',fontsize=14,fontweight='bold')
    plt.tight_layout();plt.savefig(os.path.join(FIGURES_DIR,'phase80_earlystop.png'),dpi=150);plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__=='__main__':main()
