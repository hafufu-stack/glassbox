"""
Phase 78: Replay-Buffered Continual Self-Play
================================================
Fix P74's catastrophic forgetting by mixing original ARC data
into every training batch (Experience Replay).
3 iterations of self-play with 50% replay ratio.
1 seed.
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

def adapt_model(model,task_samples,steps=100,lr=0.1):
    if not task_samples:return model
    am=copy.deepcopy(model);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am

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

def eval_model(model,task):
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

def generate_synthetic_task():
    h,w=random.randint(3,10),random.randint(3,10)
    bg=random.randint(0,9);grid=[[bg]*w for _ in range(h)]
    for _ in range(random.randint(1,3)):
        color=random.choice([c for c in range(10) if c!=bg])
        r0,c0=random.randint(0,h-1),random.randint(0,w-1)
        for r in range(r0,min(r0+random.randint(1,3),h)):
            for c in range(c0,min(c0+random.randint(1,3),w)):grid[r][c]=color
    arr=np.array(grid);op=random.choice(['identity','recolor','flipv','fliph','fill'])
    if op=='identity':out=grid
    elif op=='recolor':
        old_c=random.choice([c for c in range(10) if c!=bg and c in arr])
        new_c=random.choice([c for c in range(10) if c!=old_c])
        o=arr.copy();o[o==old_c]=new_c;out=o.tolist()
    elif op=='flipv':out=np.flipud(arr).tolist()
    elif op=='fliph':out=np.fliplr(arr).tolist()
    elif op=='fill':out=[[random.randint(0,9)]*w for _ in range(h)]
    else:out=grid
    return{'train':[{'input':grid,'output':out}],'test':[{'input':grid,'output':out}]}

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t


def main():
    print("=" * 60)
    print("Phase 78: Replay-Buffered Continual Self-Play")
    print("Fix catastrophic forgetting with experience replay")
    print("=" * 60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8);test_tasks=tasks[split:]

    random.seed(42);np.random.seed(42);torch.manual_seed(42)

    # Initial training
    original_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:original_samples.append(s)

    model=Agent().to(DEVICE)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(80):
        model.train();random.shuffle(original_samples)
        for s in original_samples:
            loss=compute_loss(model,s);opt.zero_grad();loss.backward();opt.step()
    model.eval()

    # Baseline
    base_ok,base_total=0,0
    for task in test_tasks:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_samples=[prep(ai,ao) for ai,ao in aug_pairs]
        aug_samples=[s for s in aug_samples if s]
        adapted=adapt_model(model,aug_samples,steps=100,lr=0.1)
        ok,tot=eval_model(adapted,task);base_ok+=ok;base_total+=tot
    base_acc=base_ok/max(base_total,1)
    print(f"  Baseline: {base_acc:.1%}")

    N_ITERATIONS=3;N_SYNTHETIC=200;REPLAY_RATIO=0.5
    accs=[base_acc]

    for iteration in range(N_ITERATIONS):
        print(f"\n  === Iteration {iteration+1}/{N_ITERATIONS} (with replay) ===")
        good_samples=[]
        n_tried=0
        for _ in range(N_SYNTHETIC*5):
            syn_task=generate_synthetic_task()
            tp=syn_task['train'][0];s=prep(tp['input'],tp['output'])
            if s is None:continue
            n_tried+=1
            with torch.no_grad():
                ol,cl1,_,_=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            if not(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item()):
                aug_pairs=augment_pair(tp['input'],tp['output'])
                aug_s=[prep(ai,ao)for ai,ao in aug_pairs]
                aug_s=[x for x in aug_s if x]
                ablated=ablate_l2(model,0.20,aug_s)
                adapted=adapt_model(ablated,aug_s,steps=100,lr=0.1)
                with torch.no_grad():
                    ol2,cl12,_,_=adapted(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
                if ol2.argmax(1).item()==s['op'].item() and cl12.argmax(1).item()==s['c1'].item():
                    good_samples.append(s)
                    if len(good_samples)>=N_SYNTHETIC:break

        print(f"    Found {len(good_samples)} ZPD samples from {n_tried} tried")
        if not good_samples:
            accs.append(accs[-1]);continue

        # KEY DIFFERENCE from P74: Experience Replay
        # Each epoch: 50% original data + 50% synthetic
        n_orig_per_epoch = len(good_samples)  # Match synthetic count
        opt=torch.optim.Adam(model.parameters(),lr=5e-4)
        for ep in range(30):
            model.train()
            # Sample original data (replay buffer)
            replay = random.sample(original_samples, min(n_orig_per_epoch, len(original_samples)))
            # Mix: synthetic + replay
            mixed = good_samples + replay
            random.shuffle(mixed)
            for s in mixed:
                loss=compute_loss(model,s);opt.zero_grad();loss.backward();opt.step()
        model.eval()

        # Evaluate
        iter_ok,iter_total=0,0
        for task in test_tasks:
            demos=task.get('train',[]);aug_pairs=[]
            for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
            aug_samples=[prep(ai,ao)for ai,ao in aug_pairs]
            aug_samples=[s for s in aug_samples if s]
            adapted=adapt_model(model,aug_samples,steps=100,lr=0.1)
            ok,tot=eval_model(adapted,task);iter_ok+=ok;iter_total+=tot
        iter_acc=iter_ok/max(iter_total,1)
        accs.append(iter_acc)
        print(f"    Iter {iteration+1}: {iter_acc:.1%} ({iter_acc-base_acc:+.1%} vs baseline)")

    results={
        'baseline_acc':base_acc,'iteration_accs':accs,'final_acc':accs[-1],
        'improvement':accs[-1]-base_acc,'replay_ratio':REPLAY_RATIO,
        'elapsed':time.time()-t0,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR,'phase78_replay.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    fig,ax=plt.subplots(1,1,figsize=(8,6))
    ax.plot(range(len(accs)),accs,'o-',linewidth=2,color='#4CAF50',markersize=8,label='With Replay (P78)')
    # P74 comparison line
    p74_accs=[base_acc,base_acc+0.034,base_acc-0.057]  # From P74 results
    ax.plot(range(len(p74_accs)),p74_accs,'o--',linewidth=2,color='#FF9800',markersize=8,alpha=0.7,label='No Replay (P74)')
    ax.set_xlabel('Self-Play Iteration');ax.set_ylabel('Accuracy')
    ax.set_title('Phase 78: Replay-Buffered Self-Play vs P74')
    ax.set_xticks(range(max(len(accs),len(p74_accs))))
    ax.set_xticklabels(['Base']+[f'Iter {i+1}' for i in range(max(len(accs),len(p74_accs))-1)])
    ax.grid(True,alpha=0.3);ax.set_ylim(0.7,1.0);ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'phase78_replay.png'),dpi=150);plt.close()

    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__=='__main__':main()
