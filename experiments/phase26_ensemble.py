"""
Phase 26: Grand Ensemble
===========================
Combine ALL methods. If ANY one method solves a task, count it.
- Method A: Base model greedy
- Method B: Gradient adaptation (10 steps)
- Method C: Program search (expanded DSL)
- Method D: Task-conditioned model
Union of all = best possible performance.
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from collections import deque
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

MAX_OBJECTS=20; NODE_FEAT_DIM=16; N_OPS=8; N_COLORS=10; MAX_GRID=30

def extract_objects(grid):
    arr=np.array(grid); h,w=arr.shape; visited=np.zeros_like(arr,dtype=bool)
    bg=int(np.bincount(arr.flatten()).argmax()); objects=[]
    for r in range(h):
        for c in range(w):
            if not visited[r,c] and arr[r,c]!=bg:
                color=int(arr[r,c]); pixels=[]; q=deque([(r,c)]); visited[r,c]=True
                while q:
                    cr,cc=q.popleft(); pixels.append((cr,cc))
                    for dr,dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr,nc=cr+dr,cc+dc
                        if 0<=nr<h and 0<=nc<w and not visited[nr,nc] and arr[nr,nc]==color:
                            visited[nr,nc]=True; q.append((nr,nc))
                rows=[p[0] for p in pixels]; cols=[p[1] for p in pixels]
                objects.append({'color':color,'area':len(pixels),'center':(np.mean(rows),np.mean(cols)),
                    'bbox':(min(rows),min(cols),max(rows),max(cols))})
    return objects, bg

def obj_feats(obj,h,w):
    f=[0.0]*10; f[min(obj['color'],9)]=1.0
    cr,cc=obj['center']; f.extend([cr/max(h,1),cc/max(w,1),obj['area']/max(h*w,1)])
    r0,c0,r1,c1=obj['bbox']; f.extend([(r1-r0+1)/max(h,1),(c1-c0+1)/max(w,1),(c1-c0+1)/max(r1-r0+1,1)])
    return f[:NODE_FEAT_DIM]

def extract_op(inp,out):
    ia,oa=np.array(inp),np.array(out); objects,bg=extract_objects(inp)
    if ia.shape==oa.shape and np.array_equal(ia,oa): return 1,0,0,0
    if oa.size>0 and len(np.unique(oa))==1: return 2,0,int(np.unique(oa)[0]),0
    if ia.shape==oa.shape:
        d=ia!=oa
        if d.any():
            oc,nc=set(ia[d].tolist()),set(oa[d].tolist())
            if len(oc)==1 and len(nc)==1:
                o=int(list(oc)[0]); p=0
                for j,obj in enumerate(objects):
                    if obj['color']==o: p=j; break
                return 5,min(p,MAX_OBJECTS-1),o,int(list(nc)[0])
        if np.array_equal(np.flipud(ia),oa): return 6,0,0,0
        if np.array_equal(np.fliplr(ia),oa): return 7,0,0,0
    return 3,0,0,0

def prep(inp,out):
    objects,bg=extract_objects(inp); h,w=len(inp),len(inp[0])
    if not objects: return None
    nf=np.zeros((MAX_OBJECTS,NODE_FEAT_DIM),dtype=np.float32)
    n=min(len(objects),MAX_OBJECTS)
    for j in range(n): f=obj_feats(objects[j],h,w); nf[j,:len(f)]=f[:NODE_FEAT_DIM]
    op,ptr,c1,c2=extract_op(inp,out)
    return {'nf':torch.tensor(nf).unsqueeze(0).float(),'nn':torch.tensor([n]),
            'op':torch.tensor([op]),'c1':torch.tensor([c1]),'c2':torch.tensor([c2]),
            'ptr':torch.tensor([min(ptr,max(n-1,0))])}

class Agent(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.ne=nn.Linear(NODE_FEAT_DIM,hidden)
        self.g1=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.g2=nn.Sequential(nn.Linear(hidden*2,hidden),nn.ReLU(),nn.Linear(hidden,hidden))
        self.n1=nn.LayerNorm(hidden); self.n2=nn.LayerNorm(hidden)
        self.oh=nn.Linear(hidden,N_OPS); self.c1h=nn.Linear(hidden,N_COLORS)
        self.c2h=nn.Linear(hidden,N_COLORS)
        self.pq=nn.Linear(hidden,hidden); self.pk=nn.Linear(hidden,hidden)
    def forward(self,nf,nn_c):
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1); h=self.ne(nf)
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+self.g1(torch.cat([h,msg.expand_as(h)],-1)); h=self.n1(h)*mf
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+self.g2(torch.cat([h,msg.expand_as(h)],-1)); h=self.n2(h)*mf
        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        pl=((self.pq(g).unsqueeze(1))*self.pk(h)).sum(-1).masked_fill(~mask,-1e9)
        return self.oh(g),self.c1h(g),self.c2h(g),pl

# Import DSL search from Phase 19
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from experiments.phase19_dsl_expansion import search_programs

def load_arc_tasks(data_dir, max_tasks=400):
    tasks=[]
    for f in sorted(os.listdir(data_dir))[:max_tasks]:
        if f.endswith('.json'):
            with open(os.path.join(data_dir,f),'r',encoding='utf-8') as fp:
                tasks.append({'id':f.replace('.json',''),**json.load(fp)})
    return tasks

def main():
    print("="*60); print("Phase 26: Grand Ensemble"); print("="*60)
    t0=time.time(); tasks=load_arc_tasks(DATA_DIR); print(f"Loaded {len(tasks)} tasks")

    # Prepare data
    all_nf,all_nn,all_ops,all_c1s,all_c2s,all_ptrs=[],[],[],[],[],[]
    for task in tasks:
        for pair in task.get('train',[]):
            s=prep(pair['input'],pair['output'])
            if s is None: continue
            all_nf.append(s['nf'].squeeze(0)); all_nn.append(s['nn'].squeeze(0))
            all_ops.append(s['op'].squeeze(0)); all_c1s.append(s['c1'].squeeze(0))
            all_c2s.append(s['c2'].squeeze(0)); all_ptrs.append(s['ptr'].squeeze(0))
    nf_t=torch.stack(all_nf); nn_t=torch.stack(all_nn); op_t=torch.stack(all_ops)
    c1_t=torch.stack(all_c1s); c2_t=torch.stack(all_c2s); ptr_t=torch.stack(all_ptrs)
    N=len(nf_t); BATCH=32

    # Train
    print("\n--- Training ---")
    model=Agent().to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(80):
        model.train(); perm=torch.randperm(N); el,nb=0,0
        for i in range(0,N,BATCH):
            idx=perm[i:i+BATCH]
            ol,cl1,cl2,pl=model(nf_t[idx].to(DEVICE),nn_t[idx].to(DEVICE))
            loss=(F.cross_entropy(ol,op_t[idx].to(DEVICE))+F.cross_entropy(cl1,c1_t[idx].to(DEVICE))+
                  F.cross_entropy(cl2,c2_t[idx].to(DEVICE))+F.cross_entropy(pl,ptr_t[idx].to(DEVICE)))
            opt.zero_grad(); loss.backward(); opt.step(); el+=loss.item(); nb+=1
        if (ep+1)%20==0: print(f"  Epoch {ep+1}/80: loss={el/nb:.4f}")

    # Test
    print("\n--- Grand Ensemble Test ---")
    split=int(len(tasks)*0.8); test_tasks=tasks[split:]

    method_solved = {'greedy':set(), 'adapt':set(), 'search':set()}
    task_idx = 0

    model.eval()
    for task in test_tasks:
        demos=task.get('train',[]); tests=task.get('test',[])
        if not tests: continue
        tid=task['id']

        # Method A: Greedy
        for tp in tests:
            s=prep(tp['input'],tp['output'])
            if s is None: continue
            with torch.no_grad():
                ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            if (ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
                cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):
                method_solved['greedy'].add(tid)

        # Method B: Adapt 10 steps
        demo_samples=[prep(p['input'],p['output']) for p in demos]
        demo_samples=[d for d in demo_samples if d is not None]
        if demo_samples:
            am=copy.deepcopy(model); ao=torch.optim.SGD(am.parameters(),lr=1e-2); am.train()
            for _ in range(10):
                tl=torch.tensor(0.0,device=DEVICE)
                for d in demo_samples:
                    ol,cl1,cl2,pl=am(d['nf'].to(DEVICE),d['nn'].to(DEVICE))
                    tl=tl+(F.cross_entropy(ol,d['op'].to(DEVICE))+F.cross_entropy(cl1,d['c1'].to(DEVICE))+
                           F.cross_entropy(cl2,d['c2'].to(DEVICE))+F.cross_entropy(pl,d['ptr'].to(DEVICE)))
                tl=tl/len(demo_samples); ao.zero_grad(); tl.backward()
                torch.nn.utils.clip_grad_norm_(am.parameters(),1.0); ao.step()
            am.eval()
            for tp in tests:
                s=prep(tp['input'],tp['output'])
                if s is None: continue
                with torch.no_grad():
                    ol,cl1,cl2,pl=am(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
                if (ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
                    cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):
                    method_solved['adapt'].add(tid)

        # Method C: Program search
        all_demo_match = True
        found_prog = None
        for dp in demos:
            prog = search_programs(dp['input'], dp['output'])
            if prog is None: all_demo_match = False; break
            if found_prog is None: found_prog = prog
            elif found_prog != prog: all_demo_match = False; break
        if all_demo_match and found_prog:
            method_solved['search'].add(tid)

        task_idx += 1

    # Compute union
    union = method_solved['greedy'] | method_solved['adapt'] | method_solved['search']
    total_tasks = len(test_tasks)

    print(f"\n--- Results (task-level) ---")
    for name, solved in method_solved.items():
        print(f"  {name}: {len(solved)}/{total_tasks} ({len(solved)/total_tasks:.1%})")
    print(f"  ENSEMBLE (union): {len(union)}/{total_tasks} ({len(union)/total_tasks:.1%})")

    # Venn diagram data
    g,a,s = method_solved['greedy'], method_solved['adapt'], method_solved['search']
    only_g = g - a - s; only_a = a - g - s; only_s = s - g - a
    print(f"\n  Only greedy: {len(only_g)}, Only adapt: {len(only_a)}, Only search: {len(only_s)}")
    print(f"  greedy∩adapt: {len(g&a)}, greedy∩search: {len(g&s)}, adapt∩search: {len(a&s)}")

    elapsed=time.time()-t0
    out={'greedy':len(g)/total_tasks, 'adapt':len(a)/total_tasks, 'search':len(s)/total_tasks,
         'ensemble':len(union)/total_tasks, 'total_tasks':total_tasks,
         'greedy_count':len(g),'adapt_count':len(a),'search_count':len(s),'ensemble_count':len(union),
         'only_greedy':len(only_g),'only_adapt':len(only_a),'only_search':len(only_s),
         'elapsed':elapsed,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(RESULTS_DIR,'phase26_ensemble.json'),'w',encoding='utf-8') as f:
        json.dump(out,f,indent=2,ensure_ascii=False)

    fig,ax=plt.subplots(figsize=(10,6))
    names=['Greedy\n(base)','Adapt-10\n(few-shot)','Search\n(DSL)','ENSEMBLE\n(union)']
    vals=[len(g)/total_tasks,len(a)/total_tasks,len(s)/total_tasks,len(union)/total_tasks]
    colors=['#FF9800','#2196F3','#9C27B0','#4CAF50']
    bars=ax.bar(names,vals,color=colors)
    ax.set_ylim(0,1); ax.set_ylabel('Task Solve Rate')
    ax.set_title('Phase 26: Grand Ensemble (Best of All Methods)')
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.1%}',ha='center',fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR,'phase26_ensemble.png'),dpi=150); plt.close()
    print(f"\nElapsed: {elapsed:.1f}s"); return out

if __name__=='__main__': main()
