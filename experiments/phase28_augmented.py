"""
Phase 28: Data-Augmented Adaptation
======================================
Demo count (2-3) causes overfitting during adaptation.
Solution: Augment demos with D8 group transforms + color permutations.
2 demos -> 20+ augmented demos -> adaptation without overfitting.
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
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10;MAX_GRID=30

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
    """Generate augmented versions via D8 transforms + color permutation."""
    ia,oa=np.array(inp),np.array(out)
    augmented=[(inp,out)]  # original
    # D8: rotations and flips (only same-shape preserving)
    if ia.shape==oa.shape:
        for k in[1,2,3]:
            augmented.append((np.rot90(ia,k).tolist(),np.rot90(oa,k).tolist()))
        augmented.append((np.flipud(ia).tolist(),np.flipud(oa).tolist()))
        augmented.append((np.fliplr(ia).tolist(),np.fliplr(oa).tolist()))
    # Color permutation (2 random)
    for _ in range(2):
        perm=list(range(10));random.shuffle(perm)
        ai=np.vectorize(lambda x:perm[x] if x<10 else x)(ia)
        ao=np.vectorize(lambda x:perm[x] if x<10 else x)(oa)
        augmented.append((ai.tolist(),ao.tolist()))
    return augmented

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

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8')as fp:t.append({'id':f[:-5],**json.load(fp)})
    return t

def main():
    print("="*60);print("Phase 28: Data-Augmented Adaptation");print("="*60)
    t0=time.time();tasks=load_arc_tasks(DATA_DIR);print(f"Loaded {len(tasks)} tasks")

    # Train base model
    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)
    print(f"Total samples: {len(all_samples)}")
    model=Agent().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(80):
        model.train();random.shuffle(all_samples);el,nb=0,0
        for s in all_samples:
            loss=compute_loss(model,s);opt.zero_grad();loss.backward();opt.step();el+=loss.item();nb+=1
        if(ep+1)%20==0:print(f"  Epoch {ep+1}/80: loss={el/nb:.4f}")

    # Evaluate: no-aug vs aug adaptation
    print("\n--- Augmented vs Non-Augmented Adaptation ---")
    split=int(len(tasks)*0.8);test_tasks=tasks[split:]
    steps_list=[0,5,10,20,50]
    results_noaug={};results_aug={}

    for ns in steps_list:
        ok_no,ok_aug,total=0,0,0
        for task in test_tasks:
            demos=task.get('train',[]);tests=task.get('test',[])
            if not demos or not tests:continue
            demo_samples=[prep(p['input'],p['output'])for p in demos]
            demo_samples=[d for d in demo_samples if d]
            # Augmented demos
            aug_pairs=[]
            for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
            aug_samples=[prep(ai,ao)for ai,ao in aug_pairs]
            aug_samples=[s for s in aug_samples if s]

            test_samples=[prep(p['input'],p['output'])for p in tests]
            test_samples=[s for s in test_samples if s]
            if not test_samples:continue

            for label,ds,res_dict,ok_var in[('noaug',demo_samples,results_noaug,None),('aug',aug_samples,results_aug,None)]:
                if ns==0:am=model
                else:
                    am=copy.deepcopy(model);ao=torch.optim.SGD(am.parameters(),lr=1e-2);am.train()
                    for _ in range(ns):
                        if not ds:break
                        batch=random.sample(ds,min(8,len(ds)))
                        tl=sum(compute_loss(am,d)for d in batch)/len(batch)
                        ao.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);ao.step()
                am.eval()
                for s in test_samples:
                    with torch.no_grad():
                        ol,cl1,cl2,pl=am(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
                    if(ol.argmax(1).item()==s['op'].item()and cl1.argmax(1).item()==s['c1'].item()and
                       cl2.argmax(1).item()==s['c2'].item()and pl.argmax(1).item()==s['ptr'].item()):
                        if label=='noaug':ok_no+=1
                        else:ok_aug+=1
            total+=len(test_samples)

        rn=ok_no/max(total,1);ra=ok_aug/max(total,1)
        results_noaug[ns]=rn;results_aug[ns]=ra
        print(f"  Steps={ns:>2d}: NoAug={rn:.1%}, Aug={ra:.1%} (delta={ra-rn:+.1%})")

    elapsed=time.time()-t0
    out={f'noaug_{k}':v for k,v in results_noaug.items()}
    out.update({f'aug_{k}':v for k,v in results_aug.items()})
    out['elapsed']=elapsed;out['timestamp']=time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(os.path.join(RESULTS_DIR,'phase28_augmented.json'),'w',encoding='utf-8')as f:
        json.dump(out,f,indent=2,ensure_ascii=False)

    fig,ax=plt.subplots(figsize=(10,6))
    ax.plot(steps_list,[results_noaug[s]for s in steps_list],'o-',label='No Augmentation',linewidth=2,color='#FF9800')
    ax.plot(steps_list,[results_aug[s]for s in steps_list],'s-',label='D8 + Color Aug',linewidth=2,color='#4CAF50')
    ax.set_xlabel('Adaptation Steps');ax.set_ylabel('Full Match');ax.set_title('Phase 28: Augmented Adaptation')
    ax.legend(fontsize=12);ax.grid(True,alpha=0.3);ax.set_ylim(0,1)
    plt.tight_layout();plt.savefig(os.path.join(FIGURES_DIR,'phase28_augmented.png'),dpi=150);plt.close()
    print(f"\nElapsed: {elapsed:.1f}s");return out

if __name__=='__main__':main()
