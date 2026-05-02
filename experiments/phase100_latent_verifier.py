"""
Phase 100: The Latent Verifier
================================
Use P97's success predictor (99.2%) as a self-verifier.
Generate multiple inference candidates, score each with the
latent verifier, and select the most confident answer.
THE GRAND FINALE - Phase 100!
"""
import os,sys,json,time,copy,random,math
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from collections import deque
from sklearn.linear_model import LogisticRegression
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

class GNNEncoder(nn.Module):
    def __init__(s,hid=64):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid);s.hid=hid
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
        return s.oh(g),s.c1h(g),s.c2h(g),pl,g

N_ACTIONS=12
ACTIONS=[
    (0.0,0.1,10),(0.0,0.05,10),(0.0,0.15,10),
    (0.10,0.1,10),(0.15,0.1,10),(0.20,0.1,10),
    (0.15,0.05,10),(0.15,0.15,10),(0.20,0.05,10),
    (0.10,0.1,25),(0.15,0.1,25),(0.20,0.1,25),
]

def compute_loss_enc(encoder,s):
    ol,cl1,cl2,pl,_=encoder(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
    return(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
           F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))

def adapt_model_enc(encoder,task_samples,steps=10,lr=0.1):
    if not task_samples:return encoder
    am=copy.deepcopy(encoder);opt=torch.optim.SGD(am.parameters(),lr=lr);am.train()
    for _ in range(steps):
        batch=random.sample(task_samples,min(8,len(task_samples)))
        tl=sum(compute_loss_enc(am,d)for d in batch)/len(batch)
        opt.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);opt.step()
    am.eval();return am

def ablate_l2_enc(encoder,rate,task_samples):
    am=copy.deepcopy(encoder);am.train()
    tl=torch.tensor(0.0,device=DEVICE)
    for s in task_samples[:8]:tl=tl+compute_loss_enc(am,s)
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

def eval_task_enc(encoder,task):
    ok,total=0,0
    for tp in task.get('test',[]):
        s=prep(tp['input'],tp['output'])
        if s is None:continue
        total+=1
        with torch.no_grad():
            ol,cl1,cl2,pl,_=encoder(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        if(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
           cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok,total

def demo_loss_enc(encoder,task_samples):
    if not task_samples:return 999.0
    total=0.0
    with torch.no_grad():
        for s in task_samples[:8]:total+=compute_loss_enc(encoder,s).item()
    return total/min(len(task_samples),8)

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t

def reptile_meta_train(encoder,train_tasks,n_outer=200,n_inner=10,meta_lr=0.1,inner_lr=0.1):
    print("  Reptile meta-learning...")
    for outer in range(n_outer):
        task=random.choice(train_tasks);demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        adapted=adapt_model_enc(encoder,aug_s,steps=n_inner,lr=inner_lr)
        with torch.no_grad():
            for p_m,p_a in zip(encoder.parameters(),adapted.parameters()):
                p_m.data+=meta_lr*(p_a.data-p_m.data)
        if outer>0 and outer%50==0:meta_lr*=0.7
    print(f"    Reptile done ({n_outer} outer steps)")
    return encoder

# ========================================
# Train Success Verifier (from P97)
# ========================================
def train_verifier(encoder,train_tasks):
    """Train a success predictor on latent states."""
    X_list=[];y_list=[]
    for task in train_tasks[:60]:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        # Try multiple strategies, record state + success
        strategies=[
            (0.0,0.1,10),(0.15,0.1,10),(0.20,0.1,10),
            (0.15,0.05,10),(0.15,0.1,25),(0.0,0.05,10),
        ]
        for a_rate,lr,steps in strategies:
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            with torch.no_grad():
                _,_,_,_,state=adapted(aug_s[0]['nf'].to(DEVICE),aug_s[0]['nn'].to(DEVICE))
            ok,tot=eval_task_enc(adapted,task)
            success=1 if ok==tot and tot>0 else 0
            X_list.append(state.cpu().numpy().flatten())
            y_list.append(success)
    X=np.array(X_list);y=np.array(y_list)
    clf=LogisticRegression(max_iter=500,random_state=42,C=1.0)
    clf.fit(X,y)
    # Report training accuracy
    train_acc=clf.score(X,y)
    print(f"  Verifier trained: {train_acc:.1%} on {len(y)} samples (pos rate={y.mean():.1%})")
    return clf

# ========================================
# Verified Inference
# ========================================
def verified_inference(encoder,verifier,task,aug_samples):
    """Generate candidates with different strategies, pick best by verifier confidence."""
    strategies=[
        (0.0,0.1,10),(0.0,0.05,10),(0.0,0.15,10),
        (0.10,0.1,10),(0.15,0.1,10),(0.20,0.1,10),
        (0.15,0.05,10),(0.15,0.15,10),(0.20,0.05,10),
        (0.10,0.1,25),(0.15,0.1,25),(0.20,0.1,25),
    ]
    best_score=-1;best_model=None
    for a_rate,lr,steps in strategies:
        m=copy.deepcopy(encoder)
        if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_samples)
        adapted=adapt_model_enc(m,aug_samples,steps=steps,lr=lr)
        with torch.no_grad():
            _,_,_,_,state=adapted(aug_samples[0]['nf'].to(DEVICE),aug_samples[0]['nn'].to(DEVICE))
        # Verifier confidence
        sv=state.cpu().numpy().flatten().reshape(1,-1)
        conf=verifier.predict_proba(sv)[0,1]  # P(success)
        if conf>best_score:
            best_score=conf;best_model=adapted
    return best_model,best_score

def main():
    print("="*60)
    print("*" * 60)
    print("  PHASE 100: THE LATENT VERIFIER")
    print("  The Grand Finale of Project GlassBox")
    print("*" * 60)
    print("="*60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8)
    train_tasks=tasks[:split];test_tasks=tasks[split:]
    random.seed(42);np.random.seed(42);torch.manual_seed(42)

    # Train encoder
    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)
    encoder=GNNEncoder().to(DEVICE)
    opt=torch.optim.Adam(encoder.parameters(),lr=1e-3)
    for ep in range(80):
        encoder.train();random.shuffle(all_samples)
        for s in all_samples:
            loss=compute_loss_enc(encoder,s);opt.zero_grad();loss.backward();opt.step()
    encoder.eval();print("  Encoder trained.")
    encoder=reptile_meta_train(encoder,train_tasks,n_outer=200,n_inner=10)
    encoder.eval()

    # Train verifier
    print("\n  Training Latent Verifier...")
    verifier=train_verifier(encoder,train_tasks)

    # === Evaluate 3 methods ===
    print("\n  Evaluating...")

    # Method 1: No verification (best single strategy)
    print("\n  [Method 1] Single best strategy (no verifier)...")
    ok1,tot1=0,0
    for task in test_tasks:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        m=copy.deepcopy(encoder)
        m=ablate_l2_enc(m,0.15,aug_s)
        adapted=adapt_model_enc(m,aug_s,steps=10,lr=0.1)
        ok,tot=eval_task_enc(adapted,task);ok1+=ok;tot1+=tot
    acc1=ok1/max(tot1,1)
    print(f"    Single strategy: {acc1:.1%}")

    # Method 2: Best of 12 by demo loss (oracle-free)
    print("\n  [Method 2] Best of 12 by demo loss...")
    ok2,tot2=0,0
    for task in test_tasks:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        best_loss=float('inf');best_m=None
        for a_rate,lr,steps in ACTIONS:
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            loss=demo_loss_enc(adapted,aug_s)
            if loss<best_loss:best_loss=loss;best_m=adapted
        ok,tot=eval_task_enc(best_m,task);ok2+=ok;tot2+=tot
    acc2=ok2/max(tot2,1)
    print(f"    Demo loss selector: {acc2:.1%}")

    # Method 3: Latent Verifier selection
    print("\n  [Method 3] Latent Verifier selection (P100)...")
    ok3,tot3=0,0;confidences=[]
    for task in test_tasks:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        best_m,conf=verified_inference(encoder,verifier,task,aug_s)
        ok,tot=eval_task_enc(best_m,task);ok3+=ok;tot3+=tot
        confidences.append(conf)
    acc3=ok3/max(tot3,1)
    print(f"    Latent Verifier: {acc3:.1%} (avg confidence={np.mean(confidences):.3f})")

    # Method 4: Verifier + demo loss combined
    print("\n  [Method 4] Verifier + Demo Loss combined...")
    ok4,tot4=0,0
    for task in test_tasks:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        best_score=-999;best_m=None
        for a_rate,lr,steps in ACTIONS:
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            loss=demo_loss_enc(adapted,aug_s)
            with torch.no_grad():
                _,_,_,_,state=adapted(aug_s[0]['nf'].to(DEVICE),aug_s[0]['nn'].to(DEVICE))
            conf=verifier.predict_proba(state.cpu().numpy().flatten().reshape(1,-1))[0,1]
            score=0.5*conf+0.5*max(0,1.0-loss)  # Combined score
            if score>best_score:best_score=score;best_m=adapted
        ok,tot=eval_task_enc(best_m,task);ok4+=ok;tot4+=tot
    acc4=ok4/max(tot4,1)
    print(f"    Combined: {acc4:.1%}")

    results={
        'single_strategy':acc1,'demo_loss_selector':acc2,
        'latent_verifier':acc3,'combined':acc4,
        'avg_confidence':float(np.mean(confidences)),
        'elapsed':time.time()-t0,
        'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR,'phase100_latent_verifier.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    fig,ax=plt.subplots(1,1,figsize=(10,6))
    methods=['Single\nStrategy','Demo Loss\nSelector','Latent\nVerifier','Combined\n(V+Loss)']
    accs=[acc1,acc2,acc3,acc4]
    colors=['#9E9E9E','#2196F3','#E91E63','#FFD700']
    bars=ax.bar(methods,accs,color=colors,edgecolor='black',linewidth=0.5)
    for bar,acc in zip(bars,accs):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                f'{acc:.1%}',ha='center',va='bottom',fontweight='bold',fontsize=14)
    ax.set_ylabel('Accuracy',fontsize=13)
    ax.set_ylim(0.6,1.0);ax.grid(True,alpha=0.3,axis='y')
    ax.axhline(y=acc1,color='gray',ls=':',alpha=0.3)
    plt.title('PHASE 100: The Latent Verifier - Grand Finale',fontsize=16,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'phase100_latent_verifier.png'),dpi=150);plt.close()

    print("\n" + "="*60)
    print("  PHASE 100 COMPLETE!")
    print(f"  Best accuracy: {max(accs):.1%}")
    print(f"  Total elapsed: {results['elapsed']:.1f}s")
    print("="*60)
    return results

if __name__=='__main__':main()
