"""
Phase 99: Orthogonal LoRA Distillation
========================================
Fix P96's catastrophic forgetting by adding LoRA modules
to GNN layers instead of directly updating backbone weights.
Freeze original weights, only train LoRA deltas.
"""
import os,sys,json,time,copy,random,math
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

# ========================================
# LoRA Module
# ========================================
class LoRALinear(nn.Module):
    """Low-Rank Adaptation wrapper for nn.Linear."""
    def __init__(s,original_linear,rank=4):
        super().__init__()
        s.original=original_linear
        in_f=original_linear.in_features
        out_f=original_linear.out_features
        s.lora_A=nn.Parameter(torch.randn(in_f,rank)*0.01)
        s.lora_B=nn.Parameter(torch.zeros(rank,out_f))
        # Freeze original
        for p in s.original.parameters():p.requires_grad=False
    def forward(s,x):
        return s.original(x)+x@s.lora_A@s.lora_B

class GNNWithLoRA(nn.Module):
    """GNN with LoRA adapters on key linear layers."""
    def __init__(s,hid=64,rank=4):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid);s.hid=hid
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)
        s.policy_head=nn.Sequential(nn.Linear(hid,hid),nn.ReLU(),nn.Linear(hid,12))
        s.value_head=nn.Sequential(nn.Linear(hid,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh())
        # LoRA adapters (added after base training)
        s.lora_ne=None;s.lora_g1_0=None;s.lora_g2_0=None
        s.lora_enabled=False

    def add_lora(s,rank=4):
        s.lora_ne=LoRALinear(s.ne,rank)
        s.lora_g1_0=LoRALinear(s.g1[0],rank)
        s.lora_g2_0=LoRALinear(s.g2[0],rank)
        s.lora_enabled=True

    def forward(s,nf,nn_c):
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1)
        h=s.lora_ne(nf) if s.lora_enabled else s.ne(nf)
        if s.lora_enabled:
            msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
            cat1=torch.cat([h,msg.expand_as(h)],-1)
            h=h+F.relu(s.g1[2](s.lora_g1_0(cat1)));h=s.n1(h)*mf
        else:
            msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
            h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf
        if s.lora_enabled:
            msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
            cat2=torch.cat([h,msg.expand_as(h)],-1)
            h=h+F.relu(s.g2[2](s.lora_g2_0(cat2)));h=s.n2(h)*mf
        else:
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

def compute_loss(model,s):
    ol,cl1,cl2,pl,_=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
    return(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
           F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))

def adapt_model(model,task_samples,steps=10,lr=0.1):
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

def eval_task(model,task):
    ok,total=0,0
    for tp in task.get('test',[]):
        s=prep(tp['input'],tp['output'])
        if s is None:continue
        total+=1
        with torch.no_grad():
            ol,cl1,cl2,pl,_=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        if(ol.argmax(1).item()==s['op'].item() and cl1.argmax(1).item()==s['c1'].item() and
           cl2.argmax(1).item()==s['c2'].item() and pl.argmax(1).item()==s['ptr'].item()):ok+=1
    return ok,total

def demo_loss(model,task_samples):
    if not task_samples:return 999.0
    total=0.0
    with torch.no_grad():
        for s in task_samples[:8]:total+=compute_loss(model,s).item()
    return total/min(len(task_samples),8)

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8') as fp:
                t.append({'id':f[:-5],**json.load(fp)})
    return t

def reptile_meta_train(model,train_tasks,n_outer=200,n_inner=10,meta_lr=0.1,inner_lr=0.1):
    print("  Reptile meta-learning...")
    for outer in range(n_outer):
        task=random.choice(train_tasks);demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        adapted=adapt_model(model,aug_s,steps=n_inner,lr=inner_lr)
        with torch.no_grad():
            for p_m,p_a in zip(model.parameters(),adapted.parameters()):
                p_m.data+=meta_lr*(p_a.data-p_m.data)
        if outer>0 and outer%50==0:meta_lr*=0.7
    print(f"    Reptile done ({n_outer} outer steps)")
    return model

def mcts_collect_visits(model,task_samples,n_rollouts=8):
    visits=np.zeros(N_ACTIONS,dtype=np.float32)
    values=np.zeros(N_ACTIONS,dtype=np.float32);tv=1;best_loss=float('inf')
    for _ in range(n_rollouts):
        ucbs=[]
        for i in range(N_ACTIONS):
            if visits[i]==0:ucbs.append(float('inf'))
            else:ucbs.append(values[i]/visits[i]+1.41*math.sqrt(math.log(tv)/visits[i]))
        idx=max(range(N_ACTIONS),key=lambda i:ucbs[i])
        a_rate,lr,steps=ACTIONS[idx]
        m=copy.deepcopy(model)
        if a_rate>0:m=ablate_l2(m,a_rate,task_samples)
        adapted=adapt_model(m,task_samples,steps=steps,lr=lr)
        loss=demo_loss(adapted,task_samples)
        visits[idx]+=1;values[idx]+=-loss;tv+=1
        if loss<best_loss:best_loss=loss
    visit_dist=visits/visits.sum() if visits.sum()>0 else np.ones(N_ACTIONS)/N_ACTIONS
    value_target=max(-best_loss,0.0)
    return visit_dist,value_target

def distill_lora(model,soft_targets,epochs=5,lr=1e-3):
    """LoRA distillation: only train LoRA parameters + policy/value heads."""
    # Freeze everything except LoRA and policy/value heads
    for p in model.parameters():p.requires_grad=False
    for p in model.policy_head.parameters():p.requires_grad=True
    for p in model.value_head.parameters():p.requires_grad=True
    if model.lora_ne:
        model.lora_ne.lora_A.requires_grad=True;model.lora_ne.lora_B.requires_grad=True
    if model.lora_g1_0:
        model.lora_g1_0.lora_A.requires_grad=True;model.lora_g1_0.lora_B.requires_grad=True
    if model.lora_g2_0:
        model.lora_g2_0.lora_A.requires_grad=True;model.lora_g2_0.lora_B.requires_grad=True

    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.Adam(trainable,lr=lr)
    model.train()
    for ep in range(epochs):
        random.shuffle(soft_targets);total_loss=0;n=0
        for nf,nn_c,target_dist,value_target in soft_targets:
            _,_,_,_,g=model(nf.to(DEVICE),nn_c.to(DEVICE))
            logits=model.policy_head(g)
            log_probs=F.log_softmax(logits,dim=-1)
            target=torch.tensor(target_dist,device=DEVICE).unsqueeze(0)
            policy_loss=F.kl_div(log_probs,target,reduction='batchmean')
            pred_v=model.value_head(g)
            val_t=torch.tensor([[value_target]],device=DEVICE)
            value_loss=F.mse_loss(pred_v,val_t)
            loss=policy_loss+0.5*value_loss
            opt.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step()
            total_loss+=loss.item();n+=1
    for p in model.parameters():p.requires_grad=True
    model.eval()
    return model,total_loss/max(n,1)

def distill_head_only(model,soft_targets,epochs=5,lr=1e-3):
    for p in model.parameters():p.requires_grad=False
    for p in model.policy_head.parameters():p.requires_grad=True
    opt=torch.optim.Adam(model.policy_head.parameters(),lr=lr)
    model.train()
    for ep in range(epochs):
        random.shuffle(soft_targets);total_loss=0;n=0
        for nf,nn_c,target_dist,_ in soft_targets:
            _,_,_,_,g=model(nf.to(DEVICE),nn_c.to(DEVICE))
            logits=model.policy_head(g)
            log_probs=F.log_softmax(logits,dim=-1)
            target=torch.tensor(target_dist,device=DEVICE).unsqueeze(0)
            loss=F.kl_div(log_probs,target,reduction='batchmean')
            opt.zero_grad();loss.backward();opt.step()
            total_loss+=loss.item();n+=1
    for p in model.parameters():p.requires_grad=True
    model.eval()
    return model,total_loss/max(n,1)

def main():
    print("="*60)
    print("Phase 99: Orthogonal LoRA Distillation")
    print("Safe full-brain distillation without catastrophic forgetting")
    print("="*60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8)
    train_tasks=tasks[:split];test_tasks=tasks[split:]
    random.seed(42);np.random.seed(42);torch.manual_seed(42)

    # Train base model
    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)
    model_head=GNNWithLoRA().to(DEVICE)
    opt=torch.optim.Adam(model_head.parameters(),lr=1e-3)
    for ep in range(80):
        model_head.train();random.shuffle(all_samples)
        for s in all_samples:
            loss=compute_loss(model_head,s);opt.zero_grad();loss.backward();opt.step()
    model_head.eval();print("  Base model trained.")
    model_head=reptile_meta_train(model_head,train_tasks,n_outer=200,n_inner=10)
    model_head.eval()

    # Create LoRA model (same starting point)
    model_lora=copy.deepcopy(model_head)
    model_lora.add_lora(rank=4)
    model_lora=model_lora.to(DEVICE)

    def eval_all(model,label):
        zs_ok,zs_tot=0,0
        for task in test_tasks:
            ok,tot=eval_task(model,task);zs_ok+=ok;zs_tot+=tot
        zs=zs_ok/max(zs_tot,1)
        ttt_ok,ttt_tot=0,0
        for task in test_tasks:
            demos=task.get('train',[]);aug_pairs=[]
            for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
            aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
            ablated=ablate_l2(model,0.15,aug_s)
            adapted=adapt_model(ablated,aug_s,steps=100,lr=0.1)
            ok,tot=eval_task(adapted,task);ttt_ok+=ok;ttt_tot+=tot
        ttt=ttt_ok/max(ttt_tot,1)
        print(f"    {label}: ZS={zs:.1%}, TTT={ttt:.1%}")
        return zs,ttt

    zs_base,ttt_base=eval_all(model_head,"Baseline")
    N_ITERS=8
    head_zs_hist=[zs_base];head_ttt_hist=[ttt_base]
    lora_zs_hist=[zs_base];lora_ttt_hist=[ttt_base]

    for iteration in range(N_ITERS):
        print(f"\n  --- Iteration {iteration+1}/{N_ITERS} ---")
        soft_targets=[]
        for task in train_tasks[:60]:
            demos=task.get('train',[]);aug_pairs=[]
            for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
            aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
            if not aug_s:continue
            visit_dist,value_target=mcts_collect_visits(model_lora,aug_s,n_rollouts=8)
            s0=aug_s[0]
            soft_targets.append((s0['nf'],s0['nn'],visit_dist,value_target))
        print(f"    {len(soft_targets)} targets")

        model_head,loss_h=distill_head_only(model_head,soft_targets,epochs=5,lr=1e-3)
        zs_h,ttt_h=eval_all(model_head,f"Head-only iter {iteration+1}")
        head_zs_hist.append(zs_h);head_ttt_hist.append(ttt_h)

        model_lora,loss_l=distill_lora(model_lora,soft_targets,epochs=5,lr=1e-3)
        zs_l,ttt_l=eval_all(model_lora,f"LoRA iter {iteration+1}")
        lora_zs_hist.append(zs_l);lora_ttt_hist.append(ttt_l)

    results={
        'zs_baseline':zs_base,'ttt_baseline':ttt_base,
        'head_only':{'zs':head_zs_hist,'ttt':head_ttt_hist},
        'lora':{'zs':lora_zs_hist,'ttt':lora_ttt_hist},
        'best_head_zs':max(head_zs_hist),'best_lora_zs':max(lora_zs_hist),
        'best_head_ttt':max(head_ttt_hist),'best_lora_ttt':max(lora_ttt_hist),
        'elapsed':time.time()-t0,
        'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR,'phase99_lora_distill.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    fig,axes=plt.subplots(1,2,figsize=(14,6))
    iters=range(len(head_zs_hist))
    ax=axes[0]
    ax.plot(iters,head_zs_hist,'o--',lw=2,ms=8,color='#9E9E9E',label='Head-only (P92)')
    ax.plot(iters,lora_zs_hist,'s-',lw=3,ms=10,color='#FF5722',label='LoRA (P99)')
    ax.axhline(y=zs_base,color='gray',ls=':',alpha=0.4)
    ax.set_xlabel('Iteration');ax.set_ylabel('Zero-Shot Accuracy')
    ax.legend();ax.grid(True,alpha=0.3);ax.set_ylim(0.5,0.9)
    ax.set_title('ZS: Head-only vs LoRA Distillation',fontweight='bold')
    ax2=axes[1]
    ax2.plot(iters,head_ttt_hist,'o--',lw=2,ms=8,color='#9E9E9E',label='Head-only')
    ax2.plot(iters,lora_ttt_hist,'s-',lw=3,ms=10,color='#4CAF50',label='LoRA')
    ax2.axhline(y=ttt_base,color='gray',ls=':',alpha=0.4)
    ax2.set_xlabel('Iteration');ax2.set_ylabel('TTT Accuracy')
    ax2.legend();ax2.grid(True,alpha=0.3);ax2.set_ylim(0.7,1.0)
    ax2.set_title('TTT: Head-only vs LoRA Distillation',fontweight='bold')
    plt.suptitle('Phase 99: LoRA Distillation',fontsize=15,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'phase99_lora_distill.png'),dpi=150);plt.close()
    print(f"\nBest Head ZS={results['best_head_zs']:.1%}, LoRA ZS={results['best_lora_zs']:.1%}")
    print(f"Elapsed: {results['elapsed']:.1f}s")
    return results

if __name__=='__main__':main()
