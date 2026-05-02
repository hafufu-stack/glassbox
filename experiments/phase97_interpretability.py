"""
Phase 97: Interpretability of Latent Dynamics
===============================================
Decode what the AI imagines during latent MCTS.
Apply linear probes to latent dynamics hidden states.
"""
import os,sys,json,time,copy,random,math
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from collections import deque
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import accuracy_score,r2_score

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

class DynamicsModel(nn.Module):
    def __init__(s,hid=64):
        super().__init__()
        s.action_emb=nn.Embedding(N_ACTIONS,16)
        s.transition=nn.Sequential(nn.Linear(hid+16,hid),nn.ReLU(),nn.Linear(hid,hid),nn.LayerNorm(hid))
        s.reward_head=nn.Sequential(nn.Linear(hid,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh())
    def forward(s,state,action_idx):
        a_emb=s.action_emb(action_idx)
        combined=torch.cat([state,a_emb],dim=-1)
        next_state=s.transition(combined)
        reward=s.reward_head(next_state)
        return next_state,reward

class PredictionHead(nn.Module):
    def __init__(s,hid=64):
        super().__init__()
        s.policy=nn.Linear(hid,N_ACTIONS)
        s.value=nn.Sequential(nn.Linear(hid,32),nn.ReLU(),nn.Linear(32,1),nn.Tanh())
    def forward(s,state):
        return s.policy(state),s.value(state)

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

def collect_dynamics_data(encoder,train_tasks,n_tasks=40):
    data=[]
    for task in train_tasks[:n_tasks]:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        with torch.no_grad():
            s0=aug_s[0];_,_,_,_,state=encoder(s0['nf'].to(DEVICE),s0['nn'].to(DEVICE))
        for action_idx in range(N_ACTIONS):
            a_rate,lr,steps=ACTIONS[action_idx]
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            with torch.no_grad():
                _,_,_,_,next_state=adapted(s0['nf'].to(DEVICE),s0['nn'].to(DEVICE))
            lb=demo_loss_enc(encoder,aug_s);la=demo_loss_enc(adapted,aug_s)
            reward=max(-1.0,min(1.0,(lb-la)))
            ok,tot=eval_task_enc(adapted,task)
            success=1.0 if ok==tot and tot>0 else 0.0
            reward=0.5*reward+0.5*success
            data.append({'state':state.detach(),'action':action_idx,
                         'next_state':next_state.detach(),'reward':reward,
                         'task_id':task['id'],'n_objects':len(extract_objects(task['train'][0]['input'])[0]),
                         'success':success})
    return data

def train_dynamics(dynamics,pred_head,dyn_data,epochs=100):
    opt=torch.optim.Adam(list(dynamics.parameters())+list(pred_head.parameters()),lr=1e-3)
    for ep in range(epochs):
        random.shuffle(dyn_data);total_loss=0;n=0
        for d in dyn_data:
            state=d['state'];action=torch.tensor([d['action']],device=DEVICE)
            target_next=d['next_state'];target_reward=torch.tensor([[d['reward']]],device=DEVICE)
            pred_next,pred_reward=dynamics(state,action)
            loss=F.mse_loss(pred_next,target_next)+F.mse_loss(pred_reward,target_reward)
            _,pred_val=pred_head(state)
            loss=loss+0.5*F.mse_loss(pred_val,target_reward)
            opt.zero_grad();loss.backward();opt.step()
            total_loss+=loss.item();n+=1
    print(f"    Dynamics loss: {total_loss/max(n,1):.4f}")
    return dynamics,pred_head

# ========================================
# Probing latent dynamics states
# ========================================
def collect_probe_data(encoder,dynamics,dyn_data):
    """Collect (latent_vector, property) pairs for probing."""
    probe_data=[]
    for d in dyn_data:
        state=d['state'].cpu().numpy().flatten()
        next_state=d['next_state'].cpu().numpy().flatten()
        action=torch.tensor([d['action']],device=DEVICE)
        with torch.no_grad():
            pred_next,pred_reward=dynamics(d['state'].to(DEVICE),action)
        pred_state=pred_next.cpu().numpy().flatten()
        probe_data.append({
            'state':state,'next_state':next_state,'pred_state':pred_state,
            'action':d['action'],'reward':d['reward'],
            'n_objects':d['n_objects'],'success':d['success'],
        })
    return probe_data

def run_probes(probe_data):
    """Run linear probes on latent states."""
    results={}
    X=np.array([d['state'] for d in probe_data])
    X_next=np.array([d['next_state'] for d in probe_data])
    X_pred=np.array([d['pred_state'] for d in probe_data])
    n=len(X);split=int(n*0.8)

    # Probe 1: Can we decode action from state delta?
    y_action=np.array([d['action'] for d in probe_data])
    X_delta=X_next-X
    clf=LogisticRegression(max_iter=500,random_state=42)
    clf.fit(X_delta[:split],y_action[:split])
    acc=accuracy_score(y_action[split:],clf.predict(X_delta[split:]))
    results['action_from_delta']={'accuracy':float(acc),'chance':1.0/N_ACTIONS}
    print(f"    Action from delta: {acc:.1%} (chance={1.0/N_ACTIONS:.1%})")

    # Probe 2: Can we decode n_objects from state?
    y_nobj=np.array([min(d['n_objects'],10) for d in probe_data])
    clf2=LogisticRegression(max_iter=500,random_state=42)
    clf2.fit(X[:split],y_nobj[:split])
    acc2=accuracy_score(y_nobj[split:],clf2.predict(X[split:]))
    n_classes=len(set(y_nobj))
    results['n_objects_from_state']={'accuracy':float(acc2),'n_classes':n_classes}
    print(f"    N_objects from state: {acc2:.1%} ({n_classes} classes)")

    # Probe 3: Can we predict success from state?
    y_success=np.array([int(d['success']) for d in probe_data])
    clf3=LogisticRegression(max_iter=500,random_state=42)
    clf3.fit(X[:split],y_success[:split])
    acc3=accuracy_score(y_success[split:],clf3.predict(X[split:]))
    base_rate=y_success.mean()
    results['success_from_state']={'accuracy':float(acc3),'base_rate':float(base_rate)}
    print(f"    Success from state: {acc3:.1%} (base={base_rate:.1%})")

    # Probe 4: Can we predict reward from state+action?
    y_reward=np.array([d['reward'] for d in probe_data])
    X_sa=np.column_stack([X,y_action])
    reg=Ridge(alpha=1.0)
    reg.fit(X_sa[:split],y_reward[:split])
    r2=r2_score(y_reward[split:],reg.predict(X_sa[split:]))
    results['reward_from_state_action']={'r2':float(r2)}
    print(f"    Reward R2: {r2:.3f}")

    # Probe 5: Dynamics model fidelity
    cos_sims=[]
    for i in range(len(X_next)):
        a=X_next[i];b=X_pred[i]
        na=np.linalg.norm(a);nb=np.linalg.norm(b)
        if na>0 and nb>0:cos_sims.append(float(np.dot(a,b)/(na*nb)))
    avg_cos=np.mean(cos_sims) if cos_sims else 0
    mse=float(np.mean((X_next-X_pred)**2))
    results['dynamics_fidelity']={'cos_sim':float(avg_cos),'mse':mse}
    print(f"    Dynamics fidelity: cos={avg_cos:.3f}, mse={mse:.4f}")

    # Probe 6: PCA of latent space - cluster by action type
    from sklearn.decomposition import PCA
    pca=PCA(n_components=2,random_state=42)
    X2d=pca.fit_transform(X_delta)
    results['pca_variance']=pca.explained_variance_ratio_.tolist()
    print(f"    PCA variance: {pca.explained_variance_ratio_[0]:.1%}, {pca.explained_variance_ratio_[1]:.1%}")

    return results,X2d,y_action

def main():
    print("="*60)
    print("Phase 97: Interpretability of Latent Dynamics")
    print("Decode what AI imagines during latent MCTS")
    print("="*60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8)
    train_tasks=tasks[:split]
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

    # Collect + train dynamics
    print("\n  Collecting dynamics data...")
    dyn_data=collect_dynamics_data(encoder,train_tasks,n_tasks=50)
    print(f"  {len(dyn_data)} transitions")
    dynamics=DynamicsModel().to(DEVICE)
    pred_head=PredictionHead().to(DEVICE)
    dynamics,pred_head=train_dynamics(dynamics,pred_head,dyn_data,epochs=100)

    # Collect probe data
    print("\n  Running interpretability probes...")
    probe_data=collect_probe_data(encoder,dynamics,dyn_data)
    probe_results,X2d,y_action=run_probes(probe_data)

    results={
        'probes':probe_results,
        'n_samples':len(probe_data),
        'elapsed':time.time()-t0,
        'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR,'phase97_interpretability.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    # Plot
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    # PCA scatter
    ax=axes[0]
    cmap=plt.cm.tab10
    for a in range(N_ACTIONS):
        mask=y_action==a
        if mask.any():
            ax.scatter(X2d[mask,0],X2d[mask,1],c=[cmap(a/N_ACTIONS)],s=15,alpha=0.5,label=f'A{a}')
    ax.set_xlabel('PC1');ax.set_ylabel('PC2')
    ax.set_title('Latent Delta PCA by Action',fontweight='bold')
    ax.legend(fontsize=6,ncol=2)

    # Probe accuracies bar chart
    ax2=axes[1]
    names=['Action\nfrom delta','N_objects\nfrom state','Success\nfrom state']
    accs=[probe_results['action_from_delta']['accuracy'],
          probe_results['n_objects_from_state']['accuracy'],
          probe_results['success_from_state']['accuracy']]
    baselines=[1.0/N_ACTIONS,1.0/probe_results['n_objects_from_state']['n_classes'],
               probe_results['success_from_state']['base_rate']]
    x=np.arange(len(names))
    ax2.bar(x-0.15,accs,0.3,color='#E91E63',label='Probe')
    ax2.bar(x+0.15,baselines,0.3,color='#9E9E9E',label='Chance')
    ax2.set_xticks(x);ax2.set_xticklabels(names);ax2.set_ylabel('Accuracy')
    ax2.legend();ax2.set_ylim(0,1.0);ax2.set_title('Linear Probe Results',fontweight='bold')

    # Dynamics fidelity
    ax3=axes[2]
    metrics=['Cos Similarity','R2 (Reward)']
    vals=[probe_results['dynamics_fidelity']['cos_sim'],
          max(0,probe_results['reward_from_state_action']['r2'])]
    ax3.barh(metrics,vals,color=['#4CAF50','#2196F3'])
    ax3.set_xlim(0,1.0);ax3.set_title('Dynamics Model Quality',fontweight='bold')
    for i,v in enumerate(vals):ax3.text(v+0.02,i,f'{v:.3f}',va='center')

    plt.suptitle('Phase 97: Latent Dynamics Interpretability',fontsize=15,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'phase97_interpretability.png'),dpi=150);plt.close()
    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__=='__main__':main()
