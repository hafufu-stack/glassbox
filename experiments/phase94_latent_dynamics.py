"""
Phase 94: Latent Dynamics MCTS (MuZero Architecture)
=====================================================
Replace Python DSL simulator with a learned latent dynamics model.
MCTS rollouts happen entirely in latent space -> near-instant search.
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

class GNNEncoder(nn.Module):
    """Representation function h: observation -> latent state."""
    def __init__(s, hid=64):
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
        return s.oh(g),s.c1h(g),s.c2h(g),pl,g

N_ACTIONS = 12
ACTIONS = [
    (0.0,0.1,10),(0.0,0.05,10),(0.0,0.15,10),
    (0.10,0.1,10),(0.15,0.1,10),(0.20,0.1,10),
    (0.15,0.05,10),(0.15,0.15,10),(0.20,0.05,10),
    (0.10,0.1,25),(0.15,0.1,25),(0.20,0.1,25),
]

class DynamicsModel(nn.Module):
    """Dynamics function g: (latent_state, action) -> (next_latent, reward)."""
    def __init__(s, hid=64):
        super().__init__()
        s.action_emb = nn.Embedding(N_ACTIONS, 16)
        s.transition = nn.Sequential(
            nn.Linear(hid + 16, hid), nn.ReLU(),
            nn.Linear(hid, hid), nn.LayerNorm(hid)
        )
        s.reward_head = nn.Sequential(
            nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh()
        )
    def forward(s, state, action_idx):
        a_emb = s.action_emb(action_idx)
        combined = torch.cat([state, a_emb], dim=-1)
        next_state = s.transition(combined)
        reward = s.reward_head(next_state)
        return next_state, reward

class PredictionHead(nn.Module):
    """Prediction function f: latent_state -> (policy, value)."""
    def __init__(s, hid=64):
        super().__init__()
        s.policy = nn.Linear(hid, N_ACTIONS)
        s.value = nn.Sequential(nn.Linear(hid, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh())
    def forward(s, state):
        return s.policy(state), s.value(state)

def compute_loss_enc(encoder, s):
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

def demo_loss_enc(encoder, task_samples):
    if not task_samples: return 999.0
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

def reptile_meta_train(encoder, train_tasks, n_outer=200, n_inner=10, meta_lr=0.1, inner_lr=0.1):
    print("  Reptile meta-learning...")
    for outer in range(n_outer):
        task=random.choice(train_tasks)
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao) for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
        adapted=adapt_model_enc(encoder,aug_s,steps=n_inner,lr=inner_lr)
        with torch.no_grad():
            for p_m,p_a in zip(encoder.parameters(),adapted.parameters()):
                p_m.data+=meta_lr*(p_a.data-p_m.data)
        if outer>0 and outer%50==0:meta_lr*=0.7
    print(f"    Reptile done ({n_outer} outer steps)")
    return encoder

# ========================================
# Collect dynamics training data
# ========================================
def collect_dynamics_data(encoder, train_tasks, n_tasks=40):
    """Collect (state, action, next_state, reward) from real rollouts."""
    data = []
    for task in train_tasks[:n_tasks]:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao) for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
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
            loss_before=demo_loss_enc(encoder,aug_s)
            loss_after=demo_loss_enc(adapted,aug_s)
            reward=max(-1.0,min(1.0,(loss_before-loss_after)))
            ok,tot=eval_task_enc(adapted,task)
            success=1.0 if ok==tot and tot>0 else 0.0
            reward=0.5*reward+0.5*success
            data.append({
                'state':state.detach(),'action':action_idx,
                'next_state':next_state.detach(),'reward':reward
            })
    return data

def train_dynamics(dynamics, pred_head, dyn_data, epochs=100):
    """Train dynamics model and prediction head."""
    opt=torch.optim.Adam(list(dynamics.parameters())+list(pred_head.parameters()),lr=1e-3)
    for ep in range(epochs):
        random.shuffle(dyn_data)
        total_loss=0;n=0
        for d in dyn_data:
            state=d['state'];action=torch.tensor([d['action']],device=DEVICE)
            target_next=d['next_state'];target_reward=torch.tensor([[d['reward']]],device=DEVICE)
            pred_next,pred_reward=dynamics(state,action)
            loss_dyn=F.mse_loss(pred_next,target_next)+F.mse_loss(pred_reward,target_reward)
            _,pred_val=pred_head(state)
            loss_pred=F.mse_loss(pred_val,target_reward)
            loss=loss_dyn+0.5*loss_pred
            opt.zero_grad();loss.backward();opt.step()
            total_loss+=loss.item();n+=1
    avg=total_loss/max(n,1)
    print(f"    Dynamics training loss: {avg:.4f} ({len(dyn_data)} samples)")
    return dynamics,pred_head,avg

# ========================================
# Latent MCTS (search in latent space)
# ========================================
class LatentNode:
    def __init__(s,state,parent=None,action=-1):
        s.state=state;s.parent=parent;s.action=action
        s.children=[];s.visits=0;s.value=0.0;s.prior=0.0
    def ucb1(s,c=1.41):
        if s.visits==0:return float('inf')
        pv=s.parent.visits if s.parent else 1
        return s.value/s.visits+c*math.sqrt(math.log(pv)/s.visits)+0.3*s.prior

def latent_mcts(encoder, dynamics, pred_head, task_samples, n_rollouts=32):
    """MCTS entirely in latent space (MuZero-style)."""
    s0=task_samples[0]
    with torch.no_grad():
        _,_,_,_,root_state=encoder(s0['nf'].to(DEVICE),s0['nn'].to(DEVICE))
    root=LatentNode(root_state);root.visits=1
    # Create children with policy priors
    with torch.no_grad():
        policy_logits,_=pred_head(root_state)
        policy=F.softmax(policy_logits,dim=-1).squeeze(0)
    for i in range(N_ACTIONS):
        action=torch.tensor([i],device=DEVICE)
        with torch.no_grad():
            next_s,reward=dynamics(root_state,action)
        child=LatentNode(next_s,parent=root,action=i)
        child.prior=policy[i].item()
        root.children.append(child)
    best_action=0;best_value=-999.0
    for _ in range(n_rollouts):
        node=max(root.children,key=lambda n:n.ucb1())
        with torch.no_grad():
            _,value=pred_head(node.state)
        v=value.item()
        node.visits+=1;node.value+=v;root.visits+=1
        if node.visits>0 and node.value/node.visits>best_value:
            best_value=node.value/node.visits;best_action=node.action
    return best_action,best_value

def real_mcts(encoder, task_samples, n_rollouts=32):
    """Standard MCTS with real simulator for comparison."""
    visits=np.zeros(N_ACTIONS);values=np.zeros(N_ACTIONS);tv=1
    best_loss=float('inf');best_action=0
    for _ in range(n_rollouts):
        ucbs=[]
        for i in range(N_ACTIONS):
            if visits[i]==0:ucbs.append(float('inf'))
            else:ucbs.append(values[i]/visits[i]+1.41*math.sqrt(math.log(tv)/visits[i]))
        idx=max(range(N_ACTIONS),key=lambda i:ucbs[i])
        a_rate,lr,steps=ACTIONS[idx]
        m=copy.deepcopy(encoder)
        if a_rate>0:m=ablate_l2_enc(m,a_rate,task_samples)
        adapted=adapt_model_enc(m,task_samples,steps=steps,lr=lr)
        loss=demo_loss_enc(adapted,task_samples)
        visits[idx]+=1;values[idx]+=-loss;tv+=1
        if loss<best_loss:best_loss=loss;best_action=idx
    return best_action,best_loss

def main():
    print("="*60)
    print("Phase 94: Latent Dynamics MCTS (MuZero)")
    print("Search in latent space, bypass Python simulator")
    print("="*60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8)
    train_tasks=tasks[:split];test_tasks=tasks[split:]
    random.seed(42);np.random.seed(42);torch.manual_seed(42)

    # Train encoder (base GNN)
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
    encoder.eval()
    print("  Encoder trained.")
    encoder=reptile_meta_train(encoder,train_tasks,n_outer=200,n_inner=10)
    encoder.eval()

    # Collect dynamics data
    print("\n  Collecting dynamics training data...")
    dyn_data=collect_dynamics_data(encoder,train_tasks,n_tasks=40)
    print(f"  Collected {len(dyn_data)} transitions")

    # Train dynamics model
    dynamics=DynamicsModel().to(DEVICE)
    pred_head=PredictionHead().to(DEVICE)
    print("  Training dynamics model...")
    dynamics,pred_head,dyn_loss=train_dynamics(dynamics,pred_head,dyn_data,epochs=100)

    # Evaluate: Real MCTS vs Latent MCTS
    print("\n  Evaluating Real vs Latent MCTS...")
    rollout_counts=[8,16,32,64,128]
    results_real={};results_latent={}

    for n_roll in rollout_counts:
        print(f"\n  === Rollouts: {n_roll} ===")
        # Real MCTS (with simulator)
        t1=time.time()
        ok_r,tot_r=0,0
        for task in test_tasks:
            demos=task.get('train',[]);aug_pairs=[]
            for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
            aug_s=[prep(ai,ao) for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
            if not aug_s:continue
            best_action,_=real_mcts(encoder,aug_s,n_rollouts=n_roll)
            a_rate,lr,steps=ACTIONS[best_action]
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            ok,tot=eval_task_enc(adapted,task);ok_r+=ok;tot_r+=tot
        acc_r=ok_r/max(tot_r,1);time_r=time.time()-t1
        results_real[n_roll]={'acc':acc_r,'time':time_r}
        print(f"    Real MCTS-{n_roll}: {acc_r:.1%} ({time_r:.1f}s)")

        # Latent MCTS (neural only)
        t2=time.time()
        ok_l,tot_l=0,0
        for task in test_tasks:
            demos=task.get('train',[]);aug_pairs=[]
            for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
            aug_s=[prep(ai,ao) for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
            if not aug_s:continue
            best_action,_=latent_mcts(encoder,dynamics,pred_head,aug_s,n_rollouts=n_roll)
            a_rate,lr,steps=ACTIONS[best_action]
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            ok,tot=eval_task_enc(adapted,task);ok_l+=ok;tot_l+=tot
        acc_l=ok_l/max(tot_l,1);time_l=time.time()-t2
        results_latent[n_roll]={'acc':acc_l,'time':time_l}
        print(f"    Latent MCTS-{n_roll}: {acc_l:.1%} ({time_l:.1f}s)")

    results={
        'dynamics_loss':dyn_loss,'n_transitions':len(dyn_data),
        'real_mcts':{str(k):v for k,v in results_real.items()},
        'latent_mcts':{str(k):v for k,v in results_latent.items()},
        'elapsed':time.time()-t0,
        'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR,'phase94_latent_dynamics.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    fig,axes=plt.subplots(1,2,figsize=(14,6))
    rolls=sorted(rollout_counts)
    # Left: Accuracy
    ax=axes[0]
    real_accs=[results_real[r]['acc'] for r in rolls]
    lat_accs=[results_latent[r]['acc'] for r in rolls]
    ax.plot(rolls,real_accs,'o--',lw=2,ms=10,color='#9E9E9E',label='Real MCTS (simulator)')
    ax.plot(rolls,lat_accs,'s-',lw=3,ms=10,color='#9C27B0',label='Latent MCTS (neural)')
    ax.set_xlabel('Rollouts');ax.set_ylabel('Accuracy')
    ax.set_xscale('log',base=2);ax.set_xticks(rolls)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend();ax.grid(True,alpha=0.3);ax.set_ylim(0.5,1.0)
    ax.set_title('Accuracy: Real vs Latent MCTS')
    # Right: Speed
    ax2=axes[1]
    real_times=[results_real[r]['time'] for r in rolls]
    lat_times=[results_latent[r]['time'] for r in rolls]
    ax2.plot(rolls,real_times,'o--',lw=2,ms=10,color='#9E9E9E',label='Real MCTS')
    ax2.plot(rolls,lat_times,'s-',lw=3,ms=10,color='#9C27B0',label='Latent MCTS')
    ax2.set_xlabel('Rollouts');ax2.set_ylabel('Time (s)')
    ax2.set_xscale('log',base=2);ax2.set_xticks(rolls)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.legend();ax2.grid(True,alpha=0.3)
    ax2.set_title('Speed: Real vs Latent MCTS')
    plt.suptitle('Phase 94: MuZero Latent Dynamics MCTS',fontsize=15,fontweight='bold')
    plt.tight_layout();plt.savefig(os.path.join(FIGURES_DIR,'phase94_latent_dynamics.png'),dpi=150);plt.close()
    print(f"\nElapsed: {results['elapsed']:.1f}s")
    return results

if __name__=='__main__':main()
