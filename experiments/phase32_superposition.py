"""
Phase 32: Structural Anti-Superposition
==========================================
Prove: Structure > Scale holds for interpretability too.
Compare polysemanticity between:
  A) 77K structured GlassBox Agent (GNN + Pointer)
  B) Unstructured MLP baseline of similar capacity
Measure: How many SAE features each physical neuron responds to.
Lower polysemanticity = more interpretable = more white-box.
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
MAX_OBJECTS=20;NODE_FEAT_DIM=16;N_OPS=8;N_COLORS=10;HID=64

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

# Model A: Structured GlassBox (GNN + Pointer)
class StructuredAgent(nn.Module):
    def __init__(s,hid=HID):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)
        s._hook_activations={}
    def forward(s,nf,nn_c):
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1);h=s.ne(nf)
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        h=h+s.g2(torch.cat([h,msg.expand_as(h)],-1));h=s.n2(h)*mf
        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        s._hook_activations['global']=g.detach()
        pl=((s.pq(g).unsqueeze(1))*s.pk(h)).sum(-1).masked_fill(~mask,-1e9)
        return s.oh(g),s.c1h(g),s.c2h(g),pl

# Model B: Unstructured MLP (same input/output, no graph structure)
class UnstructuredMLP(nn.Module):
    def __init__(s,hid=HID):
        super().__init__()
        flat_dim=MAX_OBJECTS*NODE_FEAT_DIM
        s.encoder=nn.Sequential(
            nn.Linear(flat_dim,hid*4),nn.ReLU(),
            nn.Linear(hid*4,hid*2),nn.ReLU(),
            nn.Linear(hid*2,hid),nn.ReLU())
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.ph=nn.Linear(hid,MAX_OBJECTS)
        s._hook_activations={}
    def forward(s,nf,nn_c):
        flat=nf.view(nf.size(0),-1)
        g=s.encoder(flat)
        s._hook_activations['global']=g.detach()
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        pl=s.ph(g).masked_fill(~mask,-1e9)
        return s.oh(g),s.c1h(g),s.c2h(g),pl

# SAE for measuring polysemanticity
class SparseAutoencoder(nn.Module):
    def __init__(s,in_dim,n_features=128):
        super().__init__()
        s.encoder=nn.Linear(in_dim,n_features)
        s.decoder=nn.Linear(n_features,in_dim)
    def forward(s,x):
        z=F.relu(s.encoder(x))
        x_hat=s.decoder(z)
        return x_hat,z

def compute_loss(model,s):
    ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
    return(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
           F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))

def train_model(model,samples,epochs=60):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(epochs):
        model.train();random.shuffle(samples);el,nb=0,0
        for s in samples:
            loss=compute_loss(model,s);opt.zero_grad();loss.backward();opt.step();el+=loss.item();nb+=1
        if(ep+1)%20==0:print(f"    Epoch {ep+1}/{epochs}: loss={el/nb:.4f}")

def collect_activations(model,samples):
    model.eval();acts=[]
    for s in samples:
        with torch.no_grad():model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
        acts.append(model._hook_activations['global'].cpu())
    return torch.cat(acts,0)

def measure_polysemanticity(activations,n_sae_features=128,sparsity_weight=1e-3):
    """Train SAE, then measure how many features each physical neuron responds to."""
    sae=SparseAutoencoder(activations.size(1),n_sae_features).to(DEVICE)
    opt=torch.optim.Adam(sae.parameters(),lr=1e-3)
    acts_d=activations.to(DEVICE)
    for _ in range(200):
        x_hat,z=sae(acts_d)
        loss=F.mse_loss(x_hat,acts_d)+sparsity_weight*z.abs().mean()
        opt.zero_grad();loss.backward();opt.step()

    # Measure: for each physical neuron, how many SAE features correlate > threshold
    with torch.no_grad():_,z=sae(acts_d)
    z_np=z.cpu().numpy();acts_np=activations.numpy()
    # Correlation matrix: (n_physical x n_sae_features)
    n_physical=acts_np.shape[1]
    poly_scores=[]
    for i in range(n_physical):
        neuron_acts=acts_np[:,i]
        if neuron_acts.std()<1e-8:
            poly_scores.append(0);continue
        n_correlated=0
        for j in range(n_sae_features):
            feat_acts=z_np[:,j]
            if feat_acts.std()<1e-8:continue
            corr=np.corrcoef(neuron_acts,feat_acts)[0,1]
            if abs(corr)>0.3:n_correlated+=1
        poly_scores.append(n_correlated)
    return np.array(poly_scores),z_np

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8')as fp:t.append({'id':f[:-5],**json.load(fp)})
    return t

def main():
    print("="*60);print("Phase 32: Structural Anti-Superposition");print("="*60)
    t0=time.time();tasks=load_arc_tasks(DATA_DIR);print(f"Loaded {len(tasks)} tasks")

    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)

    # Train Model A: Structured
    print("\n--- A: Structured GlassBox Agent ---")
    model_a=StructuredAgent().to(DEVICE)
    train_model(model_a,all_samples,60)

    # Train Model B: Unstructured MLP
    print("\n--- B: Unstructured MLP ---")
    model_b=UnstructuredMLP().to(DEVICE)
    train_model(model_b,all_samples,60)

    # Collect activations
    print("\n--- Collecting Activations ---")
    acts_a=collect_activations(model_a,all_samples[:500])
    acts_b=collect_activations(model_b,all_samples[:500])
    print(f"  Structured activations: {acts_a.shape}")
    print(f"  Unstructured activations: {acts_b.shape}")

    # Measure polysemanticity
    print("\n--- Measuring Polysemanticity via SAE ---")
    poly_a,z_a=measure_polysemanticity(acts_a)
    poly_b,z_b=measure_polysemanticity(acts_b)

    avg_poly_a=poly_a.mean()
    avg_poly_b=poly_b.mean()
    mono_a=(poly_a<=1).sum()/len(poly_a)  # Fraction of monosemantic neurons
    mono_b=(poly_b<=1).sum()/len(poly_b)

    print(f"\n--- Results ---")
    print(f"  Structured GlassBox:")
    print(f"    Avg polysemanticity score: {avg_poly_a:.2f} features/neuron")
    print(f"    Monosemantic neurons: {mono_a:.1%}")
    print(f"  Unstructured MLP:")
    print(f"    Avg polysemanticity score: {avg_poly_b:.2f} features/neuron")
    print(f"    Monosemantic neurons: {mono_b:.1%}")
    print(f"\n  Structure reduces polysemanticity by {(1-avg_poly_a/max(avg_poly_b,0.01)):.1%}")

    # SAE sparsity comparison
    sparsity_a=(z_a>0).mean()
    sparsity_b=(z_b>0).mean()
    print(f"  SAE sparsity - Structured: {sparsity_a:.1%}, Unstructured: {sparsity_b:.1%}")

    elapsed=time.time()-t0
    out={'structured':{'avg_polysemanticity':float(avg_poly_a),'monosemantic_frac':float(mono_a),
                       'sae_sparsity':float(sparsity_a),'params':sum(p.numel()for p in model_a.parameters())},
         'unstructured':{'avg_polysemanticity':float(avg_poly_b),'monosemantic_frac':float(mono_b),
                         'sae_sparsity':float(sparsity_b),'params':sum(p.numel()for p in model_b.parameters())},
         'reduction':float(1-avg_poly_a/max(avg_poly_b,0.01)),
         'elapsed':elapsed,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(RESULTS_DIR,'phase32_superposition.json'),'w',encoding='utf-8')as f:
        json.dump(out,f,indent=2,ensure_ascii=False)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    ax=axes[0]
    ax.hist(poly_a,bins=20,alpha=0.7,label=f'Structured (avg={avg_poly_a:.1f})',color='#4CAF50')
    ax.hist(poly_b,bins=20,alpha=0.7,label=f'Unstructured (avg={avg_poly_b:.1f})',color='#F44336')
    ax.set_xlabel('Features per Neuron (Polysemanticity)');ax.set_ylabel('Count')
    ax.set_title('Polysemanticity Distribution');ax.legend()

    ax=axes[1]
    names=['Structured\nGlassBox','Unstructured\nMLP']
    mono_vals=[mono_a,mono_b];colors=['#4CAF50','#F44336']
    bars=ax.bar(names,mono_vals,color=colors)
    ax.set_ylabel('Monosemantic Neuron Fraction');ax.set_ylim(0,1)
    ax.set_title('Monosemanticity: Structure vs Scale')
    for b,v in zip(bars,mono_vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.1%}',ha='center',fontsize=13)

    plt.tight_layout();plt.savefig(os.path.join(FIGURES_DIR,'phase32_superposition.png'),dpi=150);plt.close()
    print(f"\nElapsed: {elapsed:.1f}s");return out

if __name__=='__main__':main()
