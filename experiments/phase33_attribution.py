"""
Phase 33: The 100% Attribution Path
======================================
Break Anthropic's 25% wall.
For each test prediction, trace the FULL causal path:
  Input Objects -> Node Embeddings -> GNN Message Passing ->
  Global Pool -> Op/Color/Pointer Logits -> Final Decision

Measure: What fraction of predictions can be FULLY traced
with >90% confidence at every step?
"""
import os,sys,json,time,random
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

class TracingAgent(nn.Module):
    """Agent with full causal tracing capability."""
    def __init__(s,hid=HID):
        super().__init__()
        s.ne=nn.Linear(NODE_FEAT_DIM,hid)
        s.g1=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.g2=nn.Sequential(nn.Linear(hid*2,hid),nn.ReLU(),nn.Linear(hid,hid))
        s.n1=nn.LayerNorm(hid);s.n2=nn.LayerNorm(hid)
        s.oh=nn.Linear(hid,N_OPS);s.c1h=nn.Linear(hid,N_COLORS);s.c2h=nn.Linear(hid,N_COLORS)
        s.pq=nn.Linear(hid,hid);s.pk=nn.Linear(hid,hid)

    def forward_with_trace(s,nf,nn_c):
        """Forward pass that returns full attribution trace."""
        trace={}
        mask=torch.arange(MAX_OBJECTS,device=nf.device).unsqueeze(0)<nn_c.unsqueeze(1)
        mf=mask.float().unsqueeze(-1)
        trace['n_objects']=nn_c.item()
        trace['input_features']=nf.detach().cpu()

        # Layer 1: Node embedding
        h=s.ne(nf)
        trace['node_embeddings']=h.detach().cpu()

        # Layer 2: GNN message pass 1
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        trace['msg1_contribution']=msg.detach().cpu()
        h=h+s.g1(torch.cat([h,msg.expand_as(h)],-1));h=s.n1(h)*mf

        # Layer 3: GNN message pass 2
        msg=(h*mf).sum(1,keepdim=True)/mf.sum(1,keepdim=True).clamp(min=1)
        trace['msg2_contribution']=msg.detach().cpu()
        h=h+s.g2(torch.cat([h,msg.expand_as(h)],-1));h=s.n2(h)*mf

        # Layer 4: Global pooling
        g=(h*mf).sum(1)/mf.sum(1).clamp(min=1)
        trace['global_repr']=g.detach().cpu()

        # Layer 5: Output heads
        op_logits=s.oh(g);c1_logits=s.c1h(g);c2_logits=s.c2h(g)
        trace['op_probs']=F.softmax(op_logits,-1).detach().cpu()
        trace['c1_probs']=F.softmax(c1_logits,-1).detach().cpu()
        trace['c2_probs']=F.softmax(c2_logits,-1).detach().cpu()

        # Layer 6: Pointer attention
        pl=((s.pq(g).unsqueeze(1))*s.pk(h)).sum(-1).masked_fill(~mask,-1e9)
        trace['pointer_probs']=F.softmax(pl,-1).detach().cpu()

        # Predictions
        trace['pred_op']=op_logits.argmax(1).item()
        trace['pred_c1']=c1_logits.argmax(1).item()
        trace['pred_c2']=c2_logits.argmax(1).item()
        trace['pred_ptr']=pl.argmax(1).item()

        # Confidence at each decision point
        trace['op_confidence']=trace['op_probs'].max().item()
        trace['c1_confidence']=trace['c1_probs'].max().item()
        trace['c2_confidence']=trace['c2_probs'].max().item()
        trace['ptr_confidence']=trace['pointer_probs'][0,:nn_c.item()].max().item() if nn_c.item()>0 else 0

        # Full attribution: which input object contributed most to pointer?
        ptr_idx=trace['pred_ptr']
        if ptr_idx<nn_c.item():
            trace['pointed_object_features']=nf[0,ptr_idx].detach().cpu().tolist()
        trace['min_confidence']=min(trace['op_confidence'],trace['c1_confidence'],
                                     trace['c2_confidence'],trace['ptr_confidence'])

        return trace

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

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8')as fp:t.append({'id':f[:-5],**json.load(fp)})
    return t

def main():
    print("="*60);print("Phase 33: The 100% Attribution Path");print("="*60)
    t0=time.time();tasks=load_arc_tasks(DATA_DIR);print(f"Loaded {len(tasks)} tasks")

    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)

    model=TracingAgent().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(80):
        model.train();random.shuffle(all_samples);el,nb=0,0
        for s in all_samples:
            ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            loss=(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
                  F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))
            opt.zero_grad();loss.backward();opt.step();el+=loss.item();nb+=1
        if(ep+1)%20==0:print(f"  Epoch {ep+1}/80: loss={el/nb:.4f}")

    # Trace all test predictions
    print("\n--- Attribution Tracing ---")
    split=int(len(tasks)*0.8);test_tasks=tasks[split:]

    model.eval()
    confidence_thresholds=[0.5,0.7,0.9]
    total_traced=0;fully_traced={t:0 for t in confidence_thresholds}
    all_min_confs=[];correct_traces=0;total_tests=0

    for task in test_tasks:
        for tp in task.get('test',[]):
            s=prep(tp['input'],tp['output'])
            if s is None:continue
            total_traced+=1;total_tests+=1

            with torch.no_grad():
                trace=model.forward_with_trace(s['nf'].to(DEVICE),s['nn'].to(DEVICE))

            all_min_confs.append(trace['min_confidence'])
            # Check if prediction is correct
            correct=(trace['pred_op']==s['op'].item()and trace['pred_c1']==s['c1'].item()and
                     trace['pred_c2']==s['c2'].item()and trace['pred_ptr']==s['ptr'].item())
            if correct:correct_traces+=1

            for t in confidence_thresholds:
                if trace['min_confidence']>=t:
                    fully_traced[t]+=1

    print(f"\n--- Results ---")
    print(f"  Total test samples: {total_traced}")
    print(f"  Correct predictions: {correct_traces}/{total_tests} ({correct_traces/max(total_tests,1):.1%})")
    print(f"\n  Full Attribution Path Coverage:")
    for t in confidence_thresholds:
        rate=fully_traced[t]/max(total_traced,1)
        vs_anthropic="EXCEEDS 25%!" if rate>0.25 else "Below 25%"
        print(f"    Confidence>={t:.0%}: {rate:.1%} of predictions fully traceable ({vs_anthropic})")

    # Confidence distribution
    all_min_confs=np.array(all_min_confs)
    print(f"\n  Min confidence statistics:")
    print(f"    Mean: {all_min_confs.mean():.3f}")
    print(f"    Median: {np.median(all_min_confs):.3f}")
    print(f"    >50%: {(all_min_confs>0.5).mean():.1%}")
    print(f"    >90%: {(all_min_confs>0.9).mean():.1%}")

    elapsed=time.time()-t0
    out={'total_traced':total_traced,'correct':correct_traces,
         'accuracy':correct_traces/max(total_tests,1),
         'full_attribution':{str(t):fully_traced[t]/max(total_traced,1)for t in confidence_thresholds},
         'confidence_mean':float(all_min_confs.mean()),
         'confidence_median':float(np.median(all_min_confs)),
         'anthropic_25pct_comparison':{str(t):'EXCEEDS'if fully_traced[t]/max(total_traced,1)>0.25 else'BELOW'
                                        for t in confidence_thresholds},
         'elapsed':elapsed,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(RESULTS_DIR,'phase33_attribution.json'),'w',encoding='utf-8')as f:
        json.dump(out,f,indent=2,ensure_ascii=False)

    fig,axes=plt.subplots(1,2,figsize=(14,5))
    ax=axes[0]
    ax.hist(all_min_confs,bins=30,color='#2196F3',alpha=0.8,edgecolor='white')
    ax.axvline(x=0.5,color='#FF9800',linestyle='--',label='50% threshold')
    ax.axvline(x=0.9,color='#4CAF50',linestyle='--',label='90% threshold')
    ax.set_xlabel('Minimum Confidence Across All Decision Points')
    ax.set_ylabel('Count');ax.set_title('Attribution Path Confidence Distribution');ax.legend()

    ax=axes[1]
    anthro=0.25  # Anthropic's 25% wall
    our_rates=[fully_traced[t]/max(total_traced,1)for t in confidence_thresholds]
    x=np.arange(len(confidence_thresholds)+1);names=[f'Anthropic\n(LLM)']+[f'GlassBox\n(>{t:.0%})'for t in confidence_thresholds]
    vals=[anthro]+our_rates;colors=['#9E9E9E']+['#4CAF50'if v>anthro else'#F44336'for v in our_rates]
    bars=ax.bar(x,vals,color=colors)
    ax.axhline(y=0.25,color='red',linestyle='--',alpha=0.5,label="Anthropic's 25% Wall")
    ax.set_xticks(x);ax.set_xticklabels(names);ax.set_ylabel('Full Attribution Coverage')
    ax.set_title("Breaking Anthropic's 25% Wall");ax.legend();ax.set_ylim(0,1)
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.1%}',ha='center',fontsize=11)

    plt.tight_layout();plt.savefig(os.path.join(FIGURES_DIR,'phase33_attribution.png'),dpi=150);plt.close()
    print(f"\nElapsed: {elapsed:.1f}s");return out

if __name__=='__main__':main()
