"""
Phase 29: Unsupervised Test-Time Training
============================================
Adapt on the TEST input itself (no labels!) by minimizing entropy.
If the model is "uncertain" about its prediction, update weights
to make it more confident. Also enforce consistency under augmentation.
"""
import os,sys,json,time,copy
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

def prep(inp,out=None):
    objs,bg=extract_objects(inp);h,w=len(inp),len(inp[0])
    if not objs:return None
    nf=np.zeros((MAX_OBJECTS,NODE_FEAT_DIM),dtype=np.float32)
    n=min(len(objs),MAX_OBJECTS)
    for j in range(n):f=obj_feats(objs[j],h,w);nf[j,:len(f)]=f[:NODE_FEAT_DIM]
    r={'nf':torch.tensor(nf).unsqueeze(0).float(),'nn':torch.tensor([n])}
    if out is not None:
        op,ptr,c1,c2=extract_op(inp,out)
        r.update({'op':torch.tensor([op]),'c1':torch.tensor([c1]),'c2':torch.tensor([c2]),
                  'ptr':torch.tensor([min(ptr,max(n-1,0))])})
    return r

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

def entropy_loss(logits):
    p=F.softmax(logits,dim=-1)
    return-(p*torch.log(p+1e-8)).sum(-1).mean()

def load_arc_tasks(d,n=400):
    t=[]
    for f in sorted(os.listdir(d))[:n]:
        if f.endswith('.json'):
            with open(os.path.join(d,f),'r',encoding='utf-8')as fp:t.append({'id':f[:-5],**json.load(fp)})
    return t

def main():
    print("="*60);print("Phase 29: Unsupervised Test-Time Training");print("="*60)
    t0=time.time();tasks=load_arc_tasks(DATA_DIR);print(f"Loaded {len(tasks)} tasks")

    all_samples=[]
    for task in tasks:
        for p in task.get('train',[]):
            s=prep(p['input'],p['output'])
            if s:all_samples.append(s)

    model=Agent().to(DEVICE);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(80):
        model.train();np.random.shuffle(all_samples);el,nb=0,0
        for s in all_samples:
            ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            loss=(F.cross_entropy(ol,s['op'].to(DEVICE))+F.cross_entropy(cl1,s['c1'].to(DEVICE))+
                  F.cross_entropy(cl2,s['c2'].to(DEVICE))+F.cross_entropy(pl,s['ptr'].to(DEVICE)))
            opt.zero_grad();loss.backward();opt.step();el+=loss.item();nb+=1
        if(ep+1)%20==0:print(f"  Epoch {ep+1}/80: loss={el/nb:.4f}")

    print("\n--- Test-Time Training Strategies ---")
    split=int(len(tasks)*0.8);test_tasks=tasks[split:]

    strategies={
        'greedy':{'ok':0,'total':0},
        'demo_adapt_10':{'ok':0,'total':0},
        'demo_adapt_10+entropy_5':{'ok':0,'total':0},
        'demo_adapt_10+entropy_10':{'ok':0,'total':0},
    }

    model.eval()
    for task in test_tasks:
        demos=task.get('train',[]);tests=task.get('test',[])
        if not tests:continue
        demo_samples=[prep(p['input'],p['output'])for p in demos]
        demo_samples=[d for d in demo_samples if d]
        test_samples=[(prep(p['input'],p['output']),p['input'])for p in tests]
        test_samples=[(s,i)for s,i in test_samples if s]
        if not test_samples:continue

        for s,inp in test_samples:
            # Greedy
            with torch.no_grad():
                ol,cl1,cl2,pl=model(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            strategies['greedy']['total']+=1
            if(ol.argmax(1).item()==s['op'].item()and cl1.argmax(1).item()==s['c1'].item()and
               cl2.argmax(1).item()==s['c2'].item()and pl.argmax(1).item()==s['ptr'].item()):
                strategies['greedy']['ok']+=1

            # Demo adapt 10
            am=copy.deepcopy(model);ao=torch.optim.SGD(am.parameters(),lr=1e-2);am.train()
            for _ in range(10):
                if not demo_samples:break
                tl=sum(F.cross_entropy(am(d['nf'].to(DEVICE),d['nn'].to(DEVICE))[0],d['op'].to(DEVICE))+
                       F.cross_entropy(am(d['nf'].to(DEVICE),d['nn'].to(DEVICE))[1],d['c1'].to(DEVICE))+
                       F.cross_entropy(am(d['nf'].to(DEVICE),d['nn'].to(DEVICE))[2],d['c2'].to(DEVICE))+
                       F.cross_entropy(am(d['nf'].to(DEVICE),d['nn'].to(DEVICE))[3],d['ptr'].to(DEVICE))
                       for d in demo_samples)/max(len(demo_samples),1)
                ao.zero_grad();tl.backward();torch.nn.utils.clip_grad_norm_(am.parameters(),1.0);ao.step()

            am.eval()
            with torch.no_grad():
                ol2,cl12,cl22,pl2=am(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
            strategies['demo_adapt_10']['total']+=1
            if(ol2.argmax(1).item()==s['op'].item()and cl12.argmax(1).item()==s['c1'].item()and
               cl22.argmax(1).item()==s['c2'].item()and pl2.argmax(1).item()==s['ptr'].item()):
                strategies['demo_adapt_10']['ok']+=1

            # Demo adapt + entropy minimization on test input
            for ent_steps,key in[(5,'demo_adapt_10+entropy_5'),(10,'demo_adapt_10+entropy_10')]:
                am2=copy.deepcopy(am);ao2=torch.optim.SGD(am2.parameters(),lr=5e-3);am2.train()
                test_s=prep(inp)
                if test_s:
                    for _ in range(ent_steps):
                        ol_e,cl1_e,cl2_e,pl_e=am2(test_s['nf'].to(DEVICE),test_s['nn'].to(DEVICE))
                        ent=entropy_loss(ol_e)+entropy_loss(cl1_e)+entropy_loss(cl2_e)+entropy_loss(pl_e)
                        ao2.zero_grad();ent.backward();torch.nn.utils.clip_grad_norm_(am2.parameters(),0.5);ao2.step()
                am2.eval()
                with torch.no_grad():
                    ol3,cl13,cl23,pl3=am2(s['nf'].to(DEVICE),s['nn'].to(DEVICE))
                strategies[key]['total']+=1
                if(ol3.argmax(1).item()==s['op'].item()and cl13.argmax(1).item()==s['c1'].item()and
                   cl23.argmax(1).item()==s['c2'].item()and pl3.argmax(1).item()==s['ptr'].item()):
                    strategies[key]['ok']+=1

    print("\n--- Results ---")
    for name,data in strategies.items():
        r=data['ok']/max(data['total'],1)
        print(f"  {name}: {r:.1%} ({data['ok']}/{data['total']})")

    elapsed=time.time()-t0
    out={k:{'rate':v['ok']/max(v['total'],1),'ok':v['ok'],'total':v['total']}for k,v in strategies.items()}
    out['elapsed']=elapsed;out['timestamp']=time.strftime('%Y-%m-%dT%H:%M:%S')
    with open(os.path.join(RESULTS_DIR,'phase29_ttt.json'),'w',encoding='utf-8')as f:
        json.dump(out,f,indent=2,ensure_ascii=False)

    fig,ax=plt.subplots(figsize=(10,5))
    names=[k.replace('_','\n')for k in strategies]
    vals=[strategies[k]['ok']/max(strategies[k]['total'],1)for k in strategies]
    colors=['#FF9800','#2196F3','#4CAF50','#9C27B0']
    bars=ax.bar(names,vals,color=colors)
    ax.set_ylim(0,1);ax.set_ylabel('Full Match')
    ax.set_title('Phase 29: Unsupervised Test-Time Training')
    for b,v in zip(bars,vals):ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.1%}',ha='center',fontsize=11)
    plt.tight_layout();plt.savefig(os.path.join(FIGURES_DIR,'phase29_ttt.png'),dpi=150);plt.close()
    print(f"\nElapsed: {elapsed:.1f}s");return out

if __name__=='__main__':main()
