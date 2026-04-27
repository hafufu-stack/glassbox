"""
Phase 25: Pixel-Level Few-Shot Adaptation
============================================
The 85% ceiling exists because the DSL can only represent ~5% of tasks.
Solution: bypass the DSL entirely. Predict OUTPUT PIXELS directly.

Train a CNN that maps input grid -> output grid (pixel classification).
Then apply MAML-style adaptation on demo examples at test time.
If adaptation works for pixel prediction, we break through the DSL ceiling.
"""
import os, sys, json, time, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'training')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIGURES_DIR, exist_ok=True)

MAX_GRID=30; N_COLORS=11

def pad_grid(grid, pad_val=10):
    h,w=len(grid),len(grid[0])
    p=np.full((MAX_GRID,MAX_GRID),pad_val,dtype=np.int64)
    p[:h,:w]=np.array(grid); return p

class PixelModel(nn.Module):
    def __init__(self, ch=48):
        super().__init__()
        self.embed = nn.Embedding(N_COLORS, ch)
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(ch,ch,3,padding=1), nn.BatchNorm2d(ch), nn.ReLU())
            for _ in range(4)])
        self.out = nn.Conv2d(ch, N_COLORS, 1)
    def forward(self, x):
        h = self.embed(x).permute(0,3,1,2)
        for layer in self.layers: h = h + layer(h)
        return self.out(h)

def load_arc_tasks(data_dir, max_tasks=400):
    tasks=[]
    for f in sorted(os.listdir(data_dir))[:max_tasks]:
        if f.endswith('.json'):
            with open(os.path.join(data_dir,f),'r',encoding='utf-8') as fp:
                tasks.append({'id':f.replace('.json',''),**json.load(fp)})
    return tasks

def pixel_accuracy(logits, target):
    return (logits.argmax(1)==target).float().mean().item()

def main():
    print("="*60); print("Phase 25: Pixel-Level Few-Shot Adaptation"); print("="*60)
    t0=time.time(); tasks=load_arc_tasks(DATA_DIR); print(f"Loaded {len(tasks)} tasks")

    # Prepare data
    all_inp, all_out = [], []
    for task in tasks:
        for pair in task.get('train',[]):
            all_inp.append(pad_grid(pair['input']))
            all_out.append(pad_grid(pair['output']))
    inp_t=torch.tensor(np.array(all_inp),dtype=torch.long)
    out_t=torch.tensor(np.array(all_out),dtype=torch.long)
    N=len(inp_t); BATCH=32

    # Train base model
    print("\n--- Training Base Pixel Model ---")
    model=PixelModel().to(DEVICE)
    params=sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(40):
        model.train(); perm=torch.randperm(N); el,nb=0,0
        for i in range(0,N,BATCH):
            idx=perm[i:i+BATCH]
            logits=model(inp_t[idx].to(DEVICE))
            loss=F.cross_entropy(logits,out_t[idx].to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step(); el+=loss.item(); nb+=1
        if (ep+1)%10==0: print(f"  Epoch {ep+1}/40: loss={el/nb:.4f}")

    # Per-task adaptation experiment
    print("\n--- Per-Task Adaptation ---")
    split=int(len(tasks)*0.8); test_tasks=tasks[split:]
    steps_list=[0,5,10,20]
    results={}

    for ns in steps_list:
        total_pa, total_samples = 0, 0
        for task in test_tasks:
            demos=task.get('train',[]); tests=task.get('test',[])
            if not demos or not tests: continue

            if ns==0: am=model
            else:
                am=copy.deepcopy(model)
                ao=torch.optim.SGD(am.parameters(),lr=5e-3)
                am.train()
                demo_inp=torch.stack([torch.tensor(pad_grid(p['input']),dtype=torch.long) for p in demos]).to(DEVICE)
                demo_out=torch.stack([torch.tensor(pad_grid(p['output']),dtype=torch.long) for p in demos]).to(DEVICE)
                for _ in range(ns):
                    loss=F.cross_entropy(am(demo_inp),demo_out)
                    ao.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(am.parameters(),1.0); ao.step()

            am.eval()
            for tp in tests:
                ti=torch.tensor(pad_grid(tp['input']),dtype=torch.long).unsqueeze(0).to(DEVICE)
                to_=torch.tensor(pad_grid(tp['output']),dtype=torch.long).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    pa=pixel_accuracy(am(ti),to_)
                total_pa+=pa; total_samples+=1

        avg_pa=total_pa/max(total_samples,1)
        results[ns]=avg_pa
        print(f"  Steps={ns:>2d}: PA={avg_pa:.1%} ({total_samples} samples)")

    elapsed=time.time()-t0
    out={'results':{str(k):v for k,v in results.items()},
         'params':params,'elapsed':elapsed,'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S')}
    with open(os.path.join(RESULTS_DIR,'phase25_pixel_adapt.json'),'w',encoding='utf-8') as f:
        json.dump(out,f,indent=2,ensure_ascii=False)

    fig,ax=plt.subplots(figsize=(9,5))
    sl=sorted(results.keys()); vals=[results[s] for s in sl]
    bars=ax.bar([f'{s} steps' for s in sl],vals,color=['#FF9800','#2196F3','#4CAF50','#9C27B0'])
    ax.set_ylim(0,1); ax.set_ylabel('Pixel Accuracy')
    ax.set_title('Phase 25: Pixel-Level Few-Shot Adaptation')
    for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2,b.get_height()+0.02,f'{v:.1%}',ha='center',fontsize=13)
    plt.tight_layout(); plt.savefig(os.path.join(FIGURES_DIR,'phase25_pixel_adapt.png'),dpi=150); plt.close()
    print(f"\nElapsed: {elapsed:.1f}s"); return out

if __name__=='__main__': main()
