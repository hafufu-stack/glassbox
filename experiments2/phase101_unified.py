"""
Phase 101: Unified Continuous Dynamics + Latent Verifier
=========================================================
Integrate P98's Continuous Dynamics Model with P100's Latent Verifier.
- P98: Gated residual continuous action embeddings for MCTS
- P100: LogisticRegression verifier to select best inference candidate
- P101: Verifier-guided MCTS with continuous dynamics
"""
import os,sys,json,time,copy,random,math
import numpy as np
import torch,torch.nn as nn,torch.nn.functional as F
from collections import deque
from sklearn.linear_model import LogisticRegression
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt

# Add experiments dir to path for reuse
EXP_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'experiments')
sys.path.insert(0,EXP_DIR)

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'data','training')
RESULTS_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'results')
FIGURES_DIR=os.path.join(os.path.dirname(os.path.dirname(__file__)),'figures')
os.makedirs(RESULTS_DIR,exist_ok=True);os.makedirs(FIGURES_DIR,exist_ok=True)

# Import shared components from P98
from phase98_continuous_actions import (
    GNNEncoder, DynamicsBaseline, DynamicsContinuous, PredictionHead,
    extract_objects, obj_feats, extract_op, prep, augment_pair,
    compute_loss_enc, adapt_model_enc, ablate_l2_enc, eval_task_enc,
    demo_loss_enc, load_arc_tasks, reptile_meta_train,
    collect_dynamics_data, train_dynamics,
    MAX_OBJECTS, NODE_FEAT_DIM, N_OPS, N_COLORS, N_ACTIONS, ACTIONS
)

# ========================================
# Latent Verifier (from P100)
# ========================================
def train_verifier(encoder, train_tasks):
    """Train a success predictor on latent states."""
    X_list=[];y_list=[]
    for task in train_tasks[:60]:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue
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
    train_acc=clf.score(X,y)
    print(f"  Verifier trained: {train_acc:.1%} on {len(y)} samples (pos rate={y.mean():.1%})")
    return clf

# ========================================
# P101: Verifier-Guided Latent MCTS
# ========================================
class VerifierNode:
    def __init__(s,state,parent=None,action=-1):
        s.state=state;s.parent=parent;s.action=action
        s.children=[];s.visits=0;s.value=0.0;s.prior=0.0
        s.verifier_score=0.0
    def ucb1(s,c=1.41,v_weight=0.3):
        if s.visits==0:return float('inf')
        pv=s.parent.visits if s.parent else 1
        exploit=s.value/s.visits
        explore=c*math.sqrt(math.log(pv)/s.visits)
        # Verifier bonus: encourage nodes the verifier likes
        return exploit+explore+v_weight*s.verifier_score+0.2*s.prior

def verifier_guided_mcts(encoder, dynamics, pred_head, verifier,
                         task_samples, n_rollouts=64):
    """MCTS with continuous dynamics + verifier-guided node selection."""
    s0=task_samples[0]
    with torch.no_grad():
        _,_,_,_,root_state=encoder(s0['nf'].to(DEVICE),s0['nn'].to(DEVICE))
    root=VerifierNode(root_state);root.visits=1
    with torch.no_grad():
        policy_logits,_=pred_head(root_state)
        policy=F.softmax(policy_logits,dim=-1).squeeze(0)
    # Expand root children
    for i in range(N_ACTIONS):
        action=torch.tensor([i],device=DEVICE)
        with torch.no_grad():
            next_s,_=dynamics(root_state,action)
        child=VerifierNode(next_s,parent=root,action=i)
        child.prior=policy[i].item()
        # Score with verifier
        sv=next_s.cpu().numpy().flatten().reshape(1,-1)
        child.verifier_score=float(verifier.predict_proba(sv)[0,1])
        root.children.append(child)
    # MCTS rollouts
    best_action=0;best_value=-999.0
    for _ in range(n_rollouts):
        node=max(root.children,key=lambda n:n.ucb1())
        with torch.no_grad():
            _,value=pred_head(node.state)
        v=value.item()
        # Blend value prediction with verifier confidence
        blended=0.6*v+0.4*node.verifier_score
        node.visits+=1;node.value+=blended;root.visits+=1
        avg=node.value/node.visits
        if avg>best_value:
            best_value=avg;best_action=node.action
    return best_action,best_value

# ========================================
# Evaluate methods
# ========================================
def eval_method(encoder, test_tasks, dynamics, pred_head, verifier,
                method_name, use_verifier_mcts=False,
                use_verifier_select=False, use_combined=False,
                n_rollouts=256):
    """Unified evaluation for all methods."""
    ok_total,tot_total=0,0
    confidences=[]
    for task in test_tasks:
        demos=task.get('train',[]);aug_pairs=[]
        for p in demos:aug_pairs.extend(augment_pair(p['input'],p['output']))
        aug_s=[prep(ai,ao)for ai,ao in aug_pairs];aug_s=[s for s in aug_s if s]
        if not aug_s:continue

        if use_verifier_mcts:
            # P101: Verifier-guided MCTS
            ba,_=verifier_guided_mcts(encoder,dynamics,pred_head,verifier,
                                       aug_s,n_rollouts=n_rollouts)
            a_rate,lr,steps=ACTIONS[ba]
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            ok,tot=eval_task_enc(adapted,task)
        elif use_verifier_select:
            # P100-style: try all, pick by verifier
            best_conf=-1;best_m=None
            for a_rate,lr,steps in ACTIONS:
                m=copy.deepcopy(encoder)
                if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
                adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
                with torch.no_grad():
                    _,_,_,_,state=adapted(aug_s[0]['nf'].to(DEVICE),aug_s[0]['nn'].to(DEVICE))
                conf=verifier.predict_proba(state.cpu().numpy().flatten().reshape(1,-1))[0,1]
                if conf>best_conf:best_conf=conf;best_m=adapted
            confidences.append(best_conf)
            ok,tot=eval_task_enc(best_m,task)
        elif use_combined:
            # P101 full: Verifier-guided MCTS + verifier re-ranking
            # First get top-3 actions from MCTS
            ba,_=verifier_guided_mcts(encoder,dynamics,pred_head,verifier,
                                       aug_s,n_rollouts=n_rollouts)
            # Then try top actions and re-rank with verifier
            candidates=[]
            top_actions=[ba]
            # Also add 2 runner-up actions
            for a_idx in range(N_ACTIONS):
                if a_idx!=ba and len(top_actions)<3:
                    top_actions.append(a_idx)
            for a_idx in top_actions:
                a_rate,lr,steps=ACTIONS[a_idx]
                m=copy.deepcopy(encoder)
                if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
                adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
                with torch.no_grad():
                    _,_,_,_,state=adapted(aug_s[0]['nf'].to(DEVICE),aug_s[0]['nn'].to(DEVICE))
                conf=verifier.predict_proba(state.cpu().numpy().flatten().reshape(1,-1))[0,1]
                loss=demo_loss_enc(adapted,aug_s)
                score=0.4*conf+0.3*max(0,1.0-loss)+0.3*(1.0 if a_idx==ba else 0.0)
                candidates.append((score,adapted,conf))
            candidates.sort(key=lambda x:-x[0])
            best_m=candidates[0][1];confidences.append(candidates[0][2])
            ok,tot=eval_task_enc(best_m,task)
        else:
            # Baseline: plain MCTS with continuous dynamics (P98 style)
            from phase98_continuous_actions import latent_mcts
            ba,_=latent_mcts(encoder,dynamics,pred_head,aug_s,n_rollouts=n_rollouts)
            a_rate,lr,steps=ACTIONS[ba]
            m=copy.deepcopy(encoder)
            if a_rate>0:m=ablate_l2_enc(m,a_rate,aug_s)
            adapted=adapt_model_enc(m,aug_s,steps=steps,lr=lr)
            ok,tot=eval_task_enc(adapted,task)

        ok_total+=ok;tot_total+=tot
    acc=ok_total/max(tot_total,1)
    avg_conf=float(np.mean(confidences)) if confidences else 0.0
    return acc,avg_conf

def main():
    print("="*60)
    print("Phase 101: Unified Continuous Dynamics + Latent Verifier")
    print("P98 gated dynamics + P100 verifier -> Verifier-Guided MCTS")
    print("="*60)
    t0=time.time()
    tasks=load_arc_tasks(DATA_DIR)
    split=int(len(tasks)*0.8)
    train_tasks=tasks[:split];test_tasks=tasks[split:]
    random.seed(42);np.random.seed(42);torch.manual_seed(42)

    # 1. Train encoder (shared)
    print("\n  [Step 1] Training GNN encoder...")
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
    encoder.eval();print("    Encoder trained.")
    encoder=reptile_meta_train(encoder,train_tasks,n_outer=200,n_inner=10)
    encoder.eval()

    # 2. Collect dynamics data & train continuous dynamics (P98)
    print("\n  [Step 2] Training Continuous Dynamics (P98)...")
    dyn_data=collect_dynamics_data(encoder,train_tasks,n_tasks=40)
    print(f"    {len(dyn_data)} transitions")
    dyn_cont=DynamicsContinuous().to(DEVICE)
    pred_head=PredictionHead().to(DEVICE)
    dyn_cont,pred_head,dyn_loss=train_dynamics(dyn_cont,pred_head,dyn_data,epochs=100)
    print(f"    Dynamics loss: {dyn_loss:.4f}")

    # 3. Train Latent Verifier (P100)
    print("\n  [Step 3] Training Latent Verifier (P100)...")
    verifier=train_verifier(encoder,train_tasks)

    # 4. Evaluate all methods
    print("\n  [Step 4] Evaluation...")
    rollout_n=256

    print("\n  --- Method A: P98 Continuous MCTS (no verifier) ---")
    t1=time.time()
    acc_a,_=eval_method(encoder,test_tasks,dyn_cont,pred_head,verifier,
                        "P98_MCTS",n_rollouts=rollout_n)
    time_a=time.time()-t1
    print(f"    P98 MCTS: {acc_a:.1%} ({time_a:.1f}s)")

    print("\n  --- Method B: P100 Verifier Selection (no dynamics) ---")
    t2=time.time()
    acc_b,conf_b=eval_method(encoder,test_tasks,dyn_cont,pred_head,verifier,
                             "P100_Verifier",use_verifier_select=True)
    time_b=time.time()-t2
    print(f"    P100 Verifier: {acc_b:.1%} (conf={conf_b:.3f}, {time_b:.1f}s)")

    print("\n  --- Method C: P101 Verifier-Guided MCTS ---")
    t3=time.time()
    acc_c,_=eval_method(encoder,test_tasks,dyn_cont,pred_head,verifier,
                        "P101_VMCTS",use_verifier_mcts=True,n_rollouts=rollout_n)
    time_c=time.time()-t3
    print(f"    P101 V-MCTS: {acc_c:.1%} ({time_c:.1f}s)")

    print("\n  --- Method D: P101 Full (V-MCTS + Verifier Re-ranking) ---")
    t4=time.time()
    acc_d,conf_d=eval_method(encoder,test_tasks,dyn_cont,pred_head,verifier,
                             "P101_Full",use_combined=True,n_rollouts=rollout_n)
    time_d=time.time()-t4
    print(f"    P101 Full: {acc_d:.1%} (conf={conf_d:.3f}, {time_d:.1f}s)")

    # 5. Scaling test: more rollouts for P101
    print("\n  [Step 5] Scaling test...")
    scaling_results={}
    for nr in [64,128,256,512,1024]:
        acc_s,_=eval_method(encoder,test_tasks,dyn_cont,pred_head,verifier,
                            f"scale_{nr}",use_verifier_mcts=True,n_rollouts=nr)
        scaling_results[nr]=acc_s
        print(f"    {nr} rollouts -> {acc_s:.1%}")

    # Save results
    results={
        'p98_mcts':{'acc':acc_a,'time':time_a},
        'p100_verifier':{'acc':acc_b,'conf':conf_b,'time':time_b},
        'p101_vmcts':{'acc':acc_c,'time':time_c},
        'p101_full':{'acc':acc_d,'conf':conf_d,'time':time_d},
        'scaling':{str(k):v for k,v in scaling_results.items()},
        'dynamics_loss':dyn_loss,
        'elapsed':time.time()-t0,
        'timestamp':time.strftime('%Y-%m-%dT%H:%M:%S'),
    }
    with open(os.path.join(RESULTS_DIR,'phase101_unified.json'),'w',encoding='utf-8') as f:
        json.dump(results,f,indent=2,ensure_ascii=False)

    # Plot
    fig,axes=plt.subplots(1,2,figsize=(14,6))

    # Left: method comparison
    ax=axes[0]
    methods=['P98\nMCTS','P100\nVerifier','P101\nV-MCTS','P101\nFull']
    accs=[acc_a,acc_b,acc_c,acc_d]
    colors=['#9E9E9E','#2196F3','#E91E63','#FFD700']
    bars=ax.bar(methods,accs,color=colors,edgecolor='black',linewidth=0.5)
    for bar,acc in zip(bars,accs):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.005,
                f'{acc:.1%}',ha='center',va='bottom',fontweight='bold',fontsize=13)
    ax.set_ylabel('Accuracy',fontsize=13)
    ax.set_ylim(0.7,1.0);ax.grid(True,alpha=0.3,axis='y')
    ax.set_title('Method Comparison',fontweight='bold',fontsize=13)

    # Right: scaling
    ax2=axes[1]
    rolls=sorted(scaling_results.keys())
    ax2.plot(rolls,[scaling_results[r]for r in rolls],'s-',lw=3,ms=10,color='#E91E63',
             label='P101 V-MCTS')
    ax2.axhline(y=acc_b,color='#2196F3',ls='--',lw=2,alpha=0.7,label=f'P100 Verifier ({acc_b:.1%})')
    ax2.set_xlabel('Rollouts',fontsize=12);ax2.set_ylabel('Accuracy',fontsize=12)
    ax2.set_xscale('log',base=2);ax2.set_xticks(rolls)
    ax2.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax2.legend();ax2.grid(True,alpha=0.3);ax2.set_ylim(0.7,1.0)
    ax2.set_title('Scaling with Rollouts',fontweight='bold',fontsize=13)

    plt.suptitle('Phase 101: Continuous Dynamics + Latent Verifier',fontsize=15,fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR,'phase101_unified.png'),dpi=150);plt.close()

    best=max(accs)
    print(f"\nPhase 101 complete! Best: {best:.1%}")
    print(f"Elapsed: {results['elapsed']:.1f}s")
    return results

if __name__=='__main__':main()
