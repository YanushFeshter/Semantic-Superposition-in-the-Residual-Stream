#!/usr/bin/env python3
"""
Pre-publication tests:
1. FDR correction (Benjamini-Hochberg) on per-layer p-values
2. Random direction control: is the variance effect specific to the semantic
   decision axis, or does it appear on random axes too?
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

torch.set_grad_enabled(False)

OUTDIR = Path("prepub_tests")
OUTDIR.mkdir(parents=True, exist_ok=True)

AMBIGUOUS = [
    {"prompt": "A bat is a", "A": [" mammal", " creature", " animal"], "B": [" club", " tool", " stick"]},
    {"prompt": "The bank was", "A": [" flooded", " muddy", " steep"], "B": [" robbed", " closed", " bankrupt"]},
    {"prompt": "The bark was", "A": [" loud", " sharp", " scary"], "B": [" rough", " brown", " thick"]},
    {"prompt": "The match was", "A": [" close", " tied", " intense"], "B": [" lit", " damp", " broken"]},
    {"prompt": "The ring was", "A": [" gold", " shiny", " beautiful"], "B": [" crowded", " noisy", " square"]},
    {"prompt": "The pitcher was", "A": [" tired", " wild", " dominant"], "B": [" empty", " cracked", " full"]},
    {"prompt": "The pen was", "A": [" leaking", " blue", " black"], "B": [" fenced", " muddy", " crowded"]},
    {"prompt": "The trunk was", "A": [" locked", " heavy", " packed"], "B": [" long", " grey", " wrinkled"]},
    {"prompt": "The crane was", "A": [" flying", " white", " tall"], "B": [" lifting", " steel", " broken"]},
    {"prompt": "The seal was", "A": [" swimming", " spotted", " fat"], "B": [" broken", " tight", " official"]},
    {"prompt": "The draft was", "A": [" cold", " chilly", " strong"], "B": [" revised", " signed", " rough"]},
    {"prompt": "The jam was", "A": [" sweet", " sticky", " red"], "B": [" severe", " endless", " bad"]},
    {"prompt": "The club was", "A": [" private", " social", " exclusive"], "B": [" heavy", " wooden", " blunt"]},
    {"prompt": "The spring was", "A": [" warm", " rainy", " early"], "B": [" steel", " broken", " tight"]},
    {"prompt": "The current was", "A": [" strong", " swift", " dangerous"], "B": [" electric", " measured", " low"]},
    {"prompt": "The mouse was", "A": [" furry", " small", " hiding"], "B": [" wireless", " broken", " clicking"]},
    {"prompt": "The organ was", "A": [" vital", " damaged", " healthy"], "B": [" tuned", " loud", " wooden"]},
    {"prompt": "The chip was", "A": [" crunchy", " salty", " fried"], "B": [" silicon", " tiny", " damaged"]},
    {"prompt": "The palm was", "A": [" sweaty", " open", " flat"], "B": [" tropical", " tall", " green"]},
    {"prompt": "The port was", "A": [" busy", " crowded", " naval"], "B": [" digital", " open", " blocked"]},
    {"prompt": "The cell was", "A": [" tiny", " dark", " locked"], "B": [" biological", " dividing", " healthy"]},
    {"prompt": "The bolt was", "A": [" rusty", " tight", " steel"], "B": [" sudden", " bright", " loud"]},
    {"prompt": "The ball was", "A": [" round", " bouncing", " red"], "B": [" formal", " elegant", " grand"]},
    {"prompt": "The nail was", "A": [" rusty", " bent", " iron"], "B": [" painted", " broken", " long"]},
]

CONTROLS = [
    {"prompt": "The capital of France is", "A": [" Paris"], "B": [" London", " Berlin", " Madrid"]},
    {"prompt": "The cat sat on the", "A": [" mat", " floor", " bed"], "B": [" roof", " car", " table"]},
    {"prompt": "Two plus two equals", "A": [" four", " 4"], "B": [" three", " five", " 3"]},
    {"prompt": "The sun rises in the", "A": [" east", " morning"], "B": [" west", " north", " south"]},
    {"prompt": "She opened the door and", "A": [" walked", " stepped", " went"], "B": [" closed", " locked", " slammed"]},
    {"prompt": "The baby was crying because", "A": [" she", " he", " it"], "B": [" the", " they", " we"]},
    {"prompt": "He picked up the phone and", "A": [" called", " dialed", " said"], "B": [" threw", " dropped", " broke"]},
    {"prompt": "The sky was blue and the", "A": [" sun", " clouds", " air"], "B": [" ground", " trees", " grass"]},
    {"prompt": "I went to the store to buy", "A": [" some", " a", " the"], "B": [" nothing", " everything", " all"]},
    {"prompt": "The dog chased the", "A": [" cat", " ball", " rabbit"], "B": [" car", " bird", " squirrel"]},
]


def get_single_token_ids(model, token_strings):
    valid = []
    for s in token_strings:
        toks = model.to_tokens(s, prepend_bos=False).squeeze()
        if toks.dim() == 0:
            valid.append(int(toks.item()))
        elif toks.shape[0] == 1:
            valid.append(int(toks[0].item()))
    return valid


def get_mean_embedding(model, token_ids):
    W_U = model.W_U.cpu().numpy()
    return np.mean([W_U[:, tid] for tid in token_ids], axis=0)


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply BH FDR correction. Returns adjusted p-values and rejection mask."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_idx]
    
    # Adjusted p-values
    adjusted = np.zeros(n)
    adjusted[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        adjusted[sorted_idx[i]] = min(adjusted[sorted_idx[i + 1]], sorted_p[i] * n / (i + 1))
    
    rejected = adjusted < alpha
    return adjusted, rejected


def main():
    from transformer_lens import HookedTransformer
    
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    
    # === EXTRACT ALL DATA ===
    
    def extract_trajectories(prompts, label):
        all_vp_semantic = []  # v_parallel on semantic axis
        all_deltas = []       # raw deltas for random projection
        valid_prompts = []
        
        for p in prompts:
            ids_a = get_single_token_ids(model, p["A"])
            ids_b = get_single_token_ids(model, p["B"])
            if not ids_a or not ids_b:
                continue
            
            E_A = get_mean_embedding(model, ids_a)
            E_B = get_mean_embedding(model, ids_b)
            D = E_A - E_B
            D_norm = np.linalg.norm(D)
            if D_norm < 1e-10:
                continue
            D_hat = D / D_norm
            
            tokens = model.to_tokens(p["prompt"], prepend_bos=True)
            _, cache = model.run_with_cache(tokens)
            last_pos = tokens.shape[1] - 1
            
            states = []
            for layer in range(n_layers):
                s = cache[f"blocks.{layer}.hook_resid_post"][0, last_pos, :].cpu().numpy()
                states.append(s)
            states = np.array(states)
            deltas = np.diff(states, axis=0)  # (n_layers-1, d_model)
            
            vp = np.array([np.dot(ds, D_hat) for ds in deltas])
            all_vp_semantic.append(vp)
            all_deltas.append(deltas)
            valid_prompts.append(p["prompt"])
        
        return np.array(all_vp_semantic), all_deltas, valid_prompts
    
    print("\nExtracting ambiguous...")
    amb_vp, amb_deltas, amb_names = extract_trajectories(AMBIGUOUS, "amb")
    print(f"  Valid: {len(amb_names)}")
    
    print("Extracting controls...")
    ctl_vp, ctl_deltas, ctl_names = extract_trajectories(CONTROLS, "ctl")
    print(f"  Valid: {len(ctl_names)}")
    
    n_transitions = amb_vp.shape[1]
    
    # ============================================================
    # TEST 1: FDR CORRECTION (Benjamini-Hochberg)
    # ============================================================
    
    print("\n" + "=" * 70)
    print("TEST 1: FDR CORRECTION ON PER-LAYER VARIANCE TESTS")
    print("=" * 70)
    
    raw_p_values = []
    raw_ratios = []
    
    for t in range(n_transitions):
        v_a = amb_vp[:, t]
        v_c = ctl_vp[:, t]
        va, vc = np.var(v_a), np.var(v_c)
        raw_ratios.append(va / (vc + 1e-15))
        
        # Bootstrap test
        observed = va - vc
        pooled = np.concatenate([v_a, v_c])
        n_a = len(v_a)
        count = 0
        n_boot = 5000
        for _ in range(n_boot):
            perm = np.random.permutation(pooled)
            if np.var(perm[:n_a]) - np.var(perm[n_a:]) >= observed:
                count += 1
        p = count / n_boot
        # Avoid p=0 for FDR
        p = max(p, 1.0 / n_boot)
        raw_p_values.append(p)
    
    adjusted_p, rejected = benjamini_hochberg(raw_p_values, alpha=0.05)
    
    print(f"\n{'Trans':>8} {'Ratio':>8} {'raw_p':>10} {'adj_p':>10} {'Sig?':>6}")
    print("-" * 50)
    for t in range(n_transitions):
        sig = "YES" if rejected[t] else "no"
        print(f"  {t:>3}→{t+1:<3} {raw_ratios[t]:>8.2f} {raw_p_values[t]:>10.4f} {adjusted_p[t]:>10.4f} {sig:>6}")
    
    n_sig = sum(rejected)
    print(f"\nLayers surviving FDR correction: {n_sig} / {n_transitions}")
    if n_sig > 0:
        sig_layers = [t for t in range(n_transitions) if rejected[t]]
        print(f"Significant layers: {[f'{t}→{t+1}' for t in sig_layers]}")
    
    # ============================================================
    # TEST 2: RANDOM DIRECTION CONTROL
    # ============================================================
    
    print("\n" + "=" * 70)
    print("TEST 2: SEMANTIC vs RANDOM DIRECTION CONTROL")
    print("=" * 70)
    
    n_random = 100  # number of random directions to test
    np.random.seed(42)
    
    # For each random direction, compute cross-prompt variance ratio
    random_mean_ratios = []
    
    for r in range(n_random):
        # Random unit vector in d_model space
        rand_dir = np.random.randn(d_model)
        rand_dir = rand_dir / np.linalg.norm(rand_dir)
        
        # Project all prompts onto this random direction
        amb_vp_rand = np.array([[np.dot(deltas[t], rand_dir) for t in range(n_transitions)] 
                                for deltas in amb_deltas])
        ctl_vp_rand = np.array([[np.dot(deltas[t], rand_dir) for t in range(n_transitions)] 
                                for deltas in ctl_deltas])
        
        # Mean variance ratio across layers
        ratios = []
        for t in range(n_transitions):
            va = np.var(amb_vp_rand[:, t])
            vc = np.var(ctl_vp_rand[:, t])
            ratios.append(va / (vc + 1e-15))
        random_mean_ratios.append(np.mean(ratios))
    
    # Semantic direction mean ratio
    semantic_ratios = []
    for t in range(n_transitions):
        va = np.var(amb_vp[:, t])
        vc = np.var(ctl_vp[:, t])
        semantic_ratios.append(va / (vc + 1e-15))
    semantic_mean_ratio = np.mean(semantic_ratios)
    
    random_mean = np.mean(random_mean_ratios)
    random_std = np.std(random_mean_ratios)
    random_95 = np.percentile(random_mean_ratios, 95)
    random_99 = np.percentile(random_mean_ratios, 99)
    
    # How many random directions give a ratio >= semantic?
    n_exceed = sum(1 for r in random_mean_ratios if r >= semantic_mean_ratio)
    p_random = n_exceed / n_random
    
    print(f"\nSemantic axis mean ratio:  {semantic_mean_ratio:.4f}")
    print(f"Random axes mean ratio:    {random_mean:.4f} ± {random_std:.4f}")
    print(f"Random 95th percentile:    {random_95:.4f}")
    print(f"Random 99th percentile:    {random_99:.4f}")
    print(f"Random axes exceeding semantic: {n_exceed}/{n_random}")
    print(f"Empirical p-value: {p_random:.4f}")
    
    if semantic_mean_ratio > random_99:
        verdict = "SEMANTIC AXIS IS SPECIAL (> 99th percentile of random)"
    elif semantic_mean_ratio > random_95:
        verdict = "SEMANTIC AXIS IS SPECIAL (> 95th percentile of random)"
    elif semantic_mean_ratio > random_mean + 2 * random_std:
        verdict = "SEMANTIC AXIS IS ELEVATED (> 2σ above random mean)"
    else:
        verdict = "SEMANTIC AXIS IS NOT DISTINGUISHABLE FROM RANDOM"
    
    print(f"\n>>> VERDICT: {verdict}")
    
    # === PLOT ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: FDR results
    colors = ["darkgreen" if rejected[t] else ("orange" if raw_p_values[t] < 0.05 else "lightgray") 
              for t in range(n_transitions)]
    ax1.bar(range(n_transitions), raw_ratios, color=colors)
    ax1.axhline(1, color="black", ls="--", alpha=0.5)
    ax1.set_xlabel("Layer transition")
    ax1.set_ylabel("Var(amb) / Var(ctl)")
    ax1.set_title("Variance Ratio by Layer\n(green = survives FDR, orange = raw p<0.05, gray = n.s.)")
    
    # Plot 2: Random direction distribution
    ax2.hist(random_mean_ratios, bins=30, color="lightblue", edgecolor="steelblue", alpha=0.7, label="Random directions")
    ax2.axvline(semantic_mean_ratio, color="red", lw=2.5, label=f"Semantic axis ({semantic_mean_ratio:.2f})")
    ax2.axvline(random_95, color="orange", ls="--", label=f"95th pctl ({random_95:.2f})")
    ax2.axvline(random_99, color="darkred", ls="--", label=f"99th pctl ({random_99:.2f})")
    ax2.set_xlabel("Mean variance ratio (amb/ctl)")
    ax2.set_ylabel("Count")
    ax2.set_title("Semantic vs Random Direction Control\n(100 random directions)")
    ax2.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "prepub_tests.png", dpi=150)
    plt.close()
    
    print(f"\nPlot saved to {OUTDIR / 'prepub_tests.png'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
