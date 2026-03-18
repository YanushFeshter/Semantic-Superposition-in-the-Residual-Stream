#!/usr/bin/env python3
"""
Experiment 3: Does variance of decision-axis velocity distinguish
ambiguous from control prompts?

Hypothesis: Ambiguous prompts produce higher cross-prompt variance
in v_parallel (projection onto decision axis) than controls,
because the model maintains competing interpretations (superposition).
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu, spearmanr

torch.set_grad_enabled(False)

OUTDIR = Path("variance_results")
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
    vecs = [W_U[:, tid] for tid in token_ids]
    return np.mean(vecs, axis=0)


def extract_v_parallel(model, prompts):
    """Extract v_parallel (decision-axis projection) for each prompt at each layer transition."""
    n_layers = model.cfg.n_layers
    all_vp = []  # list of arrays, each (n_layers-1,)
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
        
        deltas = np.diff(states, axis=0)
        vp = np.array([np.dot(ds, D_hat) for ds in deltas])
        
        all_vp.append(vp)
        valid_prompts.append(p["prompt"])
    
    return np.array(all_vp), valid_prompts  # (n_prompts, n_layers-1)


def main():
    from transformer_lens import HookedTransformer
    
    print("Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    model.eval()
    
    print("Extracting ambiguous...")
    amb_vp, amb_names = extract_v_parallel(model, AMBIGUOUS)
    print("Extracting controls...")
    ctl_vp, ctl_names = extract_v_parallel(model, CONTROLS)
    
    n_transitions = amb_vp.shape[1]  # 11
    
    print(f"\nValid: {len(amb_names)} ambiguous, {len(ctl_names)} control")
    print(f"Transitions per prompt: {n_transitions}")
    
    # === CORE TEST: Variance of v_parallel at each layer ===
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: VARIANCE OF DECISION-AXIS VELOCITY")
    print("=" * 70)
    
    print(f"\n{'Layer':>8} {'Var(amb)':>12} {'Var(ctl)':>12} {'Ratio':>8} {'Note':>15}")
    print("-" * 60)
    
    amb_vars = []
    ctl_vars = []
    
    for t in range(n_transitions):
        v_amb = amb_vp[:, t]  # all ambiguous prompts at this transition
        v_ctl = ctl_vp[:, t]  # all control prompts at this transition
        
        var_a = np.var(v_amb)
        var_c = np.var(v_ctl)
        ratio = var_a / var_c if var_c > 1e-10 else float('inf')
        
        amb_vars.append(var_a)
        ctl_vars.append(var_c)
        
        note = "AMB >> CTL" if ratio > 2 else ("AMB > CTL" if ratio > 1.2 else ("~EQUAL" if ratio > 0.8 else "CTL > AMB"))
        print(f"{t:>5}→{t+1:<3} {var_a:>12.4f} {var_c:>12.4f} {ratio:>8.2f} {note:>15}")
    
    amb_vars = np.array(amb_vars)
    ctl_vars = np.array(ctl_vars)
    
    # Overall: is amb variance consistently higher?
    print(f"\n=== SUMMARY ===")
    print(f"Mean Var(amb): {np.mean(amb_vars):.4f}")
    print(f"Mean Var(ctl): {np.mean(ctl_vars):.4f}")
    print(f"Mean ratio:    {np.mean(amb_vars / (ctl_vars + 1e-10)):.2f}")
    
    # Per-prompt total variance (across all layers)
    amb_total_var = np.var(amb_vp, axis=1)  # variance across layers for each prompt
    ctl_total_var = np.var(ctl_vp, axis=1)
    
    # Alternative: mean absolute v_parallel (how much the prompt moves on decision axis)
    amb_mean_abs = np.mean(np.abs(amb_vp), axis=1)
    ctl_mean_abs = np.mean(np.abs(ctl_vp), axis=1)
    
    print(f"\n=== PER-PROMPT TRAJECTORY VARIANCE (across layers) ===")
    print(f"Ambiguous: mean = {np.mean(amb_total_var):.4f} ± {np.std(amb_total_var):.4f}")
    print(f"Control:   mean = {np.mean(ctl_total_var):.4f} ± {np.std(ctl_total_var):.4f}")
    u, p = mannwhitneyu(amb_total_var, ctl_total_var, alternative="two-sided")
    print(f"Mann-Whitney: U={u:.1f}, p={p:.4g}")
    
    print(f"\n=== CROSS-PROMPT VARIANCE AT EACH LAYER (the superposition test) ===")
    # For each layer, compute variance ACROSS prompts
    # This is the key measure: how much do different prompts diverge at each layer?
    
    for t in range(n_transitions):
        v_amb = amb_vp[:, t]
        v_ctl = ctl_vp[:, t]
        
        # Bootstrap test for variance difference
        n_boot = 5000
        observed_diff = np.var(v_amb) - np.var(v_ctl)
        
        # Pool and permute
        pooled = np.concatenate([v_amb, v_ctl])
        n_a = len(v_amb)
        count_extreme = 0
        for _ in range(n_boot):
            perm = np.random.permutation(pooled)
            var_a = np.var(perm[:n_a])
            var_c = np.var(perm[n_a:])
            if var_a - var_c >= observed_diff:
                count_extreme += 1
        p_boot = count_extreme / n_boot
        
        sig = "***" if p_boot < 0.001 else ("**" if p_boot < 0.01 else ("*" if p_boot < 0.05 else ""))
        print(f"  Layer {t}→{t+1}: Var_diff = {observed_diff:+.4f}, bootstrap p = {p_boot:.4f} {sig}")
    
    # === VERDICT ===
    # Count how many layers show amb > ctl with p < 0.05
    print(f"\n=== OVERALL VERDICT ===")
    late_amb = np.mean(amb_vars[7:])  # layers 7-10
    late_ctl = np.mean(ctl_vars[7:])
    mid_amb = np.mean(amb_vars[3:7])   # layers 3-6
    mid_ctl = np.mean(ctl_vars[3:7])
    
    print(f"Mid-layers (3-6):  Var ratio amb/ctl = {mid_amb/(mid_ctl+1e-10):.2f}")
    print(f"Late layers (7-10): Var ratio amb/ctl = {late_amb/(late_ctl+1e-10):.2f}")
    
    # === PLOT ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Variance profiles
    ax = axes[0]
    ax.plot(range(n_transitions), amb_vars, 'o-', color="steelblue", label="Ambiguous", lw=2)
    ax.plot(range(n_transitions), ctl_vars, 'o-', color="coral", label="Control", lw=2)
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("Cross-prompt Var(v_∥)")
    ax.set_title("Superposition Signal:\nCross-prompt variance of decision velocity")
    ax.legend()
    
    # 2. Ratio
    ax = axes[1]
    ratios = amb_vars / (ctl_vars + 1e-10)
    ax.bar(range(n_transitions), ratios, color=["steelblue" if r > 1 else "coral" for r in ratios])
    ax.axhline(y=1, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("Var(amb) / Var(ctl)")
    ax.set_title("Variance Ratio by Layer")
    
    # 3. Superimposed trajectories
    ax = axes[2]
    for vp in amb_vp:
        ax.plot(range(n_transitions), vp, alpha=0.25, color="steelblue")
    for vp in ctl_vp:
        ax.plot(range(n_transitions), vp, alpha=0.35, color="coral")
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax.set_xlabel("Layer transition")
    ax.set_ylabel("v_∥ (projection onto decision axis)")
    ax.set_title("All trajectories overlaid\n(blue=ambiguous, red=control)")
    
    plt.tight_layout()
    plt.savefig(OUTDIR / "superposition_analysis.png", dpi=150)
    plt.close()
    
    # Save
    summary = {
        "amb_cross_prompt_var_by_layer": amb_vars.tolist(),
        "ctl_cross_prompt_var_by_layer": ctl_vars.tolist(),
        "ratio_by_layer": (amb_vars / (ctl_vars + 1e-10)).tolist(),
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
