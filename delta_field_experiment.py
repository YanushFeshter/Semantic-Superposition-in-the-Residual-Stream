#!/usr/bin/env python3
"""
Δ-Field Crucial Experiment: Does computational velocity correlate with decision dynamics?

Tests the core hypothesis of the Perceptual Interpretability Engine proposal:
that differential fields in the residual stream contain information about
the model's decision process during ambiguous prompts.
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr, mannwhitneyu

torch.set_grad_enabled(False)

OUTDIR = Path("delta_field_results")
OUTDIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"

# --- Prompts ---

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
    {"prompt": "Water freezes at zero degrees", "A": [" Celsius", " C"], "B": [" Fahrenheit", " F"]},
    {"prompt": "The cat sat on the", "A": [" mat", " floor", " bed"], "B": [" roof", " car", " table"]},
    {"prompt": "Two plus two equals", "A": [" four", " 4"], "B": [" three", " five", " 3"]},
    {"prompt": "The sun rises in the", "A": [" east", " morning"], "B": [" west", " north", " south"]},
    {"prompt": "She opened the door and", "A": [" walked", " stepped", " went"], "B": [" closed", " locked", " slammed"]},
    {"prompt": "The baby was crying because", "A": [" she", " he", " it"], "B": [" the", " they", " we"]},
    {"prompt": "He picked up the phone and", "A": [" called", " dialed", " said"], "B": [" threw", " dropped", " broke"]},
    {"prompt": "The sky was blue and the", "A": [" sun", " clouds", " air"], "B": [" ground", " trees", " grass"]},
    {"prompt": "I went to the store to buy", "A": [" some", " a", " the"], "B": [" nothing", " everything", " all"]},
    {"prompt": "The dog chased the", "A": [" cat", " ball", " rabbit"], "B": [" car", " bird", " squirrel"]},
    {"prompt": "He read the book and then", "A": [" went", " fell", " put"], "B": [" ate", " ran", " drove"]},
]


def get_single_token_ids(model, token_strings):
    """Return list of token IDs for strings that tokenize to exactly one token."""
    valid = []
    for s in token_strings:
        toks = model.to_tokens(s, prepend_bos=False).squeeze()
        if toks.dim() == 0:  # single token
            valid.append(int(toks.item()))
        elif toks.shape[0] == 1:
            valid.append(int(toks[0].item()))
    return valid


def run_experiment(model, prompts, label):
    """Run the Δ-field experiment on a set of prompts."""
    results = []
    n_layers = model.cfg.n_layers  # 12 for GPT-2 small
    
    for i, p in enumerate(prompts):
        prompt_text = p["prompt"]
        
        # Get valid single-token IDs for each group
        ids_a = get_single_token_ids(model, p["A"])
        ids_b = get_single_token_ids(model, p["B"])
        
        if len(ids_a) == 0 or len(ids_b) == 0:
            print(f"  Skipping '{prompt_text}': no valid single-token candidates")
            continue
        
        # Run model and cache all residual states
        tokens = model.to_tokens(prompt_text, prepend_bos=True)
        _, cache = model.run_with_cache(tokens)
        
        last_pos = tokens.shape[1] - 1
        
        # Extract residual stream at each layer (post) at last token position
        states = []
        for layer in range(n_layers):
            s = cache[f"blocks.{layer}.hook_resid_post"][0, last_pos, :].cpu().numpy()
            states.append(s)
        states = np.array(states)  # shape: (n_layers, d_model)
        
        # --- Computational Velocity ---
        deltas = np.diff(states, axis=0)  # (n_layers-1, d_model)
        velocity = np.linalg.norm(deltas, axis=1)  # (n_layers-1,)
        curvature = np.abs(np.diff(velocity))  # (n_layers-2,)
        
        # --- Decision Signal (logit lens) ---
        # TransformerLens folds layernorm into W_U by default
        # So we apply ln_final (just centering+scaling) then multiply by W_U
        W_U = model.W_U.cpu().numpy()  # (d_model, vocab)
        
        delta_logits = []
        for layer in range(n_layers):
            s = states[layer]
            # Apply layernorm pre (center and scale)
            s_norm = (s - s.mean()) / (s.std() + 1e-8)
            logits = s_norm @ W_U
            
            max_a = max(logits[tid] for tid in ids_a)
            max_b = max(logits[tid] for tid in ids_b)
            delta_logits.append(abs(max_a - max_b))
        
        delta_logits = np.array(delta_logits)  # (n_layers,)
        
        # Decision rate: centered finite difference on interior layers
        # Gives values for layers 1..10 (10 values), aligned with curvature
        decision_rate = np.abs(delta_logits[2:] - delta_logits[:-2]) / 2.0  # (n_layers-2,)
        
        # --- Attention Divergence ---
        attn_divs = []
        for layer in range(n_layers):
            # attention pattern: (n_heads, seq_len, seq_len)
            attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, last_pos, :].cpu().numpy()
            # attn shape: (n_heads, seq_len)
            n_heads = attn.shape[0]
            
            # Mean pairwise KL divergence (symmetrized)
            kl_sum = 0.0
            count = 0
            for h1 in range(n_heads):
                for h2 in range(h1 + 1, n_heads):
                    p_h1 = attn[h1] + 1e-10
                    p_h2 = attn[h2] + 1e-10
                    p_h1 = p_h1 / p_h1.sum()
                    p_h2 = p_h2 / p_h2.sum()
                    kl_12 = np.sum(p_h1 * np.log(p_h1 / p_h2))
                    kl_21 = np.sum(p_h2 * np.log(p_h2 / p_h1))
                    kl_sum += (kl_12 + kl_21) / 2.0
                    count += 1
            attn_divs.append(kl_sum / count if count > 0 else 0.0)
        
        attn_divs = np.array(attn_divs)  # (n_layers,)
        # Align to curvature: interior layers 1..10
        attn_divs_aligned = attn_divs[1:-1]  # (n_layers-2,)
        
        results.append({
            "prompt": prompt_text,
            "label": label,
            "velocity": velocity.tolist(),
            "curvature": curvature.tolist(),
            "delta_logits": delta_logits.tolist(),
            "decision_rate": decision_rate.tolist(),
            "attn_divergence": attn_divs_aligned.tolist(),
        })
        
        print(f"  [{label}] '{prompt_text}' — peak v at layer {np.argmax(velocity)}, peak κ at layer {np.argmax(curvature)+1}")
    
    return results


def safe_corr(x, y, method="spearman"):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    if method == "spearman":
        return spearmanr(x, y)
    return pearsonr(x, y)


def main():
    from transformer_lens import HookedTransformer
    
    print(f"Loading GPT-2 small on {DEVICE}...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    model.eval()
    
    print(f"\n--- Running ambiguous prompts ({len(AMBIGUOUS)}) ---")
    amb_results = run_experiment(model, AMBIGUOUS, "ambiguous")
    
    print(f"\n--- Running control prompts ({len(CONTROLS)}) ---")
    ctl_results = run_experiment(model, CONTROLS, "control")
    
    print(f"\nValid: {len(amb_results)} ambiguous, {len(ctl_results)} control")
    
    # === PRIMARY ANALYSIS ===
    # Pool all (curvature, decision_rate) pairs across ambiguous prompts
    amb_kappa = np.concatenate([r["curvature"] for r in amb_results])
    amb_dr = np.concatenate([r["decision_rate"] for r in amb_results])
    
    sp_r, sp_p = safe_corr(amb_kappa, amb_dr, "spearman")
    pe_r, pe_p = safe_corr(amb_kappa, amb_dr, "pearson")
    
    # === CONTROL ANALYSIS ===
    ctl_kappa = np.concatenate([r["curvature"] for r in ctl_results])
    ctl_dr = np.concatenate([r["decision_rate"] for r in ctl_results])
    
    ctl_sp_r, ctl_sp_p = safe_corr(ctl_kappa, ctl_dr, "spearman")
    
    # === AMBIGUOUS vs CONTROL peak curvature ===
    amb_peaks = [max(r["curvature"]) for r in amb_results]
    ctl_peaks = [max(r["curvature"]) for r in ctl_results]
    mw_u, mw_p = mannwhitneyu(amb_peaks, ctl_peaks, alternative="greater")
    
    # === ATTENTION CORRELATION ===
    amb_kappa_att = np.concatenate([r["curvature"] for r in amb_results])
    amb_att = np.concatenate([r["attn_divergence"] for r in amb_results])
    att_sp_r, att_sp_p = safe_corr(amb_kappa_att, amb_att, "spearman")
    
    # === VELOCITY PEAK LAYERS ===
    amb_vpeak = [np.argmax(r["velocity"]) for r in amb_results]
    ctl_vpeak = [np.argmax(r["velocity"]) for r in ctl_results]
    
    # === VERDICT ===
    if sp_r > 0.3 and sp_p < 0.05:
        verdict = "SUPPORTED"
    elif sp_r < 0.1 or sp_p > 0.1:
        verdict = "NOT SUPPORTED"
    else:
        verdict = "MIXED"
    
    # === PRINT RESULTS ===
    print("\n" + "=" * 70)
    print("CRUCIAL EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"\nPRIMARY ANALYSIS (ambiguous prompts, pooled across layers)")
    print(f"  N data points: {len(amb_kappa)}")
    print(f"  Pearson  r = {pe_r:.4f}, p = {pe_p:.4g}")
    print(f"  Spearman ρ = {sp_r:.4f}, p = {sp_p:.4g}")
    print(f"  >>> VERDICT: {verdict}")
    print(f"\nCONTROL ANALYSIS")
    print(f"  Spearman ρ = {ctl_sp_r:.4f}, p = {ctl_sp_p:.4g}")
    print(f"\nAMBIGUOUS vs CONTROL (peak curvature)")
    print(f"  Ambiguous mean peak κ: {np.mean(amb_peaks):.4f} ± {np.std(amb_peaks):.4f}")
    print(f"  Control mean peak κ:   {np.mean(ctl_peaks):.4f} ± {np.std(ctl_peaks):.4f}")
    print(f"  Mann-Whitney U (amb > ctl): U={mw_u:.1f}, p={mw_p:.4g}")
    print(f"\nATTENTION-CURVATURE CORRELATION (ambiguous)")
    print(f"  Spearman ρ(κ, attn_div) = {att_sp_r:.4f}, p = {att_sp_p:.4g}")
    print(f"\nVELOCITY PEAK LAYERS")
    print(f"  Ambiguous: {dict(zip(*np.unique(amb_vpeak, return_counts=True)))}")
    print(f"  Control:   {dict(zip(*np.unique(ctl_vpeak, return_counts=True)))}")
    print("=" * 70)
    
    # === PLOTS ===
    
    # 1. Velocity profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in amb_results:
        ax1.plot(r["velocity"], alpha=0.3, color="steelblue")
    ax1.plot(np.mean([r["velocity"] for r in amb_results], axis=0), color="darkblue", lw=2, label="Mean")
    ax1.set_title("Velocity profiles — Ambiguous")
    ax1.set_xlabel("Layer transition (L→L+1)")
    ax1.set_ylabel("||Δs_L||")
    ax1.legend()
    
    for r in ctl_results:
        ax2.plot(r["velocity"], alpha=0.3, color="coral")
    ax2.plot(np.mean([r["velocity"] for r in ctl_results], axis=0), color="darkred", lw=2, label="Mean")
    ax2.set_title("Velocity profiles — Control")
    ax2.set_xlabel("Layer transition (L→L+1)")
    ax2.set_ylabel("||Δs_L||")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "velocity_profiles.png", dpi=150)
    plt.close()
    
    # 2. Curvature profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in amb_results:
        ax1.plot(range(1, len(r["curvature"])+1), r["curvature"], alpha=0.3, color="steelblue")
    mean_curv = np.mean([r["curvature"] for r in amb_results], axis=0)
    ax1.plot(range(1, len(mean_curv)+1), mean_curv, color="darkblue", lw=2, label="Mean")
    ax1.set_title("Curvature profiles — Ambiguous")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("|v_{L+1} - v_L|")
    ax1.legend()
    
    for r in ctl_results:
        ax2.plot(range(1, len(r["curvature"])+1), r["curvature"], alpha=0.3, color="coral")
    mean_curv_c = np.mean([r["curvature"] for r in ctl_results], axis=0)
    ax2.plot(range(1, len(mean_curv_c)+1), mean_curv_c, color="darkred", lw=2, label="Mean")
    ax2.set_title("Curvature profiles — Control")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("|v_{L+1} - v_L|")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "curvature_profiles.png", dpi=150)
    plt.close()
    
    # 3. κ vs decision rate scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(amb_kappa, amb_dr, alpha=0.3, s=15, color="steelblue", label="Ambiguous")
    ax.scatter(ctl_kappa, ctl_dr, alpha=0.3, s=15, color="coral", label="Control")
    ax.set_xlabel("Curvature κ_L")
    ax.set_ylabel("Decision rate |d(Δlogit)/dL|")
    ax.set_title(f"κ vs Decision Rate\nSpearman ρ={sp_r:.3f} (p={sp_p:.3g}) — {verdict}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "kappa_vs_decision_rate.png", dpi=150)
    plt.close()
    
    # 4. κ vs attention divergence
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(amb_kappa_att, amb_att, alpha=0.3, s=15, color="steelblue")
    ax.set_xlabel("Curvature κ_L")
    ax.set_ylabel("Mean pairwise attention KL divergence")
    ax.set_title(f"κ vs Attention Divergence\nSpearman ρ={att_sp_r:.3f} (p={att_sp_p:.3g})")
    plt.tight_layout()
    plt.savefig(OUTDIR / "kappa_vs_attention.png", dpi=150)
    plt.close()
    
    # === SAVE JSON ===
    summary = {
        "primary": {"spearman_r": sp_r, "spearman_p": sp_p, "pearson_r": pe_r, "pearson_p": pe_p, "verdict": verdict, "n_points": len(amb_kappa)},
        "control": {"spearman_r": ctl_sp_r, "spearman_p": ctl_sp_p},
        "amb_vs_ctl_peak_kappa": {"amb_mean": float(np.mean(amb_peaks)), "ctl_mean": float(np.mean(ctl_peaks)), "mw_u": float(mw_u), "mw_p": float(mw_p)},
        "attention": {"spearman_r": att_sp_r, "spearman_p": att_sp_p},
    }
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPlots and data saved to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
