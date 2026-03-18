#!/usr/bin/env python3
"""
Experiment 2: Semantic Orthogonal Velocity (SOV)
Tests whether projecting residual stream changes onto a prompt-specific
decision axis produces a signal that correlates with decision dynamics
WITHOUT the architectural confound that killed Experiment 1.
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from numpy.polynomial.polynomial import polyfit, polyval

torch.set_grad_enabled(False)

OUTDIR = Path("sov_results")
OUTDIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cpu"

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
    """Get mean unembedding vector for a group of tokens."""
    W_U = model.W_U.cpu().numpy()  # (d_model, vocab)
    vecs = [W_U[:, tid] for tid in token_ids]
    return np.mean(vecs, axis=0)


def safe_corr(x, y):
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 5 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0, 1.0
    return spearmanr(x, y)


def residualize(x, z):
    coeffs = polyfit(z, x, 1)
    return x - polyval(z, coeffs)


def run_sov_experiment(model, prompts, label):
    n_layers = model.cfg.n_layers
    results = []
    
    for p in prompts:
        ids_a = get_single_token_ids(model, p["A"])
        ids_b = get_single_token_ids(model, p["B"])
        if not ids_a or not ids_b:
            print(f"  Skipping '{p['prompt']}': no valid tokens")
            continue
        
        # Decision axis from unembedding space
        E_A = get_mean_embedding(model, ids_a)
        E_B = get_mean_embedding(model, ids_b)
        D = E_A - E_B
        D_norm = np.linalg.norm(D)
        if D_norm < 1e-10:
            continue
        D_hat = D / D_norm
        
        # Run model
        tokens = model.to_tokens(p["prompt"], prepend_bos=True)
        _, cache = model.run_with_cache(tokens)
        last_pos = tokens.shape[1] - 1
        
        # Extract residual states
        states = []
        for layer in range(n_layers):
            s = cache[f"blocks.{layer}.hook_resid_post"][0, last_pos, :].cpu().numpy()
            states.append(s)
        states = np.array(states)
        
        # Inter-layer deltas
        deltas = np.diff(states, axis=0)  # (n_layers-1, d_model)
        
        # === SOV METRICS ===
        v_parallel = []  # decision velocity (projection onto D)
        sov = []         # semantic orthogonal velocity
        angular_curv = [] # angular curvature between consecutive deltas
        
        for i in range(len(deltas)):
            ds = deltas[i]
            
            # Projection onto decision axis
            vp = np.dot(ds, D_hat)
            v_parallel.append(vp)
            
            # Orthogonal component
            ds_orth = ds - vp * D_hat
            sov.append(np.linalg.norm(ds_orth))
        
        # Angular curvature: angle between consecutive delta vectors
        for i in range(len(deltas) - 1):
            d1 = deltas[i]
            d2 = deltas[i + 1]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 < 1e-10 or n2 < 1e-10:
                angular_curv.append(0.0)
            else:
                cos_angle = np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0)
                angular_curv.append(np.arccos(cos_angle))
        
        v_parallel = np.array(v_parallel)  # (n_layers-1,)
        sov = np.array(sov)                # (n_layers-1,)
        angular_curv = np.array(angular_curv)  # (n_layers-2,)
        
        # Decision velocity rate (how fast v_parallel changes)
        dv_parallel = np.abs(np.diff(v_parallel))  # (n_layers-2,)
        
        # Decision signal from logit lens
        W_U = model.W_U.cpu().numpy()
        delta_logits = []
        for layer in range(n_layers):
            s = states[layer]
            s_norm = (s - s.mean()) / (s.std() + 1e-8)
            logits = s_norm @ W_U
            max_a = max(logits[tid] for tid in ids_a)
            max_b = max(logits[tid] for tid in ids_b)
            delta_logits.append(abs(max_a - max_b))
        delta_logits = np.array(delta_logits)
        decision_rate = np.abs(delta_logits[2:] - delta_logits[:-2]) / 2.0  # (n_layers-2,)
        
        # Align all to n_layers-2 length (interior layers)
        # v_parallel: trim first and last -> [1:-1] but it's n_layers-1 long
        # angular_curv: already n_layers-2
        # dv_parallel: already n_layers-2
        # decision_rate: already n_layers-2
        # sov: trim to match -> [1:] gives n_layers-2... no, [:-1] or [1:]
        # Let's align: angular_curv[i] corresponds to transition i→i+1→i+2
        # decision_rate[i] corresponds to layer i+1 (centered diff)
        # dv_parallel[i] corresponds to transition i→i+1
        # sov: take interior [1:-1] to get n_layers-2 if we want layer alignment
        # Actually: v_parallel has n_layers-1 entries (transitions 0→1, 1→2, ..., 10→11)
        # dv_parallel = |diff(v_parallel)| has n_layers-2 entries
        # angular_curv has n_layers-2 entries  
        # decision_rate has n_layers-2 entries (layers 1..10)
        # sov has n_layers-1 entries; take [1:] to align with layers 1..10? No, [:-1] gives 0..9
        # Let's just use dv_parallel and angular_curv aligned with decision_rate
        
        results.append({
            "prompt": p["prompt"],
            "label": label,
            "v_parallel": v_parallel.tolist(),
            "sov": sov.tolist(),
            "angular_curv": angular_curv.tolist(),
            "dv_parallel": dv_parallel.tolist(),
            "decision_rate": decision_rate.tolist(),
            "delta_logits": delta_logits.tolist(),
        })
        
        # Quick per-prompt summary
        peak_dvp = np.argmax(dv_parallel)
        peak_ac = np.argmax(angular_curv)
        print(f"  [{label}] '{p['prompt']}' — peak dv_parallel at layer {peak_dvp+1}, peak angular_curv at layer {peak_ac+1}")
    
    return results


def main():
    from transformer_lens import HookedTransformer
    
    print(f"Loading GPT-2 small on {DEVICE}...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    model.eval()
    
    print(f"\n--- Ambiguous prompts ({len(AMBIGUOUS)}) ---")
    amb = run_sov_experiment(model, AMBIGUOUS, "ambiguous")
    
    print(f"\n--- Control prompts ({len(CONTROLS)}) ---")
    ctl = run_sov_experiment(model, CONTROLS, "control")
    
    print(f"\nValid: {len(amb)} ambiguous, {len(ctl)} control")
    
    n_interior = len(amb[0]["angular_curv"])  # should be 10
    layers = np.arange(1, n_interior + 1)
    
    # === POOLED CORRELATIONS ===
    # Align: dv_parallel, angular_curv, decision_rate all have n_interior elements
    
    amb_dvp = np.concatenate([r["dv_parallel"] for r in amb])
    amb_ac = np.concatenate([r["angular_curv"] for r in amb])
    amb_dr = np.concatenate([r["decision_rate"] for r in amb])
    amb_layers = np.tile(layers, len(amb))
    
    ctl_dvp = np.concatenate([r["dv_parallel"] for r in ctl])
    ctl_ac = np.concatenate([r["angular_curv"] for r in ctl])
    ctl_dr = np.concatenate([r["decision_rate"] for r in ctl])
    ctl_layers = np.tile(layers, len(ctl))
    
    print("\n" + "=" * 70)
    print("SOV EXPERIMENT RESULTS")
    print("=" * 70)
    
    # Raw correlations
    r1, p1 = safe_corr(amb_dvp, amb_dr)
    r2, p2 = safe_corr(amb_ac, amb_dr)
    print(f"\n=== RAW CORRELATIONS (ambiguous, pooled) ===")
    print(f"  |dv_parallel| ↔ decision_rate:  ρ = {r1:.4f}, p = {p1:.4g}")
    print(f"  angular_curv  ↔ decision_rate:  ρ = {r2:.4f}, p = {p2:.4g}")
    
    # Correlation with layer
    rl1, _ = safe_corr(amb_dvp, amb_layers)
    rl2, _ = safe_corr(amb_ac, amb_layers)
    rl3, _ = safe_corr(amb_dr, amb_layers)
    print(f"\n=== CORRELATION WITH LAYER (architectural check) ===")
    print(f"  |dv_parallel| ↔ layer:  ρ = {rl1:.4f}")
    print(f"  angular_curv  ↔ layer:  ρ = {rl2:.4f}")
    print(f"  decision_rate ↔ layer:  ρ = {rl3:.4f}")
    
    # Partial correlations (layer removed)
    dvp_res = residualize(amb_dvp, amb_layers)
    ac_res = residualize(amb_ac, amb_layers)
    dr_res = residualize(amb_dr, amb_layers)
    
    rp1, pp1 = safe_corr(dvp_res, dr_res)
    rp2, pp2 = safe_corr(ac_res, dr_res)
    print(f"\n=== PARTIAL CORRELATIONS (layer effect removed) ===")
    print(f"  |dv_parallel| ↔ decision_rate:  ρ = {rp1:.4f}, p = {pp1:.4g}")
    print(f"  angular_curv  ↔ decision_rate:  ρ = {rp2:.4f}, p = {pp2:.4g}")
    
    # Per-layer analysis
    print(f"\n=== PER-LAYER CORRELATIONS ===")
    for layer in range(n_interior):
        l_idx = layer  # 0-indexed within interior
        actual_layer = layer + 1
        mask_a = amb_layers == actual_layer
        
        dvp_l = amb_dvp[mask_a]
        ac_l = amb_ac[mask_a]
        dr_l = amb_dr[mask_a]
        
        r_dvp, p_dvp = safe_corr(dvp_l, dr_l)
        r_ac, p_ac = safe_corr(ac_l, dr_l)
        
        sig_dvp = "*" if p_dvp < 0.05 else " "
        sig_ac = "*" if p_ac < 0.05 else " "
        print(f"  Layer {actual_layer:2d}: dvp↔dr ρ={r_dvp:+.3f} (p={p_dvp:.3g}){sig_dvp} | ac↔dr ρ={r_ac:+.3f} (p={p_ac:.3g}){sig_ac}")
    
    # Per-prompt correlations
    prompt_corrs_dvp = []
    prompt_corrs_ac = []
    for r in amb:
        dvp = np.array(r["dv_parallel"])
        ac = np.array(r["angular_curv"])
        dr = np.array(r["decision_rate"])
        rc1, _ = safe_corr(dvp, dr)
        rc2, _ = safe_corr(ac, dr)
        prompt_corrs_dvp.append(rc1)
        prompt_corrs_ac.append(rc2)
    
    print(f"\n=== PER-PROMPT CORRELATIONS (across layers within each prompt) ===")
    print(f"  dvp↔dr:  mean ρ = {np.mean(prompt_corrs_dvp):.4f} ± {np.std(prompt_corrs_dvp):.4f}")
    print(f"  ac↔dr:   mean ρ = {np.mean(prompt_corrs_ac):.4f} ± {np.std(prompt_corrs_ac):.4f}")
    
    # === AMBIGUOUS vs CONTROL comparison ===
    # Compare v_parallel profiles - do ambiguous prompts show more sign changes?
    amb_sign_changes = []
    for r in amb:
        vp = np.array(r["v_parallel"])
        signs = np.sign(vp)
        changes = np.sum(np.abs(np.diff(signs)) > 0)
        amb_sign_changes.append(changes)
    
    ctl_sign_changes = []
    for r in ctl:
        vp = np.array(r["v_parallel"])
        signs = np.sign(vp)
        changes = np.sum(np.abs(np.diff(signs)) > 0)
        ctl_sign_changes.append(changes)
    
    print(f"\n=== SIGN CHANGES IN v_parallel (decision oscillation) ===")
    print(f"  Ambiguous: mean = {np.mean(amb_sign_changes):.2f} ± {np.std(amb_sign_changes):.2f}")
    print(f"  Control:   mean = {np.mean(ctl_sign_changes):.2f} ± {np.std(ctl_sign_changes):.2f}")
    if len(amb_sign_changes) > 2 and len(ctl_sign_changes) > 2:
        u, p = mannwhitneyu(amb_sign_changes, ctl_sign_changes, alternative="greater")
        print(f"  Mann-Whitney (amb > ctl): U={u:.1f}, p={p:.4g}")
    
    # === VERDICT ===
    print(f"\n=== VERDICT ===")
    if abs(rp1) > 0.3 and pp1 < 0.05:
        v1 = "SUPPORTED"
    elif abs(rp1) < 0.1 or pp1 > 0.1:
        v1 = "NOT SUPPORTED"
    else:
        v1 = "MIXED"
    
    if abs(rp2) > 0.3 and pp2 < 0.05:
        v2 = "SUPPORTED"
    elif abs(rp2) < 0.1 or pp2 > 0.1:
        v2 = "NOT SUPPORTED"
    else:
        v2 = "MIXED"
    
    print(f"  Decision velocity ↔ decision rate (layer-corrected): {v1}")
    print(f"  Angular curvature ↔ decision rate (layer-corrected): {v2}")
    print("=" * 70)
    
    # === PLOTS ===
    
    # 1. v_parallel profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    transitions = range(len(amb[0]["v_parallel"]))
    for r in amb:
        ax1.plot(transitions, r["v_parallel"], alpha=0.3, color="steelblue")
    mean_vp = np.mean([r["v_parallel"] for r in amb], axis=0)
    ax1.plot(transitions, mean_vp, color="darkblue", lw=2, label="Mean")
    ax1.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax1.set_title("Decision Velocity v_∥ — Ambiguous")
    ax1.set_xlabel("Layer transition")
    ax1.set_ylabel("Projection onto decision axis")
    ax1.legend()
    
    for r in ctl:
        ax2.plot(range(len(r["v_parallel"])), r["v_parallel"], alpha=0.3, color="coral")
    mean_vp_c = np.mean([r["v_parallel"] for r in ctl], axis=0)
    ax2.plot(range(len(mean_vp_c)), mean_vp_c, color="darkred", lw=2, label="Mean")
    ax2.axhline(y=0, color="black", linestyle="--", alpha=0.3)
    ax2.set_title("Decision Velocity v_∥ — Control")
    ax2.set_xlabel("Layer transition")
    ax2.set_ylabel("Projection onto decision axis")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "v_parallel_profiles.png", dpi=150)
    plt.close()
    
    # 2. Angular curvature profiles
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for r in amb:
        ax1.plot(range(1, len(r["angular_curv"])+1), r["angular_curv"], alpha=0.3, color="steelblue")
    mean_ac = np.mean([r["angular_curv"] for r in amb], axis=0)
    ax1.plot(range(1, len(mean_ac)+1), mean_ac, color="darkblue", lw=2, label="Mean")
    ax1.set_title("Angular Curvature — Ambiguous")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Angle (radians)")
    ax1.legend()
    
    for r in ctl:
        ax2.plot(range(1, len(r["angular_curv"])+1), r["angular_curv"], alpha=0.3, color="coral")
    mean_ac_c = np.mean([r["angular_curv"] for r in ctl], axis=0)
    ax2.plot(range(1, len(mean_ac_c)+1), mean_ac_c, color="darkred", lw=2, label="Mean")
    ax2.set_title("Angular Curvature — Control")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Angle (radians)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(OUTDIR / "angular_curvature_profiles.png", dpi=150)
    plt.close()
    
    # 3. Scatter: dvp vs decision_rate (with and without layer correction)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.scatter(amb_dvp, amb_dr, alpha=0.3, s=15, color="steelblue")
    ax1.set_xlabel("|dv_parallel|")
    ax1.set_ylabel("Decision rate")
    ax1.set_title(f"Raw: ρ={r1:.3f} (p={p1:.3g})")
    
    ax2.scatter(dvp_res, dr_res, alpha=0.3, s=15, color="steelblue")
    ax2.set_xlabel("|dv_parallel| (layer-corrected)")
    ax2.set_ylabel("Decision rate (layer-corrected)")
    ax2.set_title(f"Layer-corrected: ρ={rp1:.3f} (p={pp1:.3g})")
    plt.tight_layout()
    plt.savefig(OUTDIR / "dvp_vs_decision_rate.png", dpi=150)
    plt.close()
    
    # Save summary
    summary = {
        "raw_correlations": {
            "dvp_vs_dr": {"rho": r1, "p": p1},
            "ac_vs_dr": {"rho": r2, "p": p2},
        },
        "layer_correlations": {
            "dvp_vs_layer": rl1, "ac_vs_layer": rl2, "dr_vs_layer": rl3
        },
        "partial_correlations": {
            "dvp_vs_dr": {"rho": rp1, "p": pp1, "verdict": v1},
            "ac_vs_dr": {"rho": rp2, "p": pp2, "verdict": v2},
        },
        "per_prompt": {
            "dvp_dr_mean_rho": float(np.mean(prompt_corrs_dvp)),
            "dvp_dr_std_rho": float(np.std(prompt_corrs_dvp)),
            "ac_dr_mean_rho": float(np.mean(prompt_corrs_ac)),
            "ac_dr_std_rho": float(np.std(prompt_corrs_ac)),
        },
        "sign_changes": {
            "ambiguous_mean": float(np.mean(amb_sign_changes)),
            "control_mean": float(np.mean(ctl_sign_changes)),
        },
    }
    
    with open(OUTDIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPlots and data saved to {OUTDIR.resolve()}")


if __name__ == "__main__":
    main()
