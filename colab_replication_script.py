# ============================================================
# Δ-Field Replication on Larger Model
# Ready for Google Colab (GPU runtime)
# ============================================================
#
# INSTRUCTIONS FOR GEMINI:
# 1. Open Google Colab, select GPU runtime (T4)
# 2. Paste this entire script into a cell
# 3. Run it
# 4. Send back: all printed output + saved plots
#
# Model options (uncomment one):
#   "pythia-1.4b"     — 24 layers, fits T4 easily
#   "pythia-2.8b"     — 32 layers, tight on T4 but doable
#   "gpt2-medium"     — 24 layers, safe fallback
#
# The script runs all three experiments:
#   Exp 1: L2 velocity/curvature vs decision rate
#   Exp 2: Semantic Orthogonal Velocity (SOV)
#   Exp 3: Cross-prompt variance of v_parallel (superposition test)
#
# Total runtime estimate: 5-15 minutes on T4
# ============================================================

!pip install transformer_lens torch numpy scipy matplotlib -q

import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr, mannwhitneyu
from numpy.polynomial.polynomial import polyfit, polyval

torch.set_grad_enabled(False)

# ============================================================
# CONFIG — CHANGE MODEL HERE
# ============================================================
MODEL_NAME = "pythia-1.4b"  # Options: "pythia-1.4b", "pythia-2.8b", "gpt2-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTDIR = Path("delta_field_replication")
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_NAME}")

# ============================================================
# PROMPTS
# ============================================================
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

# ============================================================
# UTILITIES
# ============================================================

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

# ============================================================
# MAIN EXTRACTION
# ============================================================

def extract_all(model, prompts, label):
    """Extract all metrics for a set of prompts."""
    n_layers = model.cfg.n_layers
    results = []

    for p in prompts:
        ids_a = get_single_token_ids(model, p["A"])
        ids_b = get_single_token_ids(model, p["B"])
        if not ids_a or not ids_b:
            print(f"  Skipping '{p['prompt']}': no valid tokens")
            continue

        # Decision axis
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

        # Residual states
        states = []
        for layer in range(n_layers):
            s = cache[f"blocks.{layer}.hook_resid_post"][0, last_pos, :].cpu().numpy()
            states.append(s)
        states = np.array(states)

        # Deltas
        deltas = np.diff(states, axis=0)

        # L2 velocity and curvature
        velocity = np.linalg.norm(deltas, axis=1)
        curvature = np.abs(np.diff(velocity))

        # SOV: projection onto decision axis
        v_parallel = np.array([np.dot(ds, D_hat) for ds in deltas])
        sov = np.array([np.linalg.norm(ds - np.dot(ds, D_hat) * D_hat) for ds in deltas])

        # Angular curvature
        angular_curv = []
        for i in range(len(deltas) - 1):
            d1, d2 = deltas[i], deltas[i+1]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            if n1 < 1e-10 or n2 < 1e-10:
                angular_curv.append(0.0)
            else:
                angular_curv.append(np.arccos(np.clip(np.dot(d1, d2) / (n1 * n2), -1, 1)))
        angular_curv = np.array(angular_curv)

        # dv_parallel
        dv_parallel = np.abs(np.diff(v_parallel))

        # Decision signal (logit lens)
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
        decision_rate = np.abs(delta_logits[2:] - delta_logits[:-2]) / 2.0

        # Attention divergence
        attn_divs = []
        for layer in range(n_layers):
            attn = cache[f"blocks.{layer}.attn.hook_pattern"][0, :, last_pos, :].cpu().numpy()
            n_heads = attn.shape[0]
            kl_sum, count = 0.0, 0
            for h1 in range(n_heads):
                for h2 in range(h1+1, n_heads):
                    p1 = attn[h1] + 1e-10; p1 /= p1.sum()
                    p2 = attn[h2] + 1e-10; p2 /= p2.sum()
                    kl_sum += (np.sum(p1*np.log(p1/p2)) + np.sum(p2*np.log(p2/p1))) / 2
                    count += 1
            attn_divs.append(kl_sum/count if count else 0)
        attn_divs = np.array(attn_divs)

        results.append({
            "prompt": p["prompt"], "label": label,
            "velocity": velocity.tolist(),
            "curvature": curvature.tolist(),
            "v_parallel": v_parallel.tolist(),
            "sov": sov.tolist(),
            "angular_curv": angular_curv.tolist(),
            "dv_parallel": dv_parallel.tolist(),
            "decision_rate": decision_rate.tolist(),
            "delta_logits": delta_logits.tolist(),
            "attn_divergence": attn_divs[1:-1].tolist(),
        })
        print(f"  [{label}] '{p['prompt']}' — peak v at L{np.argmax(velocity)}, peak v_∥ change at L{np.argmax(dv_parallel)+1}")

    return results

# ============================================================
# ANALYSIS
# ============================================================

def run_analysis(amb, ctl, n_layers):
    n_interior = n_layers - 2
    layers = np.arange(1, n_interior + 1)

    print("\n" + "=" * 70)
    print(f"RESULTS FOR {MODEL_NAME} ({n_layers} layers)")
    print("=" * 70)

    # --- EXP 1: L2 curvature vs decision rate ---
    amb_kappa = np.concatenate([r["curvature"] for r in amb])
    amb_dr = np.concatenate([r["decision_rate"] for r in amb])
    amb_layers_k = np.tile(np.arange(1, len(amb[0]["curvature"])+1), len(amb))

    r1, p1 = safe_corr(amb_kappa, amb_dr)
    k_res = residualize(amb_kappa, amb_layers_k)
    dr_res_k = residualize(amb_dr, amb_layers_k)
    r1p, p1p = safe_corr(k_res, dr_res_k)

    print(f"\n--- EXP 1: L2 Curvature κ vs Decision Rate ---")
    print(f"  Raw:            ρ = {r1:.4f}, p = {p1:.4g}")
    print(f"  Layer-corrected: ρ = {r1p:.4f}, p = {p1p:.4g}")

    # --- EXP 2: SOV vs decision rate ---
    amb_dvp = np.concatenate([r["dv_parallel"] for r in amb])
    amb_ac = np.concatenate([r["angular_curv"] for r in amb])
    amb_dr2 = np.concatenate([r["decision_rate"] for r in amb])
    amb_layers2 = np.tile(layers, len(amb))

    r2, p2 = safe_corr(amb_dvp, amb_dr2)
    r3, p3 = safe_corr(amb_ac, amb_dr2)

    dvp_res = residualize(amb_dvp, amb_layers2)
    ac_res = residualize(amb_ac, amb_layers2)
    dr_res2 = residualize(amb_dr2, amb_layers2)

    r2p, p2p = safe_corr(dvp_res, dr_res2)
    r3p, p3p = safe_corr(ac_res, dr_res2)

    print(f"\n--- EXP 2: SOV Metrics vs Decision Rate ---")
    print(f"  |dv_∥| raw:            ρ = {r2:.4f}, p = {p2:.4g}")
    print(f"  |dv_∥| layer-corrected: ρ = {r2p:.4f}, p = {p2p:.4g}")
    print(f"  angular_curv raw:            ρ = {r3:.4f}, p = {p3:.4g}")
    print(f"  angular_curv layer-corrected: ρ = {r3p:.4f}, p = {p3p:.4g}")

    # --- EXP 3: Variance of v_parallel ---
    n_transitions = len(amb[0]["v_parallel"])
    amb_vp_matrix = np.array([r["v_parallel"] for r in amb])
    ctl_vp_matrix = np.array([r["v_parallel"] for r in ctl])

    print(f"\n--- EXP 3: Cross-prompt Variance of v_∥ (Superposition Test) ---")
    print(f"{'Trans':>8} {'Var(amb)':>12} {'Var(ctl)':>12} {'Ratio':>8} {'boot_p':>10}")
    print("-" * 55)

    amb_vars = []
    ctl_vars = []
    boot_ps = []

    for t in range(n_transitions):
        v_a = amb_vp_matrix[:, t]
        v_c = ctl_vp_matrix[:, t]
        va, vc = np.var(v_a), np.var(v_c)
        amb_vars.append(va)
        ctl_vars.append(vc)
        ratio = va / vc if vc > 1e-10 else float('inf')

        # Bootstrap
        observed = va - vc
        pooled = np.concatenate([v_a, v_c])
        n_a = len(v_a)
        count = sum(1 for _ in range(3000)
                     if np.var(np.random.permutation(pooled)[:n_a]) -
                        np.var(np.random.permutation(pooled)[n_a:]) >= observed)
        bp = count / 3000
        boot_ps.append(bp)

        sig = "***" if bp < 0.001 else ("**" if bp < 0.01 else ("*" if bp < 0.05 else ""))
        print(f"  {t:>3}→{t+1:<3} {va:>12.4f} {vc:>12.4f} {ratio:>8.2f} {bp:>9.4f} {sig}")

    amb_vars = np.array(amb_vars)
    ctl_vars = np.array(ctl_vars)

    # Summary by region
    third = n_transitions // 3
    early = slice(0, third)
    mid = slice(third, 2*third)
    late = slice(2*third, n_transitions)

    print(f"\n  Region summary:")
    print(f"    Early (0-{third-1}):     ratio = {np.mean(amb_vars[early])/(np.mean(ctl_vars[early])+1e-10):.2f}")
    print(f"    Mid   ({third}-{2*third-1}):    ratio = {np.mean(amb_vars[mid])/(np.mean(ctl_vars[mid])+1e-10):.2f}")
    print(f"    Late  ({2*third}-{n_transitions-1}):   ratio = {np.mean(amb_vars[late])/(np.mean(ctl_vars[late])+1e-10):.2f}")

    # Sign changes
    amb_sc = [np.sum(np.abs(np.diff(np.sign(r["v_parallel"]))) > 0) for r in amb]
    ctl_sc = [np.sum(np.abs(np.diff(np.sign(r["v_parallel"]))) > 0) for r in ctl]
    print(f"\n  Sign changes in v_∥:")
    print(f"    Ambiguous: {np.mean(amb_sc):.1f} ± {np.std(amb_sc):.1f}")
    print(f"    Control:   {np.mean(ctl_sc):.1f} ± {np.std(ctl_sc):.1f}")

    # Velocity peak distribution
    amb_vpeak = [np.argmax(r["velocity"]) for r in amb]
    ctl_vpeak = [np.argmax(r["velocity"]) for r in ctl]
    print(f"\n  Velocity peak layers:")
    print(f"    Ambiguous: {dict(zip(*np.unique(amb_vpeak, return_counts=True)))}")
    print(f"    Control:   {dict(zip(*np.unique(ctl_vpeak, return_counts=True)))}")

    # --- PLOTS ---

    # Plot 1: v_parallel trajectories
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for r in amb:
        ax1.plot(r["v_parallel"], alpha=0.25, color="steelblue")
    ax1.plot(np.mean(amb_vp_matrix, axis=0), color="darkblue", lw=2.5, label="Mean")
    ax1.axhline(0, color="black", ls="--", alpha=0.3)
    ax1.set_title(f"Decision Velocity v_∥ — Ambiguous\n({MODEL_NAME})")
    ax1.set_xlabel("Layer transition"); ax1.set_ylabel("v_∥"); ax1.legend()

    for r in ctl:
        ax2.plot(r["v_parallel"], alpha=0.35, color="coral")
    ax2.plot(np.mean(ctl_vp_matrix, axis=0), color="darkred", lw=2.5, label="Mean")
    ax2.axhline(0, color="black", ls="--", alpha=0.3)
    ax2.set_title(f"Decision Velocity v_∥ — Control\n({MODEL_NAME})")
    ax2.set_xlabel("Layer transition"); ax2.set_ylabel("v_∥"); ax2.legend()
    plt.tight_layout(); plt.savefig(OUTDIR / "v_parallel.png", dpi=150); plt.close()

    # Plot 2: Variance comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    ax1.plot(amb_vars, 'o-', color="steelblue", label="Ambiguous", lw=2)
    ax1.plot(ctl_vars, 'o-', color="coral", label="Control", lw=2)
    ax1.set_xlabel("Layer transition"); ax1.set_ylabel("Var(v_∥)")
    ax1.set_title(f"Cross-prompt Variance\n({MODEL_NAME})"); ax1.legend()

    ratios = amb_vars / (ctl_vars + 1e-10)
    colors = ["steelblue" if r > 1 else "coral" for r in ratios]
    ax2.bar(range(len(ratios)), ratios, color=colors)
    ax2.axhline(1, color="black", ls="--", alpha=0.5)
    ax2.set_xlabel("Layer transition"); ax2.set_ylabel("Var(amb)/Var(ctl)")
    ax2.set_title("Variance Ratio")
    plt.tight_layout(); plt.savefig(OUTDIR / "variance.png", dpi=150); plt.close()

    # Plot 3: Angular curvature
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    for r in amb:
        ax1.plot(r["angular_curv"], alpha=0.25, color="steelblue")
    ax1.plot(np.mean([r["angular_curv"] for r in amb], axis=0), color="darkblue", lw=2.5, label="Mean")
    ax1.set_title(f"Angular Curvature — Ambiguous\n({MODEL_NAME})")
    ax1.set_xlabel("Layer"); ax1.set_ylabel("Angle (rad)"); ax1.legend()

    for r in ctl:
        ax2.plot(r["angular_curv"], alpha=0.35, color="coral")
    ax2.plot(np.mean([r["angular_curv"] for r in ctl], axis=0), color="darkred", lw=2.5, label="Mean")
    ax2.set_title(f"Angular Curvature — Control\n({MODEL_NAME})")
    ax2.set_xlabel("Layer"); ax2.set_ylabel("Angle (rad)"); ax2.legend()
    plt.tight_layout(); plt.savefig(OUTDIR / "angular_curv.png", dpi=150); plt.close()

    print(f"\nPlots saved to {OUTDIR.resolve()}")
    print("=" * 70)

# ============================================================
# RUN
# ============================================================

from transformer_lens import HookedTransformer

print(f"\nLoading {MODEL_NAME}...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
model.eval()
n_layers = model.cfg.n_layers
print(f"Loaded: {n_layers} layers, d_model={model.cfg.d_model}")

print(f"\n--- Ambiguous ({len(AMBIGUOUS)} prompts) ---")
amb = extract_all(model, AMBIGUOUS, "ambiguous")

print(f"\n--- Control ({len(CONTROLS)} prompts) ---")
ctl = extract_all(model, CONTROLS, "control")

run_analysis(amb, ctl, n_layers)
