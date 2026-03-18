# Semantic Superposition in the Residual Stream

**Paper:** Residual Stream Geometry Reveals Semantic Superposition: Evidence from Decision-Axis Trajectory Analysis  
**Author:** Yanush Feshter (ORCID: 0009-0002-1330-7530)  
**DOI:** [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX) *(update after upload)*

## Overview

This repository contains code to reproduce findings from our paper on semantic superposition in transformer residual streams.

We test three differential metrics for mechanistic interpretability on GPT-2 small (12 layers) and Pythia-1.4b (24 layers). Two metrics fail honestly (L2 velocity = architectural artifact; logit lens = unreliable ground truth in middle layers). The third — **cross-prompt variance of decision-axis velocity** — reveals a robust signal:

- Ambiguous prompts produce **2–3× higher variance** on the semantic decision axis than controls
- The effect is **direction-specific**: semantic axis achieves ratio 1.81, exceeding the 99th percentile of 100 random directions (0.92)
- **Empirical p < 0.01** — no random direction reproduces the effect
- **Replicated** across GPT-2 small and Pythia-1.4b

We interpret this as geometric evidence that transformers maintain competing interpretations simultaneously (semantic superposition) rather than deciding at a discrete layer.

## Key Figure

![Direction Specificity Control](prepub_tests/prepub_tests.png)

*Left: Variance ratio by layer (no individual layer survives FDR correction). Right: The semantic decision axis (red, ratio 1.81) far exceeds 100 random directions (blue histogram). Zero random directions reproduce the effect.*

## Repository Structure

```
experiment.py              # Exp 1: L2 velocity/curvature vs decision rate
sov_experiment.py          # Exp 2: Semantic Orthogonal Velocity (SOV)
variance_experiment.py     # Exp 3: Cross-prompt variance (superposition test)
prepub_tests.py            # FDR correction + random direction control
colab_replication_script.py  # One-file replication for Google Colab (GPU)
paper/                     # LaTeX source + compiled PDF
```

## Quick Start

### Local (CPU, ~5 min)
```bash
pip install transformer_lens torch numpy scipy matplotlib
python prepub_tests.py
```

### Google Colab (GPU, for larger models)
1. Open Colab, select T4 GPU runtime
2. Upload `colab_replication_script.py`
3. Run — outputs plots + statistics for Pythia-1.4b

## Requirements

- Python 3.10+
- PyTorch
- TransformerLens
- NumPy, SciPy, Matplotlib

## Method

For each ambiguous prompt (e.g., "A bat is a"), we define a **semantic decision axis**:

```
D = mean_unembedding(interpretation_A) - mean_unembedding(interpretation_B)
```

We project inter-layer residual stream changes onto this axis and measure **cross-prompt variance** at each layer. High variance = the model is pulled in different directions by different prompts = superposition. Low variance = convergence.

The critical control: we repeat this for 100 random directions in the embedding space. Only the semantic axis shows elevated variance for ambiguous prompts.

## Citation

```bibtex
@misc{feshter2026superposition,
  title={Residual Stream Geometry Reveals Semantic Superposition: 
         Evidence from Decision-Axis Trajectory Analysis},
  author={Feshter, Yanush},
  year={2026},
  publisher={Zenodo},
  doi={10.5281/zenodo.XXXXXXX}
}
```

## License

MIT
