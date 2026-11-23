# C-TA-HLP: Capacity-Constrained Traffic-Adaptive Hybrid Link Protection for Carrier-Grade IP Networks

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Tests: 43 passed](https://img.shields.io/badge/tests-43%20passed-brightgreen.svg)](https://github.com/ihkokil/hybrid-link-protection)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Target Venue: IEEE TNSM](https://img.shields.io/badge/Target-IEEE%20TNSM-orange.svg)](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=4275028)

This repository contains the complete reference implementation, exact Mixed-Integer Linear Programming (MIP) formulation, evaluation framework, and manuscript assets for **C-TA-HLP** (*Capacity-Constrained Traffic-Adaptive Hybrid Link Protection*), submitted to *IEEE Transactions on Network and Service Management (IEEE TNSM)*.

---

## 1. Overview and Problem Statement

Carrier-grade IP backbones require sub-50 ms packet forwarding upon link failures while keeping router forwarding table state bounded. **Hybrid Link Protection (HLP)** ([Geng *et al.*, JCN 2020](https://doi.org/10.1109/JCN.2019.000056)) combines local **Loop-Free Alternates (LFA)** with selective multi-hop **Backup Path Protection (BPP)** tunnels. However, standard HLP suffers from two critical limitations:

1. **Traffic Blindness**: It ranks candidate key links purely by unweighted topological reachability ($|R_{\text{cov}}(e)|$), treating heavily loaded core corridors and lightly loaded edge stubs identically.
2. **Congestion Blindness**: It ignores post-failure link capacity constraints, steering failover traffic into bottleneck links and inducing secondary downstream congestion drops.

### The C-TA-HLP Solution

**C-TA-HLP** resolves both limitations by introducing:
* **Traffic-Adaptive Gain ($\Delta_T(e)$)**: Proactively weights each protected link by its aggregate rescued traffic volume and repair path reliability:
  $$\Delta_T(e) = \sum_{(s,d) \in e.P} T(s, d) \cdot p_e \cdot A(P_{\text{repair}}(e))$$
* **Continuous Congestion Penalty ($\Phi(e)$)**: Dynamically derates candidate gains when shifted detour flows exceed link headroom:
  $$\Phi(e) = \prod_{k \in P_{\text{repair}}(e)} \min\left(1.0, \; \frac{C_k}{L_e(k)}\right)$$
* **Exact Theoretical Benchmark**: Formulates the budget- and capacity-constrained problem as a 0-1 Multidimensional Knapsack Problem (0-1 MKP) solved to global optimality using the SciPy HiGHS branch-and-cut solver.

---

## 2. Key Theoretical Contributions

* **Proposition 1 (Modularity under Strict Single-Link Failures)**: Under independent single-link failures with precomputed repair paths and non-binding capacities, the rescued traffic objective is strictly modular (additive). Consequently, greedy sorting by marginal traffic coefficients is **globally optimal**, matching the exact 0-1 MIP.
* **Theorem 2 (Monotone Submodularity under SRLG Failures)**: When links share physical conduits (Shared Risk Link Groups), overlapping rescued flows introduce diminishing returns. The unconstrained objective becomes a monotone submodular set function under a cardinality constraint, provably guaranteeing a $(1 - 1/e) \approx 63.2\%$ approximation bound for greedy selection.
* **NP-Hardness under Bounded Capacities**: When link bandwidth constraints are active, detour flows from multiple protected links interact on shared links, transforming the problem into an NP-hard 0-1 Multidimensional Knapsack Problem.

---

## 3. Empirical Results Summary

Evaluated across four real-world carrier topologies from **SNDlib** (Abilene, GÉANT, Germany50, Nobel-EU) with empirical NetFlow demand matrices across budgets $B \in \{2, 4, 6, 8\}$:

| Metric | C-TA-HLP Performance | Comparison against Baseline HLP (2020) |
|---|---|---|
| **MIP Optimality Ratio** | **100.0%** across all 16 configurations | Matches the exact branch-and-cut theoretical optimum |
| **Availability (Germany50)** | **99.08% -- 99.11%** | $+0.048$ to $+0.063$ pp gain ($p = 0.0012$, Cohen's $d = 2.83$) |
| **Availability (Nobel-EU)** | **98.57% -- 98.86%** | Up to $+0.1308$ pp gain ($p = 0.0092$, Cohen's $d = 3.33$) |
| **Peak Link Utilization (MLU)** | Reduced to **1.88** (Abilene $B=6$) | **17.3% reduction** in peak link overload vs. HLP (2.24) |
| **Congestion Drop Mitigation** | Reduced to **5.0** (Nobel-EU $B=8$) | **5.3% reduction** in post-failure flow drops vs. HLP (5.3) |
| **Traffic Uncertainty Robustness** | Statistically significant gains ($p_{\text{Bonf}} < 0.001$) | Validated across $n = 19$ perturbed demand matrices ($\sigma = 20\%$) |
| **ISP-Scale Precomputation** | Selection time **$< 0.1$ s** ($|V|=1000, |E|=1996$) | Fully scalable to carrier network management planes |

---

## 4. Repository Structure

```
hybrid-link-protection/
├── data/
│   ├── gml/                      # Network topologies (Internet Topology Zoo & synthetic ISP graphs)
│   ├── rocketfuel/               # CAIDA Rocketfuel trace directory (.gitkeep)
│   └── sndlib/                   # SNDlib carrier graphs (Abilene, GÉANT, Germany50, Nobel-EU)
├── experiments/
│   ├── run_comprehensive_campaign.py  # End-to-end benchmark campaign runner
│   ├── run_evaluator_campaign.py      # Unified flow-level simulation campaign runner
│   ├── run_statistical_experiments.py # Multi-replicate hypothesis testing runner
│   ├── run_budget_sweep.py            # Budget sweep diagnostic runner
│   ├── run_compute_time.py            # Computational complexity benchmarking
│   ├── run_demo.py                    # Quick interactive demonstration
│   └── run_table_reproduction.py      # Baseline reproduction diagnostics
├── plots/
│   └── sndlib/                   # Publication availability curves for SNDlib carrier networks
├── plotting/
│   ├── plot_comprehensive.py     # Figure generator for manuscript PDFs
│   ├── plot_results.py           # General curve and distribution plotter
│   └── plot_table_reproduction.py# Table diagnostic plotting
├── results/
│   ├── ablation_slf/             # Strict single-link failure ablation logs
│   ├── ablation_srlg/            # Correlated SRLG ablation logs
│   ├── final_campaign/           # Final evaluation campaign frozen execution logs
│   ├── sndlib/                   # Multi-replicate SNDlib statistical outputs
│   ├── baseline_comparisons.csv  # 6-baseline comparison dataset across budgets B ∈ [2, 8]
│   ├── headroom_sensitivity.csv  # Capacity headroom sweep data (α ∈ [1.1x, 2.0x])
│   └── traffic_uncertainty_stats.csv # Paired statistical test outputs (n=19 perturbed matrices)
├── scripts/
│   ├── build_comprehensive_tables.py  # Generates LaTeX benchmark tables from results/
│   ├── build_results_table.py         # Formats statistical comparison tables
│   └── build_submission_zip.py        # Rebuilds the IEEEtran submission zip package
├── src/
│   ├── evaluator.py              # Unified flow-level Monte Carlo evaluator & MIP solver
│   ├── evaluator_srlg.py         # Submodular SRLG evaluator and fluid capacity model
│   ├── simulator.py              # Discrete-event datapath forwarding simulator
│   ├── ta_bpp.py                 # Core C-TA-HLP algorithm and regularized marginal gain
│   ├── protection.py             # LFC, HLP, and C-TA-HLP composition interfaces
│   ├── lfc.py                    # Loop-Free Criterion (LFC / LFA) evaluation
│   ├── bpp.py                    # Baseline topological Backup Path Protection (Geng 2020)
│   ├── mnp_e.py                  # MNP-e reference implementation
│   ├── shortest_path.py          # Deterministic tie-breaking Dijkstra utilities
│   ├── topology.py               # Graph parsers, synthetic BA generators, and metrics
│   ├── sndlib.py                 # SNDlib XML/JSON loader and demand parser
│   ├── rocketfuel.py             # Rocketfuel loader and scale-matched synthetic fallback
│   ├── metrics.py                # TCAM footprint, stretch, and MLU diagnostics
│   └── utils.py                  # Logging and serialization utilities
├── tests/                        # 43 automated unit and integration tests (pytest)
├── requirements.txt              # Production dependency specifications
├── LICENSE                       # Apache License 2.0
└── README.md                     # Comprehensive project documentation
```

---

## 5. Installation and Setup

### Prerequisites
* Python 3.10, 3.11, 3.12, or 3.13
* Virtual environment recommended

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/ihkokil/hybrid-link-protection.git
cd hybrid-link-protection

# 2. Create and activate a virtual environment
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Install core dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Run Automated Test Suite

Verify complete algorithmic correctness, solver feasibility, and regression tests:

```bash
pytest -v
```

All 43 unit and integration tests should pass (`43 passed`).

---

## 6. Reproducing Paper Experiments

### A. Quick Demo
Run an interactive demonstration of LFC, topological HLP, and C-TA-HLP on Abilene:

```bash
python main.py --topology data/gml/Abilene.gml --budget 6 --trials 300
```

### B. Full 6-Baseline Comparison Campaign
Reproduce the multi-baseline benchmark comparison across all 4 SNDlib carrier networks (Abilene, GÉANT, Germany50, Nobel-EU) across budgets $B \in \{2, 4, 6, 8\}$:

```bash
python experiments/run_comprehensive_campaign.py
```
This executes the 6 evaluated schemes:
1. **LFC Only**: Pure hop-by-hop loop-free alternates (no tunnels).
2. **Random Selection**: Random selection of $B$ key links.
3. **HLP (2020)**: Topological greedy reachability ranking ([Geng *et al.*, 2020](https://doi.org/10.1109/JCN.2019.000056)).
4. **Max-Traffic**: Unconstrained greedy traffic volume ranking.
5. **C-TA-HLP (Proposed)**: Congestion-regularized traffic-adaptive greedy selection.
6. **Optimal-ILP**: Exact 0-1 branch-and-cut optimization via SciPy HiGHS.

### C. Multi-Matrix Traffic Uncertainty Campaign
Evaluate stability under dynamic traffic demand fluctuations using $n=19$ perturbed traffic matrices ($T'(s,d) = T(s,d) \cdot [1 + \mathcal{N}(0, 0.20)]$):

```bash
python experiments/run_evaluator_campaign.py --traffic-uncertainty
```

### D. Capacity Headroom Sensitivity Sweep
Sweep capacity headroom factors $\alpha \in \{1.10\times, 1.25\times, 1.50\times, 2.00\times\}$ to evaluate the transition from bottleneck-dominated to headroom-abundant regimes:

```bash
python experiments/run_evaluator_campaign.py --headroom-sweep
```

### E. Regenerate Publication Figures and Tables

```bash
# Regenerate vector PDF plots
python plotting/plot_comprehensive.py

# Rebuild all summary tables
python scripts/build_comprehensive_tables.py
```

---

## 7. Evaluator and Simulation Methodology

* **Flow-Level Evaluator**: Resolves every origin-destination flow $(s, d)$ against the link failure state. Flows traversing a severed link attempt fast reroute: (1) primary shortest path, (2) LFC alternate next-hop, (3) multi-hop BPP repair tunnel, or (4) dropped.
* **Capacity Derating Model**: Follows proportional fluid allocation. When transient shifted traffic exceeds physical link bandwidth $C_k$, delivery is derated by the bottleneck survival ratio $\min(1, C_k / L_k)$.
* **Dual Estimands**:
  * `unconditional` (Default): Every link fails independently with probability $p_e$. Evaluates global network availability including zero-failure trials.
  * `conditional`: Exactly one link is drawn per trial. Evaluates delivery performance conditioned on a failure occurrence.
* **Deterministic Tie-Breaking**: All shortest path and greedy candidate evaluations enforce strict tie-breaking by node index and link pair tuple, guaranteeing reproducibility across platforms.

---

## 8. License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete details.

---

## 9. Citation

If you use this codebase, the C-TA-HLP algorithm, or the evaluation framework in your research, please cite:

```bibtex
@article{khan2026ctahlp,
  title={{C-TA-HLP: Capacity-Constrained Traffic-Adaptive Hybrid Link Protection for Carrier-Grade IP Networks}},
  author={Khan, Md. Iqbal Haider and Imran, Md Latiful Khabir Khan},
  journal={IEEE Transactions on Network and Service Management},
  year={2026},
  note={Under Review}
}
```

For the baseline two-stage HLP scheme, please also cite the foundational work:

```bibtex
@article{geng2020hlp,
  title={A Hybrid Link Protection Scheme for Ensuring Network Service Availability in Link-State Routing Networks},
  author={Geng, Haijun and Zhang, Han and Shi, Xingang and Wang, Zhiliang and Yin, Xia and Zhang, Ju and Hu, Zhiguo and Wu, Yong},
  journal={Journal of Communications and Networks},
  volume={22},
  number={1},
  pages={46--60},
  year={2020},
  doi={10.1109/JCN.2019.000056}
}
```
