# ODE Simulation via Diffusion Processes

## Overview

Three approaches allow AI to simulate ODEs. Each exploits a different paradigm: **Neural ODE** directly parameterises the derivative, **Score-based DDPM** models the trajectory distribution through iterative denoising, and **Probability Flow ODE** reformulates denoising as a deterministic ODE.

```
x(t) target ODE
      │
      ├─ Neural ODE          → learns dx/dt = fθ(x,t)
      ├─ Score-based DDPM    → learns the distribution p(x₀)
      └─ Probability Flow    → denoising = deterministic ODE
```

---

## Approach 1 — Neural ODE

### Principle

The neural network **directly parameterises the derivative** of the state. The trajectory is then obtained by numerical integration.

$$\frac{dx}{dt} = f_\theta(x, t), \quad x(t_0) = x_0$$

Integration is made differentiable through the **adjoint method**, which allows backpropagating gradients without storing all intermediate steps.

### Architecture

```
Input  : (x, t)   — current state + continuous time
Network: MLP or ResNet with sinusoidal time embedding
Output : dx/dt    — same dimension as x
```

### Minimal PyTorch example (`torchdiffeq`)

```python
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    """Parameterises dx/dt = fθ(x, t)"""
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden),  nn.Tanh(),
            nn.Linear(hidden, dim)
        )

    def forward(self, t, x):
        # t : scalar, x : (B, dim)
        t_vec = t.expand(x.shape[0], 1)
        return self.net(torch.cat([x, t_vec], dim=-1))


# Training
func = ODEFunc(dim=2)
optimizer = torch.optim.Adam(func.parameters(), lr=1e-3)

for x0, x_target, t_span in dataloader:
    # x_target : (B, T_obs, dim) — observed trajectory
    t_eval = torch.linspace(0, 1, x_target.shape[1])
    x_pred = odeint(func, x0, t_eval, method='dopri5')  # (T, B, dim)
    loss = ((x_pred.permute(1,0,2) - x_target) ** 2).mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()


# Inference
@torch.no_grad()
def simulate(func, x0, t_span):
    return odeint(func, x0, t_span, method='dopri5')
```

### Advantages / limitations

| ✓ Advantages | ✗ Limitations |
|---|---|
| Few training samples required | No uncertainty on the trajectory |
| Fast inference (single pass) | May extrapolate poorly out of distribution |
| Adaptive solver (variable precision) | Unstable for chaotic dynamics |
| Natively continuous in time | Requires labelled trajectories |

---

## Approach 2 — Score-based DDPM

### Principle

**DDPM (Denoising Diffusion Probabilistic Model)** applied to ODEs treats **trajectories** `x(t)` as data to be modelled. The model learns the distribution of solutions, then generates new ones through iterative denoising.

---

## 1. Forward process (noising)

The trajectory is progressively destroyed by adding Gaussian noise at each step `t`:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t;\, \sqrt{1-\beta_t}\, x_{t-1},\, \beta_t I\right)$$

**Direct shortcut** (without recursion):

$$x_t = \sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1 - \bar{\alpha}_t}\, \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, I)$$

with $\bar{\alpha}_t = \prod_{i=1}^{t}(1-\beta_i)$.

| t | $\bar{\alpha}_t$ | Content |
|---|---|---|
| 0 | 1.0 | Clean ODE trajectory |
| T/2 | ~0.3 | Signal + heavy noise |
| T | ~0.0 | Pure Gaussian noise |

---

## 2. Reverse process (denoising)

The network $\varepsilon_\theta$ learns to invert the noising:

$$p_\theta(x_{t-1} \mid x_t) = \mathcal{N}\!\left(x_{t-1};\, \mu_\theta(x_t, t),\, \sigma_t^2 I\right)$$

with:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}}\,\varepsilon_\theta(x_t, t)\right)$$

---

## 3. Loss function

$$\mathcal{L} = \mathbb{E}_{t,\, x_0,\, \varepsilon}\!\left[\|\varepsilon - \varepsilon_\theta(\sqrt{\bar{\alpha}_t}\, x_0 + \sqrt{1-\bar{\alpha}_t}\,\varepsilon,\; t)\|^2\right]$$

The network simply predicts **the added noise**, not the trajectory directly.

---

## 4. Network architecture (for ODEs)

```
Input  : (x_t, t)
         x_t  → discretised trajectory vector (length L)
         t    → integer ∈ [1, T], encoded via embedding

Network: MLP or 1D U-Net
         - Time embedding  : sinusoidal or nn.Embedding
         - Hidden layers   : Linear + SiLU (or attention for long trajectories)

Output : ε̂ ∈ ℝᴸ  (estimated noise, same dimension as x_t)
```

### Minimal PyTorch example

```python
import torch
import torch.nn as nn

class ScoreNet(nn.Module):
    def __init__(self, traj_len=100, hidden=256):
        super().__init__()
        self.time_emb = nn.Embedding(1000, 32)
        self.net = nn.Sequential(
            nn.Linear(traj_len + 32, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden),        nn.SiLU(),
            nn.Linear(hidden, traj_len)
        )

    def forward(self, x_t, t):
        return self.net(torch.cat([x_t, self.time_emb(t)], dim=-1))


# Training loop
model = ScoreNet(traj_len=100)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
alpha_bar = ...  # tensor of size T+1

for x0 in dataloader:                         # x0 : (B, L) ODE trajectories
    t   = torch.randint(1, 1000, (x0.shape[0],))
    eps = torch.randn_like(x0)
    ab  = alpha_bar[t].unsqueeze(-1)
    x_t = torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps
    loss = ((eps - model(x_t, t)) ** 2).mean()
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

---

## 5. Inference (simulation)

```python
@torch.no_grad()
def sample(model, T=1000, traj_len=100):
    x = torch.randn(1, traj_len)          # pure noise x_T
    for t in reversed(range(1, T + 1)):
        ts = torch.tensor([t])
        eps_hat = model(x, ts)
        # DDPM reverse formula
        ab, ab_prev = alpha_bar[t], alpha_bar[t-1]
        x0_hat = (x - torch.sqrt(1-ab) * eps_hat) / torch.sqrt(ab)
        mean = torch.sqrt(ab_prev) * x0_hat + torch.sqrt(1-ab_prev) * eps_hat
        z = torch.randn_like(x) if t > 1 else 0
        x = mean + torch.sqrt(beta[t]) * z
    return x  # simulated trajectory
```

---

## 6. Noise schedules βₜ

| Schedule | Formula | Recommended for |
|---|---|---|
| Linear | $\beta_t = \beta_1 + (t/T)(\beta_T - \beta_1)$ | Basic, original DDPM |
| Cosine | $\bar{\alpha}_t = \cos^2\!\left(\frac{t/T + s}{1+s}\cdot\frac{\pi}{2}\right)$ | ODE trajectories (better signal preservation early on) |
| Sigmoid | $\beta_t = \sigma(-6 + 12t/T)$ | Nonlinear dynamics |

---

## 7. Inference acceleration

| Method | Steps | Quality | Notes |
|---|---|---|---|
| Standard DDPM | 1000 | ★★★ | Reference, slow |
| DDIM | 20–50 | ★★★ | Deterministic, invertible |
| DPM-Solver | 10–20 | ★★★ | Fast, recommended |
| DDIM + PLMS | 25 | ★★☆ | Good trade-off |

---

## 8. Extensions for ODEs

### Parametric conditioning

Pass ODE parameters (e.g. `μ, σ, k`) as a condition:

```python
# Conditioning by concatenation
def forward(self, x_t, t, params):
    h = torch.cat([x_t, self.time_emb(t), params], dim=-1)
    return self.net(h)
```

### Ensemble generation (uncertainty quantification)

Generate N trajectories from N different noise samples → empirical distribution of solutions.

### Inverse problem

Condition on partial observations `y = x₀[mask]` via score guidance:

$$\nabla_{x_t} \log p(y \mid x_t) \approx -\frac{1}{\sigma^2}(x_0^{\text{pred}} - y)[mask]$$

---

## Approach 3 — Probability Flow ODE

### Principle

Every diffusion SDE admits a **deterministic ODE** that exactly preserves the marginals `p_t(x)` at every instant. This ODE is called the *probability flow ODE* (Song et al., 2020):

$$\frac{dx}{dt} = f(x,t) - \frac{1}{2}\,g(t)^2\,\nabla_x \log p_t(x)$$

where `∇_x log p_t(x)` is the **score** — the log-density gradient — learned by the same network `εθ` as DDPM. In practice:

$$\nabla_x \log p_t(x) \approx -\frac{\varepsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$$

### Relationship with DDPM and DDIM

| | DDPM | DDIM | Probability Flow ODE |
|---|---|---|---|
| Stochastic? | Yes | No | No |
| Based on | Markov chain | DDPM sub-sequence | Continuous ODE |
| Invertible? | ✗ | Partially | ✓ exact |
| Min. steps | ~1000 | ~20 | ~10–50 |

DDIM is a **discretisation** of the probability flow ODE — both converge to the same trajectory as the step size goes to zero.

### SDE → ODE formulation

The Variance Preserving SDE (VP-SDE, general framework):

$$dx = -\frac{1}{2}\beta(t)\,x\,dt + \sqrt{\beta(t)}\,dW_t \quad \text{(forward)}$$

Its equivalent probability flow ODE:

$$\frac{dx}{dt} = -\frac{1}{2}\beta(t)\left[x + \nabla_x \log p_t(x)\right]$$

### Example: inference with an ODE solver

```python
import torch
from torchdiffeq import odeint

def score_fn(x_t, t_continuous, model, alpha_bar_fn):
    """Converts continuous t ∈ [0,1] to a discrete step and computes the score."""
    t_disc = (t_continuous * 999).long().clamp(1, 999)
    ab = alpha_bar_fn(t_disc).unsqueeze(-1)
    eps_hat = model(x_t, t_disc)
    return -eps_hat / torch.sqrt(1 - ab)

def probability_flow_ode(model, alpha_bar_fn, beta_fn):
    def ode_func(t, x):
        score = score_fn(x, t, model, alpha_bar_fn)
        b = beta_fn(t)
        return -0.5 * b * (x + score)   # dx/dt
    return ode_func

@torch.no_grad()
def sample_pfode(model, alpha_bar_fn, beta_fn, traj_len=100, steps=50):
    x = torch.randn(1, traj_len)               # pure noise at t=1
    t_span = torch.linspace(1.0, 0.0, steps)   # integrate from T→0
    ode_func = probability_flow_ode(model, alpha_bar_fn, beta_fn)
    trajectory = odeint(ode_func, x, t_span, method='rk4')
    return trajectory[-1]  # x at t=0: simulated trajectory


# Encoding (exact inversion): x₀ → xT
@torch.no_grad()
def encode_pfode(model, x0, alpha_bar_fn, beta_fn, steps=50):
    t_span = torch.linspace(0.0, 1.0, steps)   # integrate from 0→T
    ode_func = probability_flow_ode(model, alpha_bar_fn, beta_fn)
    return odeint(ode_func, x0, t_span, method='rk4')[-1]  # xT
```

### Specific use cases

**Inverse problem / interpolation**: exact invertibility allows encoding two trajectories `x₀^A` and `x₀^B` into latent space (`xT^A`, `xT^B`), linearly interpolating, then decoding — useful for exploring the solution space.

**Maximum acceleration**: combined with DPM-Solver++ (order 2–3), simulation in 5–10 steps with quality close to 1000-step DDPM.

### Advantages / limitations

| ✓ Advantages | ✗ Limitations |
|---|---|
| Exact inversion (encoding) | Same data requirements as DDPM |
| Fast inference (ODE solver) | More complex implementation |
| Interpolable latent space | Sensitive to ODE solver choice |
| No stochastic variance | No uncertainty on the trajectory |

---

## 9. Key libraries

| Library | Approach | Usage |
|---|---|---|
| `torchdiffeq` | Neural ODE + PF-ODE | Differentiable ODE integration, adjoint method |
| `diffrax` (JAX) | Neural ODE + PF-ODE | High-performance ODE solvers in JAX |
| `diffusers` (HuggingFace) | DDPM / DDIM | Ready-to-use score-based pipelines |
| `score_sde` (Yang Song) | DDPM + PF-ODE | Reference implementation, VP/VE schedules |
| `torchsde` | Stochastic SDE | Differentiable SDE (foundation of PF-ODE) |

---

## 10. Comparative summary

| Criterion | Neural ODE | Score-based DDPM | Probability Flow ODE |
|---|---|---|---|
| Output type | Single trajectory | Stochastic ensemble | Deterministic trajectory |
| Uncertainty | ✗ | ✓ | ✗ |
| Inference speed | Fast | Slow (DDPM) / Medium (DDIM) | Fast |
| Data required | Little | A lot | A lot |
| Invertibility | ✓ | ✗ | ✓ |
| Conditioning | Easy | Easy | Moderate |

---

## 11. Bibliography

### Neural ODE

- **[1]** Chen, R. T. Q., Rubanova, Y., Bettencourt, J., & Duvenaud, D. (2018).  
  *Neural Ordinary Differential Equations.*  
  NeurIPS 2018. [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)  
  > Foundational paper — introduces the adjoint method for backpropagating through an ODE solver.

- **[2]** Rubanova, Y., Chen, R. T. Q., & Duvenaud, D. (2019).  
  *Latent ODEs for Irregularly-Sampled Time Series.*  
  NeurIPS 2019. [arXiv:1907.03907](https://arxiv.org/abs/1907.03907)  
  > Extension to irregularly-sampled time series — highly relevant for partially observed ODE data.

- **[3]** Kidger, P., Morrill, J., Foster, J., & Lyons, T. (2020).  
  *Neural Controlled Differential Equations for Irregular Time Series.*  
  NeurIPS 2020. [arXiv:2005.08926](https://arxiv.org/abs/2005.08926)  
  > Introduces Neural CDEs, more stable than Neural ODEs for continuous inputs.

- **[4]** Kidger, P. (2022).  
  *On Neural Differential Equations.*  
  PhD Thesis, University of Oxford. [arXiv:2202.02435](https://arxiv.org/abs/2202.02435)  
  > Comprehensive survey of neural ODE/SDE/CDE — recommended as a foundational reference.

---

### Score-based DDPM

- **[5]** Ho, J., Jain, A., & Abbeel, P. (2020).  
  *Denoising Diffusion Probabilistic Models.*  
  NeurIPS 2020. [arXiv:2006.11239](https://arxiv.org/abs/2006.11239)  
  > Foundational DDPM paper — defines the forward/reverse process and the denoising loss.

- **[6]** Song, Y., & Ermon, S. (2019).  
  *Generative Modeling by Estimating Gradients of the Data Distribution.*  
  NeurIPS 2019. [arXiv:1907.05600](https://arxiv.org/abs/1907.05600)  
  > Introduction of score matching and score-based models (NCSN).

- **[7]** Nichol, A., & Dhariwal, P. (2021).  
  *Improved Denoising Diffusion Probabilistic Models.*  
  ICML 2021. [arXiv:2102.09672](https://arxiv.org/abs/2102.09672)  
  > Improves DDPM: cosine schedule, learned variance — recommended for ODE applications.

- **[8]** Batzolis, G., Stanczuk, J., Schönlieb, C.-B., & Etmann, C. (2021).  
  *Conditional Image Generation with Score-Based Diffusion Models.*  
  [arXiv:2111.13606](https://arxiv.org/abs/2111.13606)  
  > Conditioning of diffusion models — applicable to conditioning on ODE parameters.

---

### Probability Flow ODE & DDIM

- **[9]** Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021).  
  *Score-Based Generative Modeling through Stochastic Differential Equations.*  
  ICLR 2021 (Outstanding Paper). [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)  
  > Key paper — unifies DDPM and score matching via SDEs, introduces the probability flow ODE.

- **[10]** Song, J., Meng, C., & Ermon, S. (2021).  
  *Denoising Diffusion Implicit Models.*  
  ICLR 2021. [arXiv:2010.02502](https://arxiv.org/abs/2010.02502)  
  > Introduces DDIM — non-Markovian, deterministic inference, 10–50× faster than DDPM.

- **[11]** Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).  
  *DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps.*  
  NeurIPS 2022. [arXiv:2206.00927](https://arxiv.org/abs/2206.00927)  
  > High-order ODE solver for the probability flow ODE — 10–20 steps suffice.

- **[12]** Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).  
  *DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models.*  
  [arXiv:2211.01095](https://arxiv.org/abs/2211.01095)  
  > Conditional extension of DPM-Solver, order 2–3, recommended for fast inference.

---

### Applications to scientific ODEs

- **[13]** Rackauckas, C., Ma, Y., Martensen, J., Warner, C., Zubov, K., Supekar, R., ... & Edelman, A. (2020).  
  *Universal Differential Equations for Scientific Machine Learning.*  
  [arXiv:2001.04385](https://arxiv.org/abs/2001.04385)  
  > Combines numerical solvers and neural networks for physics-based ODEs.

- **[14]** Haussmann, M., Gerwinn, S., Look, A., Rakitsch, B., & Kandemir, M. (2021).  
  *Inferring Latent Dynamics Underlying Neural Population Activity with Variational Sequential Monte Carlo.*  
  ICML 2021. [arXiv:2105.04390](https://arxiv.org/abs/2105.04390)  
  > Bayesian inference on latent dynamics — connections with the score-based approach.

- **[15]** Gao, Y., Shi, J., Luo, D., Wen, H., & Li, Q. (2023).  
  *EHRDiff: Exploring Realistic EHR Synthesis with Diffusion Models.*  
  [arXiv:2303.05656](https://arxiv.org/abs/2303.05656)  
  > Applied example: diffusion for generating medical time series (format close to ODE trajectories).

---

### Further resources

- **Blog**: Lilian Weng, *"What are Diffusion Models?"* (2021) — [lilianweng.github.io](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  
  > Very thorough pedagogical overview of DDPM, score matching, and SDEs.

- **Blog**: Yang Song, *"Generative Modeling by Estimating Gradients of the Data Distribution"* — [yang-song.net](https://yang-song.net/blog/2021/score/)  
  > Intuitive explanation of score matching by the author of the reference papers.

- **Code**: `score_sde` (Yang Song) — [github.com/yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch)  
  > Reference implementation of VP-SDE / VE-SDE / Probability Flow ODE.