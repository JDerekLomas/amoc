# Physics formulation — amoc-jax

This document records the equations, the discretization choices, and the
non-obvious derivations that shaped the code. It is the reference to consult
when adding a new term, debugging a blow-up, or asking "why does this differ
from the existing browser model?"

## 0. Conventions

- λ = longitude, φ = latitude. Cell-centered grid; longitude is periodic
  over 360°, latitude is bounded.
- Streamfunction ψ on a sphere of radius R:
  $u = -\tfrac{1}{R}\partial_\varphi\psi,\quad v = \tfrac{1}{R\cos\varphi}\partial_\lambda\psi$.
  This is divergence-free on a sphere.
- All v1a parameters are dimensionless. We use grid units (dλ = dφ = 1, R = 1)
  and absorb physical scaling into parameter values. Calibration to Sverdrups
  / m·s⁻¹ happens in v1b/c when we add buoyancy.

## 1. The barotropic vorticity equation (v1a)

In dimensional form on a sphere:

$$\partial_t\zeta + \tfrac{1}{R^2\cos\varphi}\,J(\psi,\zeta) + \beta v
= \tfrac{1}{\rho H}\,\mathrm{curl}(\boldsymbol\tau) - r\,\zeta + A\,\nabla^2\zeta$$

where $J(\psi,\zeta)=\partial_\lambda\psi\,\partial_\varphi\zeta -
\partial_\varphi\psi\,\partial_\lambda\zeta$ and $\beta=\tfrac{1}{R}\partial_\varphi f
= \tfrac{2\Omega\cos\varphi}{R}$.

In dimensionless grid units (R = 1, dλ = dφ = 1):

$$\partial_t\zeta = -\tfrac{1}{\cos\varphi}J_{\text{grid}}(\psi,\zeta) - \beta\,\partial_i\psi
- r\,\zeta + A\,\nabla^2_{\text{grid}}\zeta + \mathrm{curl}(\tau)$$

That is what `physics.vorticity_rhs` computes. The mask multiplies the
right-hand side at land cells so ζ never accumulates over land.

## 2. Why β·v has no cos(φ) factor — and what it cost to find that out

Substitute $v = \tfrac{1}{R\cos\varphi}\partial_\lambda\psi$ into β·v:

$$\beta v = \underbrace{\tfrac{2\Omega\cos\varphi}{R}}_{\beta} \cdot
\underbrace{\tfrac{1}{R\cos\varphi}\partial_\lambda\psi}_{v}
= \tfrac{2\Omega}{R^2}\,\partial_\lambda\psi$$

The cos(φ) in β cancels the 1/cos(φ) in v. The physical β·v term has **no
metric factor in (λ, φ) coordinates.** In dimensionless grid units that
becomes simply `beta * dpsi/di`.

The existing browser model has `betaV = beta * cos(lat) * dpsi/dx` per its
PHYSICS_REGISTRY. That double-counts the cosine: it suppresses β where it
should be strongest (high latitudes) and amplifies it near the equator. We
inherited the same bug and saw zonal flow instead of gyres. Removing the
spurious cos(φ) is what made gyres appear.

## 3. Discretization

### 3a. Poisson inversion (`poisson.py`)

The "grid Laplacian" is the bare 5-point stencil with periodic-x and
Dirichlet-y (ψ = 0 at j = -1 and j = ny). Solver:

1. FFT in x → eigenvalue $\lambda_x[k_x] = 2(\cos(2\pi k_x/n_x) - 1)$.
2. DST-I in y → eigenvalue $\lambda_y[k_y] = 2(\cos(\pi(k_y+1)/(n_y+1)) - 1)$.
3. Divide by $\lambda_x + \lambda_y$ in spectral space.
4. Inverse transforms.

**Open formulation question.** The existing model and v1a both use the bare
grid Laplacian and put the cos(φ) corrections on the *dynamics* operators.
The textbook spherical Laplacian — $\frac{1}{R^2}\!\left[\frac{\partial_\lambda^2\psi}{\cos^2\varphi} + \frac{1}{\cos\varphi}\partial_\varphi(\cos\varphi\,\partial_\varphi\psi)\right]$
— would put the cos(φ) inside the Poisson solve and define ζ as the true
spherical relative vorticity. Both choices are internally consistent; the
spherical one is more parameter-meaningful; the grid one is easier to invert
with FFT/DST. **Defer the decision to v1b** when we add the second layer —
the spherical operator is more important when ψ changes sign across coupled
layers and we want comparison against textbook Bryan/Cessi scaling.

### 3b. Arakawa Jacobian (`physics.arakawa_jacobian`)

The 9-point Arakawa (1966) form $\tfrac{1}{12}(J_{++} + J_{+\times} + J_{\times+})$ —
each is a different second-order discretization of $J$, and their average
conserves discretely:

- $\sum a\,J(a,b) = 0$ (energy conservation when a = ψ, b = ζ)
- $\sum J(a,b)^2$ etc. (enstrophy)
- $J(A, A) \equiv 0$ at every grid point (verified in `test_jacobian_self_is_zero`)

Spherical metric: divide by $\cos\varphi$ at each row.

### 3c. Time integration (`step.py`)

Heun's RK2: predictor + corrector, with the Poisson solve re-run at the
half-step so ψ is always consistent with ζ at every RHS evaluation. This
matters more once we add buoyancy in v1b — a stale ψ in the half-step would
violate the thermal-wind balance.

`run_with_history` wraps `lax.scan` so a 10000-step integration is one JIT
trace, one launch.

## 4. Sign conventions (the bit that's easy to get wrong)

Convention used throughout this codebase:

$$u = -\partial_y\psi, \qquad v = +\partial_x\psi, \qquad \zeta = \nabla^2\psi$$

This is internally consistent (it makes $\zeta = \partial_xv - \partial_y u
= \nabla^2\psi$ automatically) but it has one consequence that is easy to
mis-remember: **a ψ maximum is anticyclonic in the Northern Hemisphere.**

Quick derivation. Around a max of ψ, $\partial_y\psi$ flips sign so $u =
-\partial_y\psi$ does too — eastward on the south flank, westward on the
north flank. Likewise $v$ — northward on the west flank, southward on
the east flank. That's clockwise looking down at the NH, which is
anticyclonic.

So in NH:
- **ψ maximum (red in our colormap)** = anticyclonic = subtropical gyre.
- **ψ minimum (blue)** = cyclonic = subpolar gyre.

Wind-curl driving:
- Positive curl(τ) → friction balance gives ζ > 0 → Poisson(ζ > 0) gives
  ψ minimum → cyclonic gyre. NH **subpolar** is forced by positive curl.
- Negative curl(τ) → ζ < 0 → ψ maximum → anticyclonic gyre. NH
  **subtropical** is forced by negative curl. ✓

The same reasoning works in the Southern Hemisphere with the sense
flipped by the orientation of f.

This convention matches Vallis 2017 §13 and pyqg, and is enforced by
the `test_curl_sign_drives_correct_gyre_sense` regression in
`tests/test_correctness.py`. If anyone refactors the velocity definitions
or the Poisson sign in the future, that test should fail.

## 5. Boundary conditions and the land mask

- **x (longitude):** periodic. ψ wraps. Implemented via `jnp.roll`.
- **y (latitude):** Dirichlet ψ = 0 just outside the first/last rows (the
  poles). Implemented in the Poisson solver via the sine basis and in the
  Jacobian via padding with zeros.
- **Land mask:** the existing model's bit-packed mask (`data/mask.json`,
  1024×512) is decoded by `data.load_mask`. The RHS is multiplied by the
  mask, so ζ stays at zero over land. The **Poisson solve runs over the full
  rectangle** — this is deliberate. Zeroing ψ over land creates infinite
  gradients at coastlines that propagate into the Jacobian and blow the
  integrator up. ψ over land is non-physical but harmless because the
  masked RHS skips those cells.

## 6. Stability, scaling, and parameter intuition

A few back-of-the-envelope balances we use to set parameters:

- **Sverdrup balance:** $\beta v \sim \text{curl}(\tau)$ → $v \sim \text{curl}/\beta$.
  In a basin of width $L$, $\psi_{\max} \sim L\,\text{curl}/\beta$. This is
  why ψ blows up at small β — it just keeps growing. Choosing β ~ basin
  width keeps ψ ~ O(1) in dimensionless units.
- **Spin-up timescale:** $\tau_{\text{fric}} = 1/r$. Run ≥ 4τ to reach
  ~98% of steady state.
- **CFL:** $\Delta t < \Delta x / |U|_{\max}$. With ψ ~ 100 in current
  scaling, |U| ~ 100/L_grid, so dt < L_grid/100. Halving the grid spacing
  halves the maximum stable dt.
- **Munk boundary layer:** $\delta_M = (A/\beta)^{1/3}$. To resolve the
  western boundary current we need $\delta_M \gtrsim \Delta x$.
- **Stommel boundary layer:** $\delta_S = r/\beta$. Same idea with friction.

Current v1a values (`scripts/run.py` defaults): β = 2, r = 0.04, A = 0.005,
wind = 0.02, dt = 0.01 at 256×128. These are tuned for stability, not for
physical fidelity. The β value is a placeholder until v1b/c calibrate
against observed ψ_AMOC.

## 7. References cited above

- Arakawa, A. (1966). *J. Comput. Phys.* 1:119.
- Cessi, P. (1994). *J. Phys. Oceanogr.* 24:1911.
- Munk, W.H. (1950). *J. Meteorol.* 7:80.
- Pedlosky, J. (1987). *Geophysical Fluid Dynamics*, 2nd ed.
- Stommel, H. (1948). *Trans. Am. Geophys. Union* 29:202.
- Stommel, H. (1961). *Tellus* 13:224.
- Sverdrup, H.U. (1947). *Proc. Natl. Acad. Sci.* 33:318.
- Vallis, G.K. (2017). *Atmospheric and Oceanic Fluid Dynamics*, 2nd ed.
  Cambridge.
- Wright, D.G. & T.F. Stocker (1991). *J. Phys. Oceanogr.* 21:1713.
