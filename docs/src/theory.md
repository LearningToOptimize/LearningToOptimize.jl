# Theory

This page fixes the notation used across the ecosystem and states the
mathematical claims that `L2OCore` checks numerically. Each claim has a name
(in **bold**); tests and future formal proofs refer to the same name.

## Parametric optimization

A *parametric problem* maps a parameter vector ``p \in \mathbb{R}^{n_p}`` to
an instance

```math
\mathrm{OPT}(p) \;=\; \min_{x \in \mathbb{R}^{n}} \; f(x; p)
\quad \text{s.t.} \quad g_i(x; p) \in \mathcal{S}_i, \quad i = 1, \dots, m,
\tag{T1}
```

where each ``\mathcal{S}_i`` is a MathOptInterface set (`EqualTo`,
`GreaterThan`, `Interval`, `Nonnegatives`, `SecondOrderCone`, …). In JuMP,
``p`` are variables in `MOI.Parameter`, ``x`` are all other variables
(`decision_variables`), and the constraints, including variable bounds, are
`constraint_refs` (in this order). Maximisation is handled by the sign
``s = \texttt{objective\_sign}(problem) \in \{1, -1\}``: ``s f`` is minimised.

A **proxy** ``\phi_\theta`` maps a batch of parameters ``P = [p_1, \dots,
p_B]`` to raw outputs; a method's **completion** turns raw outputs into a
prediction ``(\hat X, \hat Y)`` of primal and/or dual solutions. Training
minimises a method-specific loss ``\mathbb{E}_{p \sim \mathcal{D}}
[\ell(\phi_\theta(p); p)]`` over a parameter distribution ``\mathcal{D}``
(an `AbstractParameterSampler`).

## Matrix form

For the classes `LinearConic` and `ConvexQuadratic`, with parameters entering
affinely, `MatrixEvaluator` represents (T1) as

```math
\min_x \; \tfrac12 x^\top Q x + c^\top x + x^\top H p + c_p^\top p + c_0
\quad \text{s.t.} \quad A x + B p + b \in \mathcal{K} = \mathcal{K}_1 \times \dots \times \mathcal{K}_m .
\tag{T2}
```

Box-like sets are written as row bounds ``l \le A x + B p + b \le u``
(``l_r = u_r`` for equalities, infinite entries allowed); other cones stay as
blocks. Violations are

```math
\nu_r(x; p) = \max(l_r - a_r^\top x - b_r^\top p - b_r^0, 0) + \max(a_r^\top x + b_r^\top p + b_r^0 - u_r, 0),
\qquad
\nu_k^{\mathrm{cone}}(x;p) = \operatorname{dist}(A_k x + B_k p + b_k, \mathcal{K}_k).
\tag{T3}
```

**MatrixEvaluatorAgreesWithJuMP**: on random ``(X, P)``, objective,
constraint values and violations of `MatrixEvaluator` equal those of the
per-instance `JuMPEvaluator`.

## Lagrangian dual bound

Dual variables follow the MathOptInterface convention returned by
`JuMP.dual`: ``y_k \in \mathcal{K}_k^*`` (the dual cone) for conic blocks; for
a box row, ``y_r \ge 0`` if only the lower side is finite, ``y_r \le 0`` if
only the upper side is finite, free otherwise. In minimisation form
(objective ``s f``), the Lagrangian is

```math
L(x, y; p) = s f(x; p) - \sum_r y_r\, (a_r^\top x + b_r^\top p + b_r^0)
  + \sum_{r \in \text{box}} \big(\max(y_r, 0)\, l_r + \min(y_r, 0)\, u_r\big)
  - \sum_k y_k^\top (A_k x + B_k p + b_k),
\tag{T4}
```

and the dual function is ``d(y; p) = \inf_x L(x, y; p)``. With
``r(y; p) = s(c + H p) - A^\top y``:

- if ``Q = 0`` (LP/conic): ``d(y;p) = s(c_p^\top p + c_0) - y^\top(Bp + b)
  + \text{box terms}`` when ``r(y;p) = 0``, and ``-\infty`` otherwise;
- if ``sQ \succeq 0``: ``d(y;p) = s(c_p^\top p + c_0) - y^\top(Bp + b)
  + \text{box terms} - \tfrac12 r^\top (sQ)^{+} r`` when ``r \in
  \operatorname{range}(sQ)``, and ``-\infty`` otherwise.

`dual_bound(evaluator, Y, P)` returns ``s\, d(y;p)`` together with the dual
infeasibility (largest of the stationarity residual, sign violations and
dual-cone distances).

**ValidDualBound** (weak duality). *Assumptions:* ``y`` is dual feasible
(signs/dual cones as above and ``r(y;p) = 0`` or ``r \in \operatorname{range}
(sQ)``). *Claim:* ``d(y; p) \le s\,\mathrm{OPT}(p)``, i.e. the returned bound
is ``\le \mathrm{OPT}(p)`` for minimisation and ``\ge`` for maximisation. No
convexity is needed: for any feasible ``x``, every subtracted term in (T4) is
``\ge 0`` by the sign conventions, so ``L(x,y;p) \le s f(x;p)``. Tested on
random instances by evaluating the optimal duals of one instance at other
instances (dual feasibility does not depend on right-hand-side parameters).

**StrongDualityAtOptimum**. *Assumptions:* convex class and a constraint
qualification (Slater). *Claim:* at a primal-dual optimal pair returned by a
solver, the bound equals the optimal value (to solver tolerance). Tested on
LP (min and max), convex QP and SOCP instances.

## Guarantees

| name | claim on completed predictions, for every instance |
|:--|:--|
| `NoGuarantee` | none |
| `Feasible` | ``\max_r \nu_r \le \varepsilon`` and ``\max_k \nu_k^{\mathrm{cone}} \le \varepsilon`` |
| `EqualityFeasible` | ``\max_{r:\, l_r = u_r} \nu_r \le \varepsilon`` |
| `BoundFeasible` | ``\max_{r \in \text{bounds}} \nu_r \le \varepsilon`` |
| `ValidDualBound` | dual infeasibility ``\le \varepsilon`` and ``s(\text{bound} - \mathrm{OPT}) \le \varepsilon \max(1, |\mathrm{OPT}|)`` |

`check(guarantee, problem, P, prediction; atol = ε)` evaluates these.

## The penalty method

The reference method minimises the soft loss of eq. (2) in Donti, Rolnick and
Kolter (2021):

```math
\ell_{\mathrm{soft}}(\hat x; p) = s f(\hat x; p)
  + \rho_{\mathrm{eq}} \sum_{r:\, l_r = u_r} \nu_r(\hat x; p)^2
  + \rho_{\mathrm{ineq}} \Big(\sum_{r:\, l_r < u_r} \nu_r(\hat x; p)^2 + \sum_k \nu_k^{\mathrm{cone}}(\hat x;p)^2\Big).
\tag{T5}
```

For a box row, ``\nu_r`` is the ReLU of both one-sided constraints, so (T5)
is eq. (2) of that paper with ``\lambda_h = \rho_{\mathrm{eq}}`` and
``\lambda_g = \rho_{\mathrm{ineq}}``; `squared = false` uses the 1-norm
instead. Deviations: variable bounds are penalised like other inequalities;
conic distances are evaluated but not differentiated.

**PenaltyLossAtOptimum**: at a feasible point the penalty vanishes and the
loss equals the mean objective. **PenaltyLossGradient**: the Zygote gradient
of the batch loss equals central finite differences.

*Guarantee:* none. As the DC3 paper notes, soft losses do not enforce the
constraints and can give infeasible outputs; use a completion method when
feasibility matters.

## References

- P. L. Donti, D. Rolnick, J. Z. Kolter. *DC3: A learning method for
  optimization with hard constraints.* ICLR 2021. arXiv:2104.12225.
- J. Kotary, F. Fioretto, P. Van Hentenryck, B. Wilder. *End-to-End
  Constrained Optimization Learning: A Survey.* IJCAI 2021. arXiv:2103.16378.
- M. Tanneau, P. Van Hentenryck. *Dual Lagrangian Learning for Conic
  Optimization.* NeurIPS 2024. arXiv:2402.03086.
- M. Klamkin, M. Tanneau, P. Van Hentenryck. *Dual Interior-Point
  Optimization Learning.* arXiv:2402.02596.
- S. Park, P. Van Hentenryck. *Self-Supervised Primal-Dual Learning for
  Constrained Optimization.* AAAI 2023. arXiv:2208.09046.
- F. Fioretto, P. Van Hentenryck, T. W. K. Mak, C. Tran, F. Baldo,
  M. Lombardi. *Lagrangian Duality for Constrained Deep Learning.*
  ECML-PKDD 2020. arXiv:2001.09394.
