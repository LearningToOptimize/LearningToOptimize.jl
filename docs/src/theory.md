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
`JuMP.dual` (MOI documentation, "Duality"): for
``\min a_0^\top x + b_0`` s.t. ``A_i x + b_i \in \mathcal{C}_i`` the dual is
``\max -\sum_i b_i^\top y_i + b_0`` s.t. ``a_0 - \sum_i A_i^\top y_i = 0``,
``y_i \in \mathcal{C}_i^*``; for maximisation the dual is
``\min \sum_i b_i^\top y_i + b_0`` s.t. ``a_0 + \sum_i A_i^\top y_i = 0``,
with the *same* cones. Hence the signs of the duals depend only on the sets,
not on the objective sense: ``y_k \in \mathcal{K}_k^*`` for conic blocks, and
for a box row ``y_r \ge 0`` if only the lower side is finite, ``y_r \le 0`` if
only the upper side is finite, free otherwise. An `Interval` row (not a cone)
is treated as the pair ``f \ge l`` (multiplier ``\max(y_r,0)``) and
``f \le u`` (multiplier ``\min(y_r,0)``), which is the sign convention solvers
report for its `ConstraintDual`; this is checked numerically (strong duality
on a QP with an `Interval` row). Writing ``f_r = a_r^\top x + b_r^\top p +
b^0_r`` for the function value of row ``r`` (all rows, box and conic), the
Lagrangian of the minimisation form (objective ``s f``) is

```math
L(x, y; p) = s f(x; p) - \sum_{r} y_r\, f_r
  + \sum_{r \in \text{box}} \big(\max(y_r, 0)\, l_r + \min(y_r, 0)\, u_r\big).
\tag{T4}
```

For a box row the last two terms equal ``-\max(y_r,0)(f_r - l_r) -
\min(y_r,0)(f_r - u_r)``; for a conic block they reduce to ``-y_k^\top f_k``.
For maximisation, ``-L`` is exactly the MOI Lagrangian ``f + \sum_i y_i^\top
(A_i x + b_i)``, which is why the minimisation form uses the same ``y``.

and the dual function is ``d(y; p) = \inf_x L(x, y; p)``. With
``r(y; p) = s(c + H p) - A^\top y``:

- if ``Q = 0`` (LP/conic; this is the MOI dual objective above): ``d(y;p) = s(c_p^\top p + c_0) - y^\top(Bp + b)
  + \text{box terms}`` when ``r(y;p) = 0``, and ``-\infty`` otherwise;
- if ``sQ \succeq 0``: ``d(y;p) = s(c_p^\top p + c_0) - y^\top(Bp + b)
  + \text{box terms} - \tfrac12 r^\top (sQ)^{+} r`` when ``r \in
  \operatorname{range}(sQ)``, and ``-\infty`` otherwise (for invertible ``Q``
  this is the MOI dual of a QP, ``-\tfrac12 z^\top Q^{-1} z + b_0 - \sum_i
  y_i^\top b_i`` with ``z = -r``);
- if ``sQ`` is not positive semidefinite, ``d(y; p) = -\infty`` for every
  ``y``: no bound is claimed, even when the class was declared
  `ConvexQuadratic` by hand.

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
instances (dual feasibility does not depend on right-hand-side parameters),
and on a strictly convex QP with random sign-feasible duals (every such ``y``
is dual feasible because ``\operatorname{range}(sQ)`` is everything).

*Tolerance.* Numerically, ``r`` is accepted when ``\|r\|_\infty \le``
`atol` (or the range residual is). The bound then holds up to the residual:
since ``L`` is affine in ``x`` along the residual, ``d(y;p) \le
s\,\mathrm{OPT}(p) + \|r\|_\infty \|x^\star\|_1`` for any optimal
``x^\star``. Methods that need exact bounds (DLL) complete ``y`` so that
``r = 0`` holds by construction.

**DLLBoundedCompletionValid**. The DLL paper (Tanneau and Van Hentenryck,
2024) writes the primal as ``\min c^\top x`` s.t. ``Ax \succeq_{\mathcal K}
b``, ``Hx \succeq_{\mathcal C} h`` (its eq. (6)) and the dual as ``\max b^\top
y + h^\top z`` s.t. ``A^\top y + H^\top z = c``, ``y \in \mathcal K^*``,
``z \in \mathcal C^*`` (eq. (7)); with ``f = Ax - b`` this is exactly the MOI
form above, so `dual_bound` of a completed DLL prediction is the bound
``b^\top \hat y + h^\top \hat z`` of eq. (7a). Its Theorem 1 assumes that for
every ``\hat y \in \mathcal K^*`` the inner problem (8) is strictly feasible
in ``Hx \succ_{\mathcal C} h`` and bounded; then a completion ``\hat z``
exists and ``(\hat y, \hat z)`` is dual feasible (Theorem 2: the optimal
completion gives the strongest bound for fixed ``\hat y``). For finite
variable bounds (its Example 1) the completion is ``\hat z^l = |c -
A^\top \hat y|^+``, ``\hat z^u = |c - A^\top \hat y|^-`` (``\hat z^u \le 0``
in MOI signs). The tests apply this completion to random ``\hat y`` and check
`ValidDualBound`, and check that ``\hat y = y^\star`` recovers the optimum.

**StrongDualityAtOptimum**. *Assumptions:* convex class and finite optimal
value; for LPs and convex QPs with affine constraints nothing else is needed,
for general cones a constraint qualification (e.g. Slater) is assumed.
*Claim:* at a primal-dual optimal pair returned by a solver, the bound equals
the optimal value (to solver tolerance). Tested on LP (min and max), convex
QP (min, and concave max), a QP with singular PSD Hessian and an SOCP.

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
instead. Deviations: none in the formula. Variable bounds are inequalities
``g`` of eq. (2) like any other, and distances to non-decomposable cones
(e.g. second-order cones) play the role of ``\mathrm{ReLU}(g)`` for conic
constraints ``g(x) \in \mathcal K``, which eq. (2) does not cover. The loss
is averaged over the batch (the expectation in the training objective).

**PenaltyLossMatchesDC3Eq2**: the loss equals eq. (2) written out by hand for
a small LP. **PenaltyLossAtOptimum**: at a feasible point the penalty vanishes
and the loss equals the mean of ``s f`` (also for maximisation).
**PenaltyLossGradient**: the Zygote gradient of the batch loss equals central
finite differences, on an LP and on an SOCP (where the cone distance is
active).

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
