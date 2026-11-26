"""
    AbstractParametricProblem

Supertype of parametric problems, independent of the modeling front end.
[`ParametricProblem`](@ref) is the JuMP front end; other front ends (for example
an ExaModels-based one) subtype this and implement [`problem_class`](@ref),
[`num_parameters`](@ref), [`num_decision_variables`](@ref),
[`num_duals`](@ref), [`objective_sign`](@ref) and [`default_evaluator`](@ref)
(and [`solve_instances`](@ref) to support solver-labeled data).
"""
abstract type AbstractParametricProblem end

"""
    ParametricProblem(model::JuMP.Model; class = detect_problem_class(model))

A parametric optimization problem: a JuMP model in which the problem data that
changes between instances are declared as parameters
(`@variable(model, p in MOI.Parameter(value))`).

The wrapper fixes three *layouts* used throughout the ecosystem:

  - **inputs**: [`parameters`](@ref), in creation order (`num_parameters × B`);
  - **primal**: [`decision_variables`](@ref) = all non-parameter variables in
    creation order (`num_decision_variables × B`);
  - **dual**: [`constraint_refs`](@ref), including variable bounds but not
    parameter or integrality constraints; constraint `i` owns the rows
    [`dual_layout`](@ref)`(problem)[i]` (`num_duals × B`). Signs follow the
    MathOptInterface convention returned by `JuMP.dual`.

`class` is a trait ([`AbstractProblemClass`](@ref)) used by
[`supports`](@ref); pass it explicitly to override detection (e.g. to declare
a convex QCQP or a custom class).

Fields are considered internal; use the accessor functions.
"""
struct ParametricProblem{C<:AbstractProblemClass} <: AbstractParametricProblem
    model::JuMP.Model
    parameters::Vector{JuMP.VariableRef}
    variables::Vector{JuMP.VariableRef}
    constraints::Vector{JuMP.ConstraintRef}
    layout::Vector{UnitRange{Int}}
    class::C
    cache::Dict{Symbol,Any}
end

function ParametricProblem(
    model::JuMP.Model;
    class::AbstractProblemClass = detect_problem_class(model),
)
    allvars = JuMP.all_variables(model)
    params = filter(JuMP.is_parameter, allvars)
    vars = filter(!JuMP.is_parameter, allvars)
    cons = JuMP.ConstraintRef[]
    for (F, S) in JuMP.list_of_constraint_types(model)
        (S <: MOI.Parameter || S <: _INTEGRALITY_SETS) && continue
        append!(cons, JuMP.all_constraints(model, F, S))
    end
    layout = Vector{UnitRange{Int}}(undef, length(cons))
    offset = 0
    for (i, c) in enumerate(cons)
        d = MOI.dimension(JuMP.moi_set(JuMP.constraint_object(c)))
        layout[i] = (offset+1):(offset+d)
        offset += d
    end
    return ParametricProblem(model, params, vars, cons, layout, class, Dict{Symbol,Any}())
end

"""
    problem_class(problem::ParametricProblem) -> AbstractProblemClass
"""
problem_class(p::ParametricProblem) = p.class

"""
    parameters(problem::ParametricProblem) -> Vector{JuMP.VariableRef}

Parameters of the problem, in input order.
"""
parameters(p::ParametricProblem) = p.parameters

"""
    decision_variables(problem::ParametricProblem) -> Vector{JuMP.VariableRef}

Non-parameter variables, in primal-prediction order.
"""
decision_variables(p::ParametricProblem) = p.variables

"""
    constraint_refs(problem::ParametricProblem) -> Vector{JuMP.ConstraintRef}

Constraints (including variable bounds) in dual-prediction order.
"""
constraint_refs(p::ParametricProblem) = p.constraints

"""
    num_parameters(problem::ParametricProblem) -> Int
"""
num_parameters(p::ParametricProblem) = length(p.parameters)

"""
    num_decision_variables(problem::ParametricProblem) -> Int
"""
num_decision_variables(p::ParametricProblem) = length(p.variables)

"""
    num_duals(problem::ParametricProblem) -> Int

Total scalar dimension of all constraints in [`constraint_refs`](@ref).
"""
num_duals(p::ParametricProblem) = isempty(p.layout) ? 0 : last(p.layout[end])

"""
    dual_layout(problem::ParametricProblem) -> Vector{UnitRange{Int}}

Rows of a dual prediction owned by each constraint in [`constraint_refs`](@ref).
"""
dual_layout(p::ParametricProblem) = p.layout

"""
    objective_sign(problem) -> Int

`1` for minimization (and feasibility) problems, `-1` for maximization;
`objective_sign(problem) * objective` is always to be minimized.
"""
objective_sign(p::ParametricProblem) =
    JuMP.objective_sense(p.model) == MOI.MAX_SENSE ? -1 : 1

"""
    parameter_values(problem::ParametricProblem) -> Vector{Float64}

Current values of the parameters stored in the JuMP model (a single instance).
"""
parameter_values(p::ParametricProblem) = JuMP.parameter_value.(p.parameters)

function Base.show(io::IO, p::ParametricProblem)
    print(
        io,
        "ParametricProblem{",
        nameof(typeof(p.class)),
        "}: ",
        num_parameters(p),
        " parameters, ",
        num_decision_variables(p),
        " variables, ",
        length(p.constraints),
        " constraints (",
        num_duals(p),
        " dual rows)",
    )
end

# Treat a vector as a batch of size one.
_asbatch(x::AbstractMatrix) = x
_asbatch(x::AbstractVector) = reshape(x, :, 1)
