using Dualization

"""
    WorstCaseProblemIterator

An iterator that iterates over a set of problems and solves the worst case
problem for each problem in the set. The worst case problem is defined as
the problem that maximizes the objective function over the set of problems.

The iterator is initialized with a set of problem ids, a function that
returns the parameters for a given problem, a function that builds the
primal problem, a function that builds the dual problem, and a function
that sets the iterator for the dual problem.

The iterator returns the number of problems that were solved.

# Arguments
- `ids::Vector{UUID}`: A vector of problem ids.
- `parameters::Function`: A function `(model) -> Vector{JuMP.Variable}` that
    returns the parameters for a given problem.
- `primal_builder!::Function`: A function `(model, parameters) -> modified model` that builds the primal problem for a given set of parameters.
- `set_iterator!::Function`: A function `(model, parameters, idx) -> modified model` that modifies the dual problem for a given set of parameters and problem index.
- `optimizer`: The optimizer to use for the primal and dual problem. # TODO: Duplicate argument
"""
struct WorstCaseProblemIterator{F} <: AbstractProblemIterator
    ids::Vector{UUID}
    parameters::Function
    primal_builder!::Function
    set_iterator!::Function
    optimizer::F
    hook::Union{Nothing,Function}
    options::Any
    ext::Dict
    early_stop::Function
    function WorstCaseProblemIterator(
        ids::Vector{UUID},
        parameters::Function,
        primal_builder!::Function,
        set_iterator!::Function,
        optimizer::F;
        hook::Union{Nothing,Function} = nothing,
        options::Any = nothing,
        ext::Dict = Dict(),
        early_stop::Function = (args...) -> false,
    ) where {F}
        return new{F}(
            ids,
            parameters,
            primal_builder!,
            set_iterator!,
            optimizer,
            hook,
            options,
            ext,
            early_stop,
        )
    end
end

"""
    solve_and_record

Solves the worst case problem for a given problem iterator and records the
solution if it passes the filter function.

# Arguments
- `problem_iterator::WorstCaseProblemIterator`: The problem iterator.
- `recorder::Recorder`: The recorder.
- `idx::Integer`: The index of the problem to solve.
"""
function solve_and_record(
    problem_iterator::WorstCaseProblemIterator,
    recorder::Recorder,
    idx::Integer,
)
    # Build Primal
    model = JuMP.Model()
    parameters = problem_iterator.parameters(model)
    problem_iterator.primal_builder!(model, parameters)

    # Parameter indices
    load_moi_idx = Vector{MOI.VariableIndex}(JuMP.index.(parameters))

    # Dualize the model
    dual_st = Dualization.dualize(JuMP.backend(model); variable_parameters = load_moi_idx)

    dual_model = dual_st.dual_model
    primal_dual_map = dual_st.primal_dual_map

    # Build Dual in JuMP
    jump_dual_model = JuMP.Model()
    map_moi_to_jump = MOI.copy_to(JuMP.backend(jump_dual_model), dual_model)
    set_optimizer(jump_dual_model, problem_iterator.optimizer())
    if !isnothing(problem_iterator.hook)
        problem_iterator.hook(jump_dual_model)
    end

    # Get dual variables for the parameters
    load_dual_idxs =
        [map_moi_to_jump[primal_dual_map.primal_parameter[l]].value for l in load_moi_idx]
    load_var_dual = JuMP.all_variables(jump_dual_model)[load_dual_idxs]

    # Add constraints to the dual associated with the parameters
    problem_iterator.set_iterator!(jump_dual_model, load_var_dual, idx)

    # Get the objective function
    obj = objective_function(jump_dual_model)
    dual_sense = JuMP.objective_sense(jump_dual_model)

    # Inforce primal constraints
    problem_iterator.primal_builder!(jump_dual_model, load_var_dual)

    # Re-set objective function in case primal_builder! overwrote it
    @objective(jump_dual_model, dual_sense, obj)

    # Solve the dual
    JuMP.optimize!(jump_dual_model)

    # Save input
    status = recorder.filterfn(jump_dual_model)
    early_stop_bool = problem_iterator.early_stop(jump_dual_model, status, recorder)
    if early_stop_bool
        return 0, early_stop_bool
    end
    if recorder.filterfn(jump_dual_model)
        recorder.primal_variables = load_var_dual
        recorder.dual_variables = []
        record(recorder, problem_iterator.ids[idx]; input = true)
    else
        return 0, early_stop_bool
    end

    optimal_loads = value.(load_var_dual)
    optimal_dual_cost = JuMP.objective_value(jump_dual_model)

    # Create final primal model and solve
    model = JuMP.Model(problem_iterator.optimizer())
    problem_iterator.primal_builder!(model, optimal_loads; recorder = recorder)
    JuMP.optimize!(model)

    termination_status = recorder.filterfn(model)
    early_stop_bool = problem_iterator.early_stop(model, termination_status, recorder)

    # Check if method was effective
    optimal_final_cost = JuMP.objective_value(model)
    solution_primal_status = JuMP.primal_status(model)
    solution_dual_status = JuMP.dual_status(model)
    termination_status == MOI.INFEASIBLE && @warn("Optimal solution not found")
    solution_primal_status != MOI.FEASIBLE_POINT && @warn("Primal solution not found")
    solution_dual_status != MOI.FEASIBLE_POINT && @warn("Dual solution not found")

    if !isapprox(optimal_final_cost, optimal_dual_cost; rtol = 1e-4)
        rtol = abs(optimal_final_cost - optimal_dual_cost) / optimal_final_cost * 100
        @warn "Final cost is not equal to dual cost by $(rtol) %" optimal_final_cost optimal_dual_cost
    end

    if recorder.filterfn(model)
        record(recorder, problem_iterator.ids[idx])
        return 1, early_stop_bool
    end
    return 0, early_stop_bool
end
