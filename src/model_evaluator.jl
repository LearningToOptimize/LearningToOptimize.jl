"""
    model_outputs(model::Function, problem_iterator::ProblemIterator)

Returns the output of a machine learning model over a ProblemIterator dataset.
"""
function model_outputs(model::Function, problem_iterator::ProblemIterator)
    kys = sort(collect(keys(problem_iterator.pairs)); by = (v) -> index(v).value)
    input = hcat([problem_iterator.pairs[ky] for ky in kys]...)
    return model(input)
end

"""
    model_outputs(mach::T, problem_iterator::ProblemIterator)

Returns the output of a machine learning model over a ProblemIterator dataset.
"""
function model_outputs(mach::T, problem_iterator::ProblemIterator) where {T<:Machine}
    return predict(
        mach,
        DataFrame(Symbol.(keys(problem_iterator.pairs)) .=> values(problem_iterator.pairs)),
    )
end

""" 
    variable_point(vals::Array{T,2}, problem_iterator::ProblemIterator, idx::Int)

Returns a dictionary with the values of the variables and parameters for the idx-th problem instance.
"""
function variable_point(
    vals::Array{T,2},
    problem_iterator::ProblemIterator,
    idx::Int,
) where {T<:Real}
    # primal variables 
    variables = all_primal_variables(problem_iterator.model)
    dict = Dict{VariableRef,Float64}(variables .=> vals[:, idx])
    # parameters
    for (v, val) in problem_iterator.pairs
        dict[v] = val[idx]
    end
    return dict
end

""" 
    variable_point(vals::Tables.MatrixTable, problem_iterator::ProblemIterator, idx::Int)

Returns a dictionary with the values of the variables and parameters for the idx-th problem instance.
"""
function variable_point(
    vals::Tables.MatrixTable,
    problem_iterator::ProblemIterator,
    idx::Int,
)
    # primal variables
    variables = all_primal_variables(problem_iterator.model)
    dict = Dict{VariableRef,Float64}(
        variables .=> [vals[Symbol(name(v))][idx] for v in variables],
    )
    # parameters
    for (v, val) in problem_iterator.pairs
        dict[v] = val[idx]
    end
    return dict
end

function JuMP.MOI.Utilities.distance_to_set(
    ::MOI.Utilities.ProjectionUpperBoundDistance,
    val::Float64,
    set::MOI.Parameter,
)
    return set.value - val
end


"""
    feasibility_evaluator(problem_iterator::ProblemIterator, output)

Feasibility evaluator of a solution over a ProblemIterator dataset.
PS.: Probably can be made much more efficient.
"""
function feasibility_evaluator(problem_iterator::ProblemIterator, output)
    average_infeasilibity = Array{Float64}(undef, length(problem_iterator.ids))
    for idx = 1:length(problem_iterator.ids)
        update_model!(
            problem_iterator.model,
            problem_iterator.pairs,
            idx,
            problem_iterator.param_type,
        )
        dct = primal_feasibility_report(
            problem_iterator.model,
            variable_point(output, problem_iterator, idx),
        )
        if isempty(dct)
            average_infeasilibity[idx] = 0.0
        else
            average_infeasilibity[idx] = sum(values(dct)) / length(dct)
        end
    end
    return average_infeasilibity
end
    
"""
    _objective_value(model::JuMP.Model, point::Dict{VariableRef,Float64})

Objective value of a solution over a JuMP model.
"""
function _objective_value(model::JuMP.Model, point::Dict{VariableRef,Float64})
    obj_function = JuMP.objective_function(model)
    return value((v) -> point[v], obj_function)
end

"""
    objective_evaluator(problem_iterator::ProblemIterator, output)

Objective evaluator of a solution over a ProblemIterator dataset.
PS.: Probably can be made much more efficient.
"""
function objective_evaluator(problem_iterator::ProblemIterator, output)
    average_objective = Array{Float64}(undef, length(problem_iterator.ids))
    for idx = 1:length(problem_iterator.ids)
        update_model!(
            problem_iterator.model,
            problem_iterator.pairs,
            idx,
            problem_iterator.param_type,
        )
        average_objective[idx] = _objective_value(
            problem_iterator.model,
            variable_point(output, problem_iterator, idx),
        )
    end
    return average_objective
end

"""
    general_evaluator(problem_iterator::ProblemIterator, model)

General evaluator of a predictor model over a ProblemIterator dataset. Returns objective, feasibility, 
inference time and allocated memory.
"""
function general_evaluator(problem_iterator::ProblemIterator, model)
    timed_output = @timed model_outputs(model, problem_iterator)
    feasibility = feasibility_evaluator(problem_iterator, timed_output.value)
    objective = objective_evaluator(problem_iterator, timed_output.value)
    return (;
        objective = objective,
        infeasibility = feasibility,
        time = timed_output.time,
        bytes = timed_output.bytes,
    )
end
