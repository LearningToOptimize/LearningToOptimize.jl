abstract type RecorderFile end

"""
    Recorder(filename; primal_variables=[], dual_variables=[], filterfn=(model)-> termination_status(model) == MOI.OPTIMAL)

Recorder of optimization problem solutions.
"""
mutable struct Recorder{T<:RecorderFile}
    filename::String
    primal_variables::AbstractArray{Symbol}
    dual_variables::AbstractArray{Symbol}
    filterfn::Function

    function Recorder{T}(
        filename::String;
        primal_variables=[],
        dual_variables=[],
        filterfn=(model) -> termination_status(model) == MOI.OPTIMAL,
    ) where {T<:RecorderFile}
        return new{T}(filename, primal_variables, dual_variables, filterfn)
    end
end

"""
    ProblemIterator(ids::Vector{Integer}, pairs::Dict{VariableRef, Vector{Real}})

Iterator for optimization problem instances.
"""
struct ProblemIterator{T<:Real,Z<:Integer}
    ids::Vector{Z}
    pairs::Dict{VariableRef,Vector{T}}
    function ProblemIterator(
        ids::Vector{Z}, pairs::Dict{VariableRef,Vector{T}}
    ) where {T<:Real,Z<:Integer}
        for (p, val) in pairs
            @assert length(ids) == length(val)
        end
        return new{T,Z}(ids, pairs)
    end
end

"""
    update_model!(model::JuMP.Model, p::VariableRef, val::Real)

Update the value of a parameter in a JuMP model.
"""
function update_model!(model::JuMP.Model, p::VariableRef, val::T) where {T<:Real}
    return MOI.set(model, POI.ParameterValue(), p, val)
end

"""
    update_model!(model::JuMP.Model, p::VariableRef, val::AbstractArray{Real})

Update the value of a parameter in a JuMP model.
"""
function update_model!(
    model::JuMP.Model, p::VariableRef, val::AbstractArray{T}
) where {T<:Real}
    return MOI.set(model, POI.ParameterValue(), p, val)
end

"""
    update_model!(model::JuMP.Model, pairs::Dict, idx::Integer)

Update the values of parameters in a JuMP model.
"""
function update_model!(model::JuMP.Model, pairs::Dict, idx::Integer)
    for (p, val) in pairs
        update_model!(model, p, val[idx])
    end
end

"""
    solve_and_record(model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder, idx::Integer)

Solve an optimization problem and record the solution.
"""
function solve_and_record(
    model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder, idx::Integer
)
    update_model!(model, problem_iterator.pairs, idx)
    optimize!(model)
    if recorder.filterfn(model)
        record(recorder, model, problem_iterator.ids[idx])
        return 1
    end
    return 0
end

"""
    solve_batch(model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder)

Solve a batch of optimization problems and record the solutions.
"""
function solve_batch(
    model::JuMP.Model, problem_iterator::ProblemIterator, recorder::Recorder
)
    return sum(
        solve_and_record(model, problem_iterator, recorder, idx) for
        idx in 1:length(problem_iterator.ids)
    ) / length(problem_iterator.ids)
end
