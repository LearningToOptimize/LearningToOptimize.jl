abstract type FileType end

mutable struct RecorderFile{T<:FileType}
    filename::String
end

filename(recorder_file::RecorderFile) = recorder_file.filename

const ACCEPTED_TERMINATION_STATUSES = [
    MOI.OPTIMAL,
    MOI.SLOW_PROGRESS,
    MOI.LOCALLY_SOLVED,
    MOI.ITERATION_LIMIT,
    MOI.ALMOST_OPTIMAL,
]

DECISION_STATUS = [MOI.FEASIBLE_POINT, MOI.NEARLY_FEASIBLE_POINT]

termination_status_filter(status) = in(status, ACCEPTED_TERMINATION_STATUSES)
primal_status_filter(status) = in(status, DECISION_STATUS)
dual_status_filter(status) = in(status, DECISION_STATUS)

function filter_fn(model; check_primal = true, check_dual = true)
    if !termination_status_filter(termination_status(model))
        return false
    elseif check_primal && !primal_status_filter(primal_status(model))
        return false
    elseif check_dual && !dual_status_filter(dual_status(model))
        return false
    end
    return true
end

"""
    Recorder(filename; primal_variables=[], dual_variables=[], filterfn=(model)-> termination_status(model) == MOI.OPTIMAL)

Recorder of optimization problem solutions.
"""
mutable struct Recorder{T<:FileType}
    model::JuMP.Model
    recorder_file::RecorderFile{T}
    recorder_file_input::RecorderFile{T}
    primal_variables::Vector
    dual_variables::Vector
    filterfn::Function

    function Recorder{T}(
        filename::String;
        filename_input::String = filename * "_input_",
        primal_variables = [],
        dual_variables = [],
        filterfn = filter_fn,
        model = if length(primal_variables) > 0
            owner_model(primal_variables[1])
        elseif length(dual_variables) > 0
            owner_model(dual_variables[1])
        else
            @error("No model provided")
        end,
    ) where {T<:FileType}
        return new{T}(
            model,
            RecorderFile{T}(filename),
            RecorderFile{T}(filename_input),
            primal_variables,
            dual_variables,
            filterfn,
        )
    end
end

filename(recorder::Recorder) = filename(recorder.recorder_file)
filename_input(recorder::Recorder) = filename(recorder.recorder_file_input)
get_primal_variables(recorder::Recorder) = recorder.primal_variables
get_dual_variables(recorder::Recorder) = recorder.dual_variables
get_filterfn(recorder::Recorder) = recorder.filterfn

function similar(recorder::Recorder{T}) where {T<:FileType}
    return Recorder{T}(
        filename(recorder);
        filename_input = filename_input(recorder),
        primal_variables = get_primal_variables(recorder),
        dual_variables = get_dual_variables(recorder),
        filterfn = get_filterfn(recorder),
    )
end

function set_primal_variable!(recorder::Recorder, p::Vector)
    return recorder.primal_variables = p
end

function set_dual_variable!(recorder::Recorder, p::Vector)
    return recorder.dual_variables = p
end

function set_model!(recorder::Recorder)
    recorder.model = if length(recorder.primal_variables) > 0
        owner_model(recorder.primal_variables[1])
    elseif length(recorder.dual_variables) > 0
        owner_model(recorder.dual_variables[1])
    else
        @error("No model provided")
    end
end

abstract type AbstractProblemIterator end

abstract type AbstractParameterType end

abstract type POIParamaterType <: AbstractParameterType end

abstract type JuMPNLPParameterType <: AbstractParameterType end

abstract type JuMPParameterType <: AbstractParameterType end

"""
    ProblemIterator(ids::Vector{UUID}, pairs::Dict{VariableRef, Vector{Real}})

Iterator for optimization problem instances.
"""
struct ProblemIterator{T<:Real} <: AbstractProblemIterator
    model::JuMP.Model
    ids::Vector{UUID}
    pairs::Dict{VariableRef,Vector{T}}
    early_stop::Function
    param_type::Type{<:AbstractParameterType}
    pre_solve_hook::Function
    function ProblemIterator(
        ids::Vector{UUID},
        pairs::Dict{VariableRef,Vector{T}},
        early_stop::Function = (args...) -> false,
        param_type::Type{<:AbstractParameterType} = POIParamaterType,
        pre_solve_hook::Function = (args...) -> nothing,
    ) where {T<:Real}
        model = JuMP.owner_model(first(keys(pairs)))
        for (p, val) in pairs
            @assert length(ids) == length(val)
        end
        return new{T}(model, ids, pairs, early_stop, param_type, pre_solve_hook)
    end
end

function ProblemIterator(
    pairs::Dict{VariableRef,Vector{T}};
    early_stop::Function = (args...) -> false,
    pre_solve_hook::Function = (args...) -> nothing,
    param_type::Type{<:AbstractParameterType} = POIParamaterType,
    ids = [uuid1() for _ = 1:length(first(values(pairs)))],
) where {T<:Real}
    return ProblemIterator(ids, pairs, early_stop, param_type, pre_solve_hook)
end

"""
    save(problem_iterator::ProblemIterator, filename::AbstractString, file_type::Type{T})

Save optimization problem instances to a file.
"""
function save(
    problem_iterator::AbstractProblemIterator,
    filename::AbstractString,
    file_type::Type{T},
) where {T<:FileType}
    kys = sort(collect(keys(problem_iterator.pairs)); by = (v) -> index(v).value)
    df = (; id = problem_iterator.ids,)
    df = merge(df, (; zip(Symbol.(kys), [problem_iterator.pairs[ky] for ky in kys])...))
    save(df, filename, file_type)
    return nothing
end

function _dataframe_to_dict(df::DataFrame, parameters::Vector{VariableRef})
    pairs = Dict{VariableRef,Vector{Float64}}()
    for ky in names(df)
        if ky != "id"
            idx = findfirst(parameters) do p
                name(p) == string(ky)
            end
            if isnothing(idx)
                @error("Parameter $ky not found in model")
                return nothing
            end
            parameter = parameters[idx]
            push!(pairs, parameter => df[!, ky])
        end
    end
    return pairs
end

function _dataframe_to_dict(df::DataFrame, model_file::AbstractString)
    # Load model
    model = read_from_file(model_file)
    # Retrieve parameters
    parameters = LearningToOptimize.load_parameters(model)
    return _dataframe_to_dict(df, parameters)
end

function load(
    model_file::AbstractString,
    input_file::AbstractString,
    ::Type{T};
    batch_size::Union{Nothing,Integer} = nothing,
    ignore_ids::Vector{UUID} = UUID[],
    param_type::Type{<:AbstractParameterType} = JuMPParameterType,
) where {T<:FileType}
    # Load full set
    df = load(input_file, T)
    # Remove ignored ids
    df.id = UUID.(df.id)
    if !isempty(ignore_ids)
        df = filter(:id => (id) -> !(id in ignore_ids), df)
        if isempty(df)
            @warn("All ids are ignored")
            return nothing
        end
    end
    ids = df.id
    # No batch
    if isnothing(batch_size)
        pairs = _dataframe_to_dict(df, model_file)
        return ProblemIterator(pairs; ids = ids, param_type = param_type)
    end
    # Batch
    num_batches = ceil(Int, length(ids) / batch_size)
    idx_range = (i) -> ((i-1)*batch_size+1):min(i*batch_size, length(ids))
    return (i) -> ProblemIterator(
        _dataframe_to_dict(df[idx_range(i), :], model_file);
        ids = ids[idx_range(i)],
        param_type = param_type,
    ),
    num_batches
end

"""
    update_model!(model::JuMP.Model, p::VariableRef, val::Real)

Update the value of a parameter in a JuMP model.
"""
function update_model!(::Type{POIParamaterType}, model::JuMP.Model, p::VariableRef, val)
    return MOI.set(model, POI.ParameterValue(), p, val)
end

function update_model!(::Type{JuMPNLPParameterType}, model::JuMP.Model, p::VariableRef, val)
    return set_parameter_value(p, val)
end

function update_model!(::Type{JuMPParameterType}, model::JuMP.Model, p::VariableRef, val)
    return fix(p, val)
end

"""
    update_model!(model::JuMP.Model, pairs::Dict, idx::Integer)

Update the values of parameters in a JuMP model.
"""
function update_model!(
    model::JuMP.Model,
    pairs::Dict,
    idx::Integer,
    param_type::Type{<:AbstractParameterType},
)
    for (p, val) in pairs
        update_model!(param_type, model, p, val[idx])
    end
end

"""
    solve_and_record(problem_iterator::ProblemIterator, recorder::Recorder, idx::Integer)

Solve an optimization problem and record the solution.
"""
function solve_and_record(
    problem_iterator::ProblemIterator,
    recorder::Recorder,
    idx::Integer,
)
    model = problem_iterator.model
    problem_iterator.pre_solve_hook(model)
    update_model!(model, problem_iterator.pairs, idx, problem_iterator.param_type)
    optimize!(model)
    status = recorder.filterfn(model)
    early_stop_bool = problem_iterator.early_stop(model, status, recorder)
    if status
        record(recorder, problem_iterator.ids[idx])
        return 1, early_stop_bool
    end
    return 0, early_stop_bool
end

"""
    solve_batch(problem_iterator::AbstractProblemIterator, recorder)

Solve a batch of optimization problems and record the solutions.
"""
function solve_batch(problem_iterator::AbstractProblemIterator, recorder)
    successfull_solves = 0.0
    for idx = 1:length(problem_iterator.ids)
        _success_bool, early_stop_bool = solve_and_record(problem_iterator, recorder, idx)
        if _success_bool == 1
            successfull_solves += 1
        end
        if early_stop_bool
            break
        end
    end
    successfull_solves = successfull_solves / length(problem_iterator.ids)

    @info "Recorded $(successfull_solves * 100) % of $(length(problem_iterator.ids)) problems"
    return successfull_solves
end
