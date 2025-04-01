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
    all_primal_variables(model::JuMP.Model)

Returns all primal variables of a JuMP model.
"""
function all_primal_variables(model::JuMP.Model)
    return sort(
        setdiff(all_variables(model), load_parameters(model));
        by = (v) -> index(v).value,
    )
end

"""
    Recorder{T}(
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
    ) where T<:FileType

Recorder of optimization problem solutions and, optionally, sensitivity information.

# Arguments
- `filename::String`: The name of the file to save the recorded solutions.
- `filename_input::String`: The name of the file containing the input data.
- `primal_variables::Vector`: The primal variables to record.
- `dual_variables::Vector`: The dual variables to record.
- `filterfn::Function`: A function that filters the recorded solutions.
- `model::JuMP.Model`: The model to record the solutions from.
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
mutable struct ProblemIterator{T<:Real} <: AbstractProblemIterator
    model::JuMP.Model
    ids::Vector{UUID}
    pairs::Dict{VariableRef,Vector{T}}
    pullback_primal_pairs::Dict{VariableRef,Vector{T}}
    early_stop::Function
    param_type::Type{<:AbstractParameterType}
    pre_solve_hook::Function
    function ProblemIterator(
        ids::Vector{UUID},
        pairs::Dict{VariableRef,Vector{T}},
        pullback_primal_pairs::Dict{VariableRef,Vector{T}} = Dict{VariableRef,Vector{T}}(),
        early_stop::Function = (args...) -> false,
        param_type::Type{<:AbstractParameterType} = POIParamaterType,
        pre_solve_hook::Function = (args...) -> nothing,
    ) where {T<:Real}
        model = JuMP.owner_model(first(keys(pairs)))
        for (p, val) in pairs
            @assert length(ids) == length(val)
        end
        if !isempty(pullback_primal_pairs)
            for (p, val) in pullback_primal_pairs
                @assert length(ids) == length(val)
            end
        end
        return new{T}(model, ids, pairs, pullback_primal_pairs, early_stop, param_type, pre_solve_hook)
    end
end

function ProblemIterator(
    pairs::Dict{VariableRef,Vector{T}};
    pullback_primal_pairs::Dict{VariableRef,Vector{T}} = Dict{VariableRef,Vector{T}}(),
    early_stop::Function = (args...) -> false,
    pre_solve_hook::Function = (args...) -> nothing,
    param_type::Type{<:AbstractParameterType} = POIParamaterType,
    ids = [uuid1() for _ = 1:length(first(values(pairs)))],
) where {T<:Real}
    return ProblemIterator(ids, pairs, pullback_primal_pairs, early_stop, param_type, pre_solve_hook)
end

"""
    save(problem_iterator::ProblemIterator, filename::AbstractString, file_type::Type{T})

Save optimization problem instances to a file.
"""
function save(
    problem_iterator::AbstractProblemIterator,
    filename::AbstractString,
    file_type::Type{T};
    pullback_filename::AbstractString = filename * "_pullback_",
) where {T<:FileType}
    kys = sort(collect(keys(problem_iterator.pairs)); by = (v) -> index(v).value)
    df = (; id = problem_iterator.ids,)
    df = merge(df, (; zip(Symbol.(kys), [problem_iterator.pairs[ky] for ky in kys])...))
    save(df, filename, file_type)
    if !isempty(problem_iterator.pullback_primal_pairs)
        kys = sort(collect(keys(problem_iterator.pullback_primal_pairs)); by = (v) -> index(v).value)
        df = (; id = problem_iterator.ids,)
        df = merge(df, (; zip(Symbol.(kys), [problem_iterator.pullback_primal_pairs[ky] for ky in kys])...))
        save(df, pullback_filename, file_type)
    end
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
                @error("Variable $ky not found in model")
                return nothing
            end
            parameter = parameters[idx]
            push!(pairs, parameter => df[!, ky])
        end
    end
    return pairs
end

function _dataframe_to_dict(df::DataFrame, model_file::AbstractString; use_nlp_block=false)
    # Load model
    model = read_from_file(model_file; use_nlp_block = use_nlp_block)
    # Retrieve parameters
    parameters = LearningToOptimize.load_parameters(model)
    return _dataframe_to_dict(df, parameters)
end

function _dataframe_to_dict(df::DataFrame, df_pullback::DataFrame, model_file::AbstractString; use_nlp_block=false)
    # Load model
    model = read_from_file(model_file; use_nlp_block = use_nlp_block)
    # Retrieve parameters
    parameters = LearningToOptimize.load_parameters(model)
    primal_variables = all_primal_variables(model)
    pairs = _dataframe_to_dict(df, parameters)
    pullback_primal_pairs = _dataframe_to_dict(df_pullback, primal_variables)
    return pairs, pullback_primal_pairs
end

function random_pullback_primal_pairs(
    problem_iterator::ProblemIterator{T},
    num_pullbacks::Integer,
) where {T<:Real}
    pullback_primal_pairs = Dict{VariableRef,Vector{T}}()
    primal_variables = all_primal_variables(problem_iterator.model)
    for x in primal_variables
        pullback_primal_pairs[x] = randn(T, num_pullbacks)
    end
    return pullback_primal_pairs
end

function load(
    model_file::AbstractString,
    input_file::AbstractString,
    ::Type{T};
    input_pullback_file::Union{Nothing,AbstractString} = nothing,
    batch_size::Union{Nothing,Integer} = nothing,
    ignore_ids::Vector{UUID} = UUID[],
    param_type::Type{<:AbstractParameterType} = JuMPParameterType,
    pullback_generator::Function = (args...) -> Dict{VariableRef,Vector{T}}(),
    use_nlp_block = false
) where {T<:FileType}
    # Load full set
    df = load(input_file, T)
    df.id = UUID.(df.id)
    # Remove ignored ids
    if !isempty(ignore_ids)
        df = filter(:id => (id) -> !(id in ignore_ids), df)
        if isempty(df)
            @warn("All ids are ignored")
            return nothing
        end
    end
    ids = df.id
    # Load pullback set
    df_pullback = if !isnothing(input_pullback_file)
        _df = load(input_pullback_file, T)
        _df.id = UUID.(_df.id)
        if !isempty(ignore_ids)
            _df = filter(:id => (id) -> !(id in ignore_ids), _df)
        end
        _df
    else
        DataFrame()
    end
    # No batch
    if isnothing(batch_size)
        if isempty(df_pullback)
            pairs = _dataframe_to_dict(df, model_file; use_nlp_block = use_nlp_block)
            problem_iterator = ProblemIterator(pairs; ids = ids, param_type = param_type)
            problem_iterator.pullback_primal_pairs = pullback_generator(problem_iterator, length(ids))
            return problem_iterator
        else
            pairs, pullback_primal_pairs = _dataframe_to_dict(df, df_pullback, model_file)
            problem_iterator = ProblemIterator(pairs; ids = ids, param_type = param_type)
            problem_iterator.pullback_primal_pairs = pullback_primal_pairs
            return problem_iterator
        end
    end
    # Batch
    num_batches = ceil(Int, length(ids) / batch_size)
    idx_range = (i) -> (i-1)*batch_size+1:min(i * batch_size, length(ids))
    return (i) -> begin
        if isempty(df_pullback)
            problem_iterator = ProblemIterator(
                _dataframe_to_dict(df[idx_range(i), :], model_file);
                ids = ids[idx_range(i)],
                param_type = param_type,
            )
            problem_iterator.pullback_primal_pairs = pullback_generator(problem_iterator, length(ids))
            return problem_iterator
        else
            pairs, pullback_primal_pairs = _dataframe_to_dict(
                df[idx_range(i), :],
                df_pullback[idx_range(i), :],
                model_file,
            )
            problem_iterator = ProblemIterator(pairs; ids = ids[idx_range(i)], param_type = param_type)
            problem_iterator.pullback_primal_pairs = pullback_primal_pairs
            return problem_iterator
        end
    end,num_batches
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

function update_diffopt_model!(
    model::JuMP.Model,
    pairs::Dict,
    idx::Integer,
)
    for (x, val) in pairs
        MOI.set(model, DiffOpt.ReverseVariablePrimal(), x, val[idx])
    end
    if !isempty(pairs)
        DiffOpt.reverse_differentiate!(model)
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
    sensitivity = !isempty(problem_iterator.pullback_primal_pairs)
    if status
        if in(termination_status(model), ACCEPTED_TERMINATION_STATUSES)
            update_diffopt_model!(model, problem_iterator.pullback_primal_pairs, idx)
        end
        record(recorder, problem_iterator.ids[idx]; sensitivity = sensitivity)
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
