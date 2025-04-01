abstract type ArrowFile <: FileType end

Base.string(::Type{ArrowFile}) = "arrow"

"""
    record(recorder::Recorder{ArrowFile}, id::UUID)

Record optimization problem solution to an Arrow file.
"""
function record(recorder::Recorder{ArrowFile}, id::UUID; input = false, sensitivity = false)
    _filename = input ? filename_input(recorder) : filename(recorder)
    _filename = _filename * "_$(string(id))." * string(ArrowFile)
    model = recorder.model

    status = JuMP.termination_status(model)
    primal_stat = JuMP.primal_status(model)
    dual_stat = JuMP.dual_status(model)
    if sensitivity
        parameters = load_parameters(model)
    else
        parameters = []
    end

    primal_values = if in(primal_stat, DECISION_STATUS)
        [[value.(p)] for p in recorder.primal_variables]
    else
        [[zeros(length(p))] for p in recorder.primal_variables]
    end

    dual_values = if in(dual_stat, DECISION_STATUS)
        [[dual.(p)] for p in recorder.dual_variables]
    else
        [[zeros(length(p))] for p in recorder.dual_variables]
    end

    objective = if in(status, ACCEPTED_TERMINATION_STATUSES)
        JuMP.objective_value(model)
    else
        0.0
    end

    parameter_values = if sensitivity && in(status, ACCEPTED_TERMINATION_STATUSES)
        [[MOI.get(model, DiffOpt.ReverseConstraintSet(), ParameterRef(p)).value] for p in parameters]
    else
        []
    end

    df = (;
        id = [id],
        zip(Symbol.(name.(recorder.primal_variables)), primal_values)...,
        zip(Symbol.("dual_" .* name.(recorder.dual_variables)), dual_values)...,
        zip(Symbol.("pb_" .* name.(parameters)), parameter_values)...,
    )
    if !input
        df = merge(
            df,
            (;
                objective = [objective],
                time = [JuMP.solve_time(model)],
                status = [string(status)],
                primal_status = [string(primal_stat)],
                dual_status = [string(dual_stat)],
            ),
        )
    end

    return Arrow.write(_filename, df)
end

function save(table::NamedTuple, filename::String, ::Type{ArrowFile})
    filename = filename * "." * string(ArrowFile)
    return Arrow.write(filename, table)
end

function load(filename::String, ::Type{ArrowFile})
    if !occursin(string(ArrowFile), filename)
        return DataFrame(Arrow.Table(filename * "." * string(ArrowFile)))
    else
        return DataFrame(Arrow.Table(filename))
    end
end

function compress_batch_arrow(
    case_file_path::String,
    case_name::String;
    keyword_all = "output",
    batch_id::String = string(uuid1()),
    keyword_any = ["_"],
)
    iter_files = readdir(case_file_path; join = true)
    file_outs = [
        file for file in iter_files if occursin(case_name, file) &&
        occursin("arrow", file) &&
        occursin(keyword_all, file) &&
        any(x -> occursin(x, file), keyword_any)
    ]
    output_table = Arrow.Table(file_outs)
    Arrow.write(
        joinpath(case_file_path, "$(case_name)_$(keyword_all)_$(batch_id).arrow"),
        output_table,
    )
    for file in file_outs
        rm(file)
    end
end
