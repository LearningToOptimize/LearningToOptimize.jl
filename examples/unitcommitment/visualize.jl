using Plots
using Arrow
using DataFrames
using Statistics
using LinearAlgebra

cossim(x, y) = dot(x, y) / (norm(x) * norm(y))

# Data Parameters
case_name = "case300"
date = "2017-01-01"
horizon = "2"
path_dataset = joinpath(dirname(@__FILE__), "data")
case_file_path = joinpath(path_dataset, case_name, date, "h" * horizon)

# Load input and output data tables
file_ins = readdir(joinpath(case_file_path, "input"), join = true)
batch_ids = [split(split(file, "_")[end], ".")[1] for file in file_ins]
file_outs = readdir(joinpath(case_file_path, "output"), join = true)
file_outs =
    [file_outs[findfirst(x -> occursin(batch_id, x), file_outs)] for batch_id in batch_ids]

# Load input and output data tables
input_tables = Array{DataFrame}(undef, length(file_ins))
output_tables = Array{DataFrame}(undef, length(file_outs))
for i = 1:length(file_ins)
    file = file_ins[i]
    input_tables[i] = Arrow.Table(file) |> DataFrame
end
for i = 1:length(file_outs)
    file = file_outs[i]
    output_tables[i] = Arrow.Table(file) |> DataFrame
    # if all the status are OPTIMAL, make them INFEASIBLE
    if all(output_tables[i].status .== "OPTIMAL")
        output_tables[i].status .= "INFEASIBLE"
    end
end

# Nominal Loads
load_columns = [col for col in names(input_tables[1]) if occursin("load", col)]
nominal_loads = Vector(input_tables[1][1, load_columns])

# Load divergence
cos_sim = [
    acos(cossim(nominal_loads, Vector(input_table[1, load_columns]))) for
    input_table in input_tables
]
norm_sim =
    [
        norm(nominal_loads) - norm(Vector(input_table[1, load_columns])) for
        input_table in input_tables
    ] ./ norm(nominal_loads) .+ 1

# Polar Plot divergence
plotly()
plt = plot(cos_sim, norm_sim, title = "Load Similarity", proj = :polar, m = 2)

# concatenate all the input and output tables
input_table = vcat(input_tables...)
output_table = vcat(output_tables...)

# sum each row of the input table (excluding the id) and store it in a new column
commit_columns =
    [col for col in names(input_table) if !occursin("load", col) && !occursin("id", col)]
input_table.total_commitment = sum(eachcol(input_table[!, commit_columns]))
input_table.total_load = sum(eachcol(input_table[!, load_columns]))

# join input and output tables keep on id and keep only total_commitment and objective
table_train = innerjoin(input_table, output_table[!, [:id, :objective, :status]]; on = :id)[
    !,
    [:id, :total_commitment, :objective, :status, :total_load],
]

# separate infeasible from feasible
table_train_optimal = table_train[table_train.status.=="OPTIMAL", :]
table_train_infeasible = table_train[table_train.status.=="INFEASIBLE", :]
table_train_localopt = table_train[table_train.status.=="LOCALLY_SOLVED", :]

# plot objective vs total commitment
# now with the y axis on the log scale
plotly()
plt = scatter(
    table_train_optimal.total_commitment,
    table_train_optimal.objective,
    label = "Optimal",
    xlabel = "Total Commitment",
    ylabel = "Objective",
    title = "",
    color = :red,
    yscale = :log10,
    legend = :outertopright,
    # marker size
    ms = 10,
);
scatter!(
    plt,
    table_train_localopt.total_commitment,
    table_train_localopt.objective,
    label = "Local Optimum",
    color = :blue,
    marker = :o,
    alpha = 0.5,
);
scatter!(
    plt,
    table_train_infeasible.total_commitment,
    table_train_infeasible.objective,
    label = "Infeasible",
    color = :yellow,
    marker = :square,
    alpha = 0.01,
    ms = 2,
)

# plot objective vs total load
# now with the y axis on the log scale
plt2 = scatter(
    table_train_optimal.total_load,
    table_train_optimal.objective,
    label = "Optimal",
    xlabel = "Total Load",
    ylabel = "Objective",
    title = "",
    color = :red,
    yscale = :log10,
    legend = :outertopright,
    # marker size
    ms = 10,
);
scatter!(
    plt2,
    table_train_localopt.total_load,
    table_train_localopt.objective,
    label = "Local Optimum",
    color = :blue,
    marker = :o,
    alpha = 0.5,
);
scatter!(
    plt2,
    table_train_infeasible.total_load,
    table_train_infeasible.objective,
    label = "Infeasible",
    color = :yellow,
    marker = :square,
    alpha = 0.5,
)
