using Plots
using Arrow
using DataFrames

# Data Parameters
case_name = "case300"
date = "2017-01-01"
horizon = "2"
path_dataset = joinpath(dirname(@__FILE__), "data")
case_file_path = joinpath(path_dataset, case_name, date, "h"*horizon)

# Load input and output data tables
file_ins = readdir(joinpath(case_file_path, "input"), join=true)
file_outs = readdir(joinpath(case_file_path, "output"), join=true)

# Load input and output data tables
input_tables = Array{DataFrame}(undef, length(file_ins))
output_tables = Array{DataFrame}(undef, length(file_outs))
for (i, file) in enumerate(file_ins)
    input_tables[i] = Arrow.Table(file) |> DataFrame
end
for (i, file) in enumerate(file_outs)
    output_tables[i] = Arrow.Table(file) |> DataFrame
    # if all the status are OPTIMAL, make them INFEASIBLE
    if all(output_tables[i].status .== "OPTIMAL")
        output_tables[i].status .= "INFEASIBLE"
    end
end

# concatenate all the input and output tables
input_table = vcat(input_tables...)
output_table = vcat(output_tables...)

# sum each row of the input table (excluding the id) and store it in a new column
commit_columns = [col for col in names(input_table) if !occursin("load", col) && !occursin("id", col)]
load_columns = [col for col in names(input_table) if occursin("load", col)]
input_table.total_commitment = sum(eachcol(input_table[!, commit_columns]))
input_table.total_load = sum(eachcol(input_table[!, load_columns]))

# join input and output tables keep on id and keep only total_commitment and objective
table_train = innerjoin(input_table, output_table[!, [:id, :objective, :status]]; on=:id)[!, [:id, :total_commitment, :objective, :status, :total_load]]

# separate infeasible from feasible
table_train_optimal = table_train[table_train.status .== "OPTIMAL", :]
table_train_infeasible = table_train[table_train.status .== "INFEASIBLE", :]
table_train_localopt = table_train[table_train.status .== "LOCALLY_SOLVED", :]

# plot objective vs total commitment
# now with the y axis on the log scale
plotly()
plt = scatter(
    table_train_optimal.total_commitment, 
    table_train_optimal.objective, 
    label="Optimal", xlabel="Total Commitment", ylabel="Objective", 
    title="", color=:red, yscale=:log10, legend=:outertopright,
    # marker size
    ms=10,
);
scatter!(plt, 
    table_train_localopt.total_commitment, 
    table_train_localopt.objective, 
    label="Local Optimum", color=:blue,
    marker=:o, alpha=0.5
);
scatter!(plt, 
    table_train_infeasible.total_commitment, 
    table_train_infeasible.objective, 
    label="Infeasible", color=:yellow,
    marker=:square, alpha=0.01, ms=2
)

# plot objective vs total load
# now with the y axis on the log scale
plt2 = scatter(
    table_train_optimal.total_load, 
    table_train_optimal.objective, 
    label="Optimal", xlabel="Total Load", ylabel="Objective", 
    title="", color=:red, yscale=:log10, legend=:outertopright,
    # marker size
    ms=10,
);
scatter!(plt2, 
    table_train_localopt.total_load, 
    table_train_localopt.objective, 
    label="Local Optimum", color=:blue,
    marker=:o, alpha=0.5
);
scatter!(plt2, 
    table_train_infeasible.total_load, 
    table_train_infeasible.objective, 
    label="Infeasible", color=:yellow,
    marker=:square, alpha=0.5
)
