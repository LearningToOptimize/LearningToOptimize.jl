module LearningToOptimize

using Arrow
using CSV
using DataFrames
using JuMP
using UUIDs
import ParametricOptInterface as POI
import JuMP.MOI as MOI
import JuMP.MOI.Utilities: distance_to_set
import Base: string
using Statistics
using Distributions

using Zygote

using MLJ
using MLJFlux
using Flux
using Flux: @functor
using Random
import MLJFlux.train!
using Optimisers

using LinearAlgebra

export ArrowFile,
    CSVFile,
    ProblemIterator,
    Recorder,
    save,
    load,
    solve_batch,
    set_primal_variable!,
    set_dual_variable!,
    set_model!,
    FullyConnected,
    FullyConnectedBuilder,
    make_convex!,
    make_convex,
    ConvexRule,
    relative_rmse,
    relative_mae,
    inconvexhull,
    line_sampler,
    box_sampler,
    scaled_distribution_sampler,
    general_sampler,
    compress_batch_arrow,
    feasibility_evaluator,
    objective_evaluator,
    general_evaluator,
    sobolev_pb_loss_generator

include("datasetgen.jl")
include("csvrecorder.jl")
include("arrowrecorder.jl")
include("FullyConnected.jl")
include("nn_expression.jl")
include("metrics.jl")
include("inconvexhull.jl")
include("samplers.jl")
include("model_evaluator.jl")

end
