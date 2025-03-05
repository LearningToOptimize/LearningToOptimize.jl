using Arrow
using DelimitedFiles
using Flux
using HiGHS
using JuMP
using LearningToOptimize
import ParametricOptInterface as POI
using Test
using UUIDs
using Ipopt
using MLJFlux
using Flux
using MLJ
using CSV
using DataFrames
using Optimisers
using Statistics
using Random
using Zygote

const test_dir = dirname(@__FILE__)
const examples_dir = joinpath(test_dir, "..", "examples")

include(joinpath(test_dir, "datasetgen.jl"))

include(joinpath(examples_dir, "powermodels", "pglib_datagen.jl"))

include(joinpath(test_dir, "test_flux_forecaster.jl"))

include(joinpath(test_dir, "nn_expression.jl"))

include(joinpath(test_dir, "inconvexhull.jl"))

include(joinpath(test_dir, "samplers.jl"))

include(joinpath(test_dir, "metrics.jl"))

@testset "LearningToOptimize.jl" begin
    test_sobolev_pb_loss()
    test_load_parameters_model()
    test_load_parameters()
    test_line_sampler()
    test_box_sampler()
    test_general_sampler()
    test_fully_connected()
    test_flux_jump_basic()
    test_inconvexhull()

    mktempdir() do path
        test_compress_batch_arrow(path)
        model_file = "pglib_opf_case5_pjm_DCPPowerModel_POI_load.mof.json"
        @testset "Samplers saving on $filetype" for filetype in [ArrowFile, CSVFile]
            file_in, ids =
                test_general_sampler_file(model_file; cache_dir = path, filetype = filetype)
            test_load(model_file, file_in, filetype, ids)
        end
        test_problem_iterator(path)
        file_in, file_out, problem_iterator =
            test_pglib_datasetgen(path, "pglib_opf_case5_pjm", 20)
        test_flux_forecaster(problem_iterator, file_in, file_out)
    end
end
