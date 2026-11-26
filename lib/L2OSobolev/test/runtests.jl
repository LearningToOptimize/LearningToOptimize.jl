using L2OSobolev
using L2OCore
using Test
using Random

include("problems.jl")

@testset "L2OSobolev" begin
    @testset "masks" include("test_masks.jl")
    @testset "backends" include("test_backends.jl")
    @testset "derivative labels" include("test_labels.jl")
    @testset "wrapper composes with methods" include("test_wrapper.jl")
    @testset "nested AD and training" include("test_training.jl")
    @testset "approximation theorems" include("test_theory.jl")
    @testset "structured masks" include("test_structured_masks.jl")
    @testset "evaluation metrics" include("test_metrics.jl")
    @testset "code quality" include("test_quality.jl")
end
