using L2OCore
using Test
using Random

include("problems.jl")

@testset "L2OCore" begin
    @testset "problem classes" include("test_problem.jl")
    @testset "evaluators" include("test_evaluators.jl")
    @testset "guarantees" include("test_guarantees.jl")
    @testset "pipeline" include("test_pipeline.jl")
    @testset "code quality" include("test_quality.jl")
end
