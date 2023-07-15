using L2O
using Arrow
using Test
using DelimitedFiles
using JuMP, HiGHS
import ParametricOptInterface as POI

include(
    joinpath(
        dirname(dirname(@__FILE__)), "examples", "powermodels", "pg_lib.jl"
    ),
)

"""
    testdataset_gen(path::String)

Test dataset generation for different filetypes
"""
function testdataset_gen(path::String)
    @testset "Type: $filetype" for filetype in [CSVFile, ArrowFile]
        # The problem to iterate over
        model = Model(() -> POI.Optimizer(HiGHS.Optimizer()))
        @variable(model, x)
        p = @variable(model, _p in POI.Parameter(1.0))
        @constraint(model, cons, x + _p >= 3)
        @objective(model, Min, 2x)

        # The problem iterator
        num_p = 10
        @test_throws AssertionError ProblemIterator(collect(1:num_p), Dict(p => collect(1.0:3.0)))
        @test_throws MethodError ProblemIterator(collect(1.0:3.0), Dict(p => collect(1.0:3.0)))
        problem_iterator = ProblemIterator(collect(1:num_p), Dict(p => collect(1.0:num_p)))

        # The recorder
        file = joinpath(path, "test.$(string(filetype))") # file path
        @test Recorder{filetype}(file; primal_variables=[:x]) isa Recorder{filetype}
        @test Recorder{filetype}(file; dual_variables=[:cons]) isa Recorder{filetype}
        recorder = Recorder{filetype}(file; primal_variables=[:x], dual_variables=[:cons])

        # Solve all problems and record solutions
        solve_batch(model, problem_iterator, recorder)

        # Check if file exists and has the correct number of rows and columns
        if filetype == CSVFile
            file1 = joinpath(path, "test.csv")
            @test isfile(file1)
            @test length(readdlm(file1, ',')[:, 1]) == num_p + 1
            @test length(readdlm(file1, ',')[1, :]) == 3
            rm(file1)
        else
            file2 = joinpath(path, "test.arrow")
            @test isfile(file2)
            df = Arrow.Table(file2)
            @test length(df) == 3
            @test length(df[1]) == num_p
        end
    end
end

@testset "L2O.jl" begin
    @testset "Dataset Generation" begin
        mktempdir() do path
            # Different filetypes
            testdataset_gen(path)
            # Pglib
            @testset "pg_lib case" begin
                # Define test case from pglib
                case_name = "pglib_opf_case5_pjm.m"

                # Define number of problems
                num_p = 10

                # Generate dataset
                success_solves, number_generators = generate_dataset_pglib(
                    path, case_name; num_p=num_p
                )

                # Check if the number of successfull solves is equal to the number of problems saved
                file = joinpath(path, "test.csv")
                @test isfile(file)
                @test length(readdlm(file, ',')[:, 1]) == num_p * success_solves + 1
                @test length(readdlm(file, ',')[1, :]) == number_generators + 1
                rm(file)
            end
        end
    end
end
