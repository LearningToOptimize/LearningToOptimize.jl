
"""
    test_problem_iterator(path::AbstractString)

Test dataset generation for different filetypes
"""
function test_problem_iterator(path::AbstractString)
    @testset "Dataset Generation (POI) Type: $filetype" for filetype in [CSVFile, ArrowFile]
        # The problem to iterate over
        model = JuMP.Model(() -> POI.Optimizer(HiGHS.Optimizer()))
        @variable(model, x)
        p = @variable(model, _p in MOI.Parameter(1.0))
        @constraint(model, cons, x + _p >= 3)
        @objective(model, Min, 2x)

        # The problem iterator
        num_p = 10
        batch_id = string(uuid1())
        @testset "Problem Iterator Builder" begin
            @test_throws AssertionError ProblemIterator(
                [uuid1() for _ in 1:num_p], Dict(p => collect(1.0:3.0))
            )
            @test_throws MethodError ProblemIterator(
                collect(1.0:3.0), Dict(p => collect(1.0:3.0))
            )
            problem_iterator = ProblemIterator(Dict(p => collect(1.0:num_p)))
            file_input = joinpath(path, "test_$(batch_id)_input") # file path
            save(problem_iterator, file_input, filetype)
            file_input = file_input * ".$(string(filetype))"
            @test isfile(file_input)
        end
        problem_iterator = ProblemIterator(Dict(p => collect(1.0:num_p)))
        file_input = joinpath(path, "test_$(batch_id)_input") # file path

        # The recorder
        file_output = joinpath(path, "test_$(batch_id)_output") # file path
        @testset "Recorder Builder" begin
            @test Recorder{filetype}(file_output; primal_variables=[x]) isa
                Recorder{filetype}
            @test Recorder{filetype}(file_output; dual_variables=[cons]) isa
                Recorder{filetype}
        end
        recorder = Recorder{filetype}(
            file_output; primal_variables=[x], dual_variables=[cons]
        )

        # Solve all problems and record solutions
        @testset "early_stop" begin
            file_dual_output = joinpath(path, "test_$(string(uuid1()))_output") # file path
            recorder_dual = Recorder{filetype}(file_dual_output; dual_variables=[cons])
            problem_iterator = ProblemIterator(
                Dict(p => collect(1.0:num_p)); early_stop=(args...) -> true
            )
            successfull_solves = solve_batch(problem_iterator, recorder_dual)
            @test num_p * successfull_solves == 1
        end

        @testset "solve_batch" begin
            successfull_solves = solve_batch(problem_iterator, recorder)

            # Check if file exists and has the correct number of rows and columns
            if filetype == CSVFile
                # test input file
                file_input = file_input * ".$(string(filetype))"
                @test length(readdlm(file_input, ',')[:, 1]) == num_p + 1 # 1 from header
                @test length(readdlm(file_input, ',')[1, :]) == 2 # 2 parameter
                rm(file_input)
                # test output file
                file_output = file_output * ".$(string(filetype))"
                @test isfile(file_output)
                @test length(readdlm(file_output, ',')[:, 1]) ==
                    num_p * successfull_solves + 1 # 1 from header
                @test length(readdlm(file_output, ',')[1, :]) == 8
                rm(file_output)
            else
                iter_files = readdir(joinpath(path))
                iter_files = filter(x -> occursin(string(filetype), x), iter_files)
                file_outs = [
                    joinpath(path, file) for
                    file in iter_files if occursin("$(batch_id)_output", file)
                ]
                file_ins = [
                    joinpath(path, file) for
                    file in iter_files if occursin("$(batch_id)_input", file)
                ]
                # test input file
                df = Arrow.Table(file_ins)
                @test length(df) == 2 # 2 parameter
                @test length(df[1]) == num_p
                rm.(file_ins)
                # test output file
                df = Arrow.Table(file_outs)
                @test length(df) == 8
                @test length(df[1]) == num_p * successfull_solves
                rm.(file_outs)
            end
        end
    end
    @testset "Dataset Generation JuMP" begin
        model = JuMP.Model(HiGHS.Optimizer)
        @variable(model, x)
        p = @variable(model, _p)
        @constraint(model, cons, x + _p >= 3)
        @objective(model, Min, 2x)
        num_p = 10
        batch_id = string(uuid1())
        problem_iterator = ProblemIterator(Dict(p => collect(1.0:num_p)); param_type=L2O.JuMPParameterType)
        file_output = joinpath(path, "test_$(batch_id)_output") # file path
        recorder = Recorder{ArrowFile}(
            file_output; primal_variables=[x], dual_variables=[cons]
        )
        successfull_solves = solve_batch(problem_iterator, recorder)
        iter_files = readdir(joinpath(path))
        iter_files = filter(x -> occursin(string(ArrowFile), x), iter_files)
        file_outs = [
            joinpath(path, file) for
            file in iter_files if occursin("$(batch_id)_output", file)
        ]
        # test output file
        df = Arrow.Table(file_outs)
        @test length(df) == 8
        @test length(df[1]) == num_p * successfull_solves
        rm.(file_outs)
    end
end

function test_load(model_file::AbstractString, input_file::AbstractString, ::Type{T}, ids::Vector{UUID};
    batch_size::Integer=32
) where {T<:L2O.FileType}
    # Test Load full set 
    problem_iterator = L2O.load(model_file, input_file, T)
    @test problem_iterator isa L2O.AbstractProblemIterator
    @test length(problem_iterator.ids) == length(ids)
    # Test load only half of the ids
    num_ids_ignored = floor(Int, length(ids) / 2)
    problem_iterator =  L2O.load(model_file, input_file, T; ignore_ids=ids[1:num_ids_ignored])
    @test length(problem_iterator.ids) == length(ids) - num_ids_ignored
    # Test warning all ids to be ignored
    problem_iterator =  L2O.load(model_file, input_file, T; ignore_ids=ids)
    @test isnothing(problem_iterator)
    # Test Load batch of problem iterators
    problem_iterator_factory, num_batches =  L2O.load(model_file, input_file, T; batch_size=batch_size)
    @test num_batches == ceil(Int, length(ids) / batch_size)
    @test problem_iterator_factory(1) isa L2O.AbstractProblemIterator
    return nothing
end

function test_compress_batch_arrow(case_file_path::AbstractString=mktempdir(), case_name::AbstractString="test"; keyword::AbstractString="output")
    random_data = rand(10, 10)
    col_names = ["col_$(i)" for i in 1:10]
    batch_ids = [string(uuid1()) for _ in 1:10]
    dfs = [DataFrame(random_data[i:i, :], col_names) for i in 1:10]
    for i in 1:10
        Arrow.write(joinpath(case_file_path, "$(case_name)_$(keyword)_$(batch_ids[i]).arrow"), dfs[i])
    end
    @test length([file for file in readdir(case_file_path; join=true) if occursin(case_name, file) && occursin("arrow", file) && occursin(keyword, file) && any(x -> occursin(x, file), batch_ids)]) == 10
    batch_id = string(uuid1())
    L2O.compress_batch_arrow(case_file_path, case_name; keyword_all=keyword, batch_id=batch_id, keyword_any=batch_ids)
    @test length([file for file in readdir(case_file_path; join=true) if occursin(case_name, file) && occursin("arrow", file) && occursin(keyword, file) && any(x -> occursin(x, file), batch_ids)]) == 0
    @test length([file for file in readdir(case_file_path; join=true) if occursin(case_name, file) && occursin("arrow", file) && occursin(keyword, file) && occursin(batch_id, file)]) == 1
    return nothing
end
