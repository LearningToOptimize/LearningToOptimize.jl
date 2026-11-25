"""
    test_flux_jump_basic()

Tests running a jump model with a flux expression.
"""
function test_flux_jump_basic()
    for i = 1:10
        model = JuMP.Model(HiGHS.Optimizer)

        @variable(model, x[i=1:3] >= 2.3)

        flux_model = Chain(Dense(3, 3, relu), Dense(3, 1))

        ex = flux_model(x)[1]

        # @constraint(model, ex >= -100.0)
        @constraint(model, sum(x) <= 10)

        @objective(model, Min, ex)

        JuMP.optimize!(model)

        @test termination_status(model) === OPTIMAL
        if flux_model(value.(x))[1] <= 1.0
            @test isapprox(flux_model(value.(x))[1], value(ex); atol = 0.01)
        else
            @test isapprox(flux_model(value.(x))[1], value(ex); rtol = 0.001)
        end
    end
end
