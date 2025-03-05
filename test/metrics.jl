
function test_sobolev_pb_loss(; num_p = 30, num_s = 100, num_y = 2)
    @testset "Sobolev Training Loss" begin
        Random.seed!(666)
        # Data
        Xtrain = rand(Float32,(num_p,num_s)) * Float32(100.0) .- Float32(50.0);
        dy = rand(Float32,(num_y,num_s)); # dL/dy
        # Ackley Function
        # sample coeficients
        coef = rand(Float32, num_y)
        f = x -> [-20 .* coef[i] .* exp(-0.2 * sqrt(mean(x .^ 2))) .- exp(mean(cos.(2 * π * x))) .+ 20 .* coef[i] .+ exp(1) for i in 1:num_y]
        Ytrain = Array{Float32}(undef, num_y, num_s)
        dx = Array{Float32}(undef, num_p, num_s) # dL/dx
        for i in 1:num_s
            Ytrain[:, i], pb = Zygote.pullback(Xtrain[:,i]) do X
                f(X)
            end
            dx[:,i] = pb(dy[:,i])[1]
        end
        Xtest = rand(Float32,(num_p,num_s)) * Float32(2.0) .- Float32(1.0);
        Ytest = hcat([f(Xtest[:,i]) for i in 1:num_s]...)

        ######## Sobolev Loss ########
        #******************* NN ********************
        model = Chain(Dense(num_p, 10, relu), Dense(10, num_y))
        model_mse = deepcopy(model)
        #******************* Sobolev ********************
        @test LearningToOptimize.sobolev_pb_loss(model, Ytrain, Xtrain, dy, dx, 0.1) >= 0.0
        ŷ, pb = LearningToOptimize.sobolev_pb(model, Xtrain, dy)
        @test ŷ == model(Xtrain)
        @test size(pb) == size(dx)
        #******************* Flux Train ********************
        # The weight is important if the loss is not scaled
        loss = sobolev_pb_loss_generator(500, num_p, num_y)
        @test loss isa Function
        opt_state = Flux.setup(Optimisers.Adam(), model)
        num_epochs = 150
        # Train the model
        data = Flux.DataLoader((vcat(Xtrain, dy), vcat(Ytrain, dx)), batchsize=32)
        log = Vector{Float32}(undef, num_epochs)
        for i = 1:num_epochs
            Flux.train!(model, data, opt_state) do m, x, y
                loss(m, x, y)
            end
            log[i] = Flux.mse(model(Xtest), Ytest)
        end

        ######## MSE Loss ########
        #******************* Flux Train ********************
        loss = Flux.mse
        opt_state = Flux.setup(Optimisers.Adam(), model)
        # Train the model_mse
        data = Flux.DataLoader((Xtrain, Ytrain), batchsize=32)
        log_mse = Vector{Float32}(undef, num_epochs)
        for i = 1:num_epochs
            Flux.train!(model_mse, data, opt_state) do m, x, y
                loss(m(x), y)
            end
            log_mse[i] = Flux.mse(model_mse(Xtest), Ytest)
        end

        #******************* Test ********************
        @test log[end] < log_mse[end] # Sobolev should be better than MSE but this is a bad test since perfomance is stochastic
    end
end