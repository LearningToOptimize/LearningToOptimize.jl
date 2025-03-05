"""
    relative_rmse(ŷ, y)

Compute the relative root mean squared error between `ŷ` and `y`.
"""
function relative_rmse(ŷ, y)
    return sqrt(mean(((ŷ .- y) ./ y) .^ 2))
end

"""
    relative_mae(ŷ, y)

Compute the relative mean absolute error between `ŷ` and `y`.
"""
function relative_mae(ŷ, y)
    return mean(abs.((ŷ .- y) ./ y))
end

"""
    sobolev_pb(model, X, dy)

Compute the output of `model` and its pullback with respect to `X` given `dy`. Used in `sobolev_pb_loss`.
"""
function sobolev_pb(model, X, dy)
    y, pb = Zygote.pullback(X) do X
        model(X)
    end
    return y, pb(dy)[1]
end

"""
    sobolev_pb_loss(model, y, X, dy, dx, λ)

Compute weighted (`λ`) mean squared error between `model(X)` and `y` and the stochastic estimation of
the mean squared error between model derivative and the true derivative. The estimation uses the pullback input `dy`
and the true derivative `dx`.
"""
function sobolev_pb_loss(model, y, X, dy, dx, λ)
    ŷ, pb = sobolev_pb(model, X, dy)
    return Flux.mse(ŷ, y) + λ * Flux.mse(dx, pb)
end

"""
    sobolev_pb_loss_generator(λ, num_p, num_y)

Generate a loss function for `sobolev_pb_loss` with fixed weight `λ` for a known 
number of model inputs `num_p` and outputs `num_y`.
"""
function sobolev_pb_loss_generator(λ, num_p, num_y)
    return (m, x_all, y_all) -> begin
        X = x_all[1:num_p, :]
        Y = y_all[1:num_y, :]
        dx = y_all[num_y + 1:end, :]
        dy = x_all[num_p + 1:end, :]
        return sobolev_pb_loss(m, Y, X, dy, dx, λ)
    end
end


# ## Simple Example function
# using Statistics
# # Data
# num_p = 30
# num_s = 100
# num_y = 2

# Xtrain = rand(Float32,(num_p,num_s)) * Float32(100.0) .- Float32(50.0);
# dy = rand(Float32,(num_y,num_s)); # dL/dy
# # Ackley Function
# # sample coeficients
# coef = rand(Float32, num_y)
# f = x -> [-20 .* coef[i] .* exp(-0.2 * sqrt(mean(x .^ 2))) .- exp(mean(cos.(2 * π * x))) .+ 20 .* coef[i] .+ exp(1) for i in 1:num_y]
# Ytrain = Array{Float32}(undef, num_y, num_s)
# dx = Array{Float32}(undef, num_p, num_s) # dL/dx
# for i in 1:num_s
#     Ytrain[:, i], pb = Zygote.pullback(Xtrain[:,i]) do X
#         f(X)
#     end
#     dx[:,i] = pb(dy[:,i])[1]
# end
# Xtest = rand(Float32,(num_p,num_s)) * Float32(2.0) .- Float32(1.0);
# Ytest = hcat([f(Xtest[:,i]) for i in 1:num_s]...)


# ######## Sobolev Loss ########
# #******************* NN ********************
# model = Chain(Dense(num_p, 10, relu), Dense(10, num_y))
# model_mse = deepcopy(model)
# #******************* Sobolev ********************

# sobolev_pb_loss(model, Ytrain, Xtrain, dy, dx, 0.1)

# #******************* Flux Train ********************
# # The weight is important if the loss is not scaled
# loss = sobolev_pb_loss_generator(500, num_p, num_y)

# num_epochs = 150
# # Train the model
# data = Flux.DataLoader((vcat(Xtrain, dy), vcat(Ytrain, dx)), batchsize=32)
# log = Vector{Float32}(undef, num_epochs)
# for i = 1:num_epochs
#     Flux.train!(model, data, Flux.Adam()) do m, x, y
#         loss(m, x, y)
#     end
#     log[i] = Flux.mse(model(Xtest), Ytest)
# end

# ######## MSE Loss ########
# #******************* NN ********************
# # model_mse = Chain(Dense(num_p, 10, relu), Dense(10, num_y))

# #******************* Flux Train ********************
# loss = Flux.mse

# # Train the model_mse
# data = Flux.DataLoader((Xtrain, Ytrain), batchsize=32)
# log_mse = Vector{Float32}(undef, num_epochs)
# for i = 1:num_epochs
#     Flux.train!(model_mse, data, Flux.Adam()) do m, x, y
#         loss(m(x), y)
#     end
#     log_mse[i] = Flux.mse(model_mse(Xtest), Ytest)
# end

# #******************* Plot ********************
# using Plots

# plot(log, label="Sobolev Loss", xlabel="Epochs", ylabel="MSE Loss", title="Ackley Function")
# plot!(log_mse, label="MSE Loss")