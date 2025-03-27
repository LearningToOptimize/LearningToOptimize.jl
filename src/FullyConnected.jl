mutable struct FullyConnected
    layers::PairwiseFusion
    pass_through::Vector{<:Dense}
end

@functor FullyConnected

function (model::FullyConnected)(x)
    pass_through_inputs = [layer(x) for layer in model.pass_through]
    return model.layers((x, pass_through_inputs...))[end]
end

function Base.show(io::IO, model::FullyConnected)
    println(io, "FullyConnected(")
    println(io, "     Layers: ", model.layers, ",")
    println(io, "     Pass-Through: ", model.pass_through)
    return println(io, ")")
end

"""
    FullyConnected(input_size::Int, hidden_sizes::Vector{Int}, output_size::Int)::FullyConnected

Create a fully connected neural network with `input_size` inputs, `hidden_sizes` hidden layers and `output_size` outputs.
Adds a pass through layer for each layer.
"""
function FullyConnected(
    input_size::Int,
    hidden_sizes::Vector{Int},
    output_size::Int;
    init = Flux.glorot_uniform(Random.GLOBAL_RNG),
)::FullyConnected
    # Create layers
    layers = []

    # Create the pass through layers
    pass_through = [Dense(input_size, 1; init = init) for _ = 2:(length(hidden_sizes)+1)]

    # Create the first layer connected to the input size
    push!(layers, Dense(input_size, hidden_sizes[1], relu; init = init))

    # Create connections between hidden layers
    for i = 2:length(hidden_sizes)
        # Create a new layer
        push!(layers, Dense(hidden_sizes[i-1] + 1, hidden_sizes[i], relu; init = init))
    end

    # Create the output layer connected to the last hidden layer
    push!(layers, Dense(hidden_sizes[end] + 1, output_size; init = init))

    return FullyConnected(PairwiseFusion(vcat, layers...), pass_through) |> gpu
end

mutable struct FullyConnectedBuilder <: MLJFlux.Builder
    hidden_sizes::Vector{Int}
end

function MLJFlux.build(builder::FullyConnectedBuilder, rng, n_in, n_out)
    init = Flux.glorot_uniform(rng)
    return Chain(FullyConnected(n_in, builder.hidden_sizes, n_out; init = init)) |> gpu
end

# mutable struct ConvexRegressor <: MLJFlux.MLJFluxDeterministic
#     model::MLJFlux.Regressor
# end

# @forward((ConvexRegressor, :model), MLJFlux.Regressor)

# Define a container to hold any optimiser specific parameters (if any):
struct ConvexRule <: Optimisers.AbstractRule
    rule::Optimisers.AbstractRule
    tol::Real
end
function ConvexRule(rule::Optimisers.AbstractRule; tol = 1e-6)
    return ConvexRule(rule, tol)
end

Optimisers.init(o::ConvexRule, x::AbstractArray) = Optimisers.init(o.rule, x)

function Optimisers.apply!(o::ConvexRule, mvel, x::AbstractArray{T}, dx) where {T}
    return Optimisers.apply!(o.rule, mvel, x, dx)
end

"""
    function make_convex!(chain::PairwiseFusion; tol = 1e-6)

Make a `PairwiseFusion` model convex by making sure all dense layers (a part from the first) have positive weights.
This procedure only makes sense for single output fully connected models.
"""
function make_convex!(chain::PairwiseFusion; tol = 1e-6)
    for layer in chain.layers[2:end]
        layer.weight .= max.(layer.weight, tol)
    end
end

"""
    function make_convex!(model::FullyConnected; tol = 1e-6)

Make a `FullyConnected` model convex by making sure all dense layers (a part from the first) have positive weights.
This procedure only makes sense for single output fully connected models.
"""
function make_convex!(model::FullyConnected; tol = 1e-6)
    return make_convex!(model.layers; tol = tol)
end

function make_convex!(model::Chain; tol = 1e-6)
    for i = 1:length(model.layers)
        make_convex!(model.layers[i]; tol = tol)
    end
end

function MLJFlux.train(
    model,
    chain,
    optimiser::ConvexRule,
    optimiser_state,
    epochs,
    verbosity,
    X,
    y,
)

    loss = model.loss

    # intitialize and start progress meter:
    meter = MLJFlux.Progress(
        epochs + 1,
        dt = 0,
        desc = "Optimising neural net:",
        barglyphs = MLJFlux.BarGlyphs("[=> ]"),
        barlen = 25,
        color = :yellow,
    )
    verbosity != 1 || MLJFlux.next!(meter)

    # initiate history:
    n_batches = length(y)

    losses = (loss(chain(X[i]), y[i]) for i = 1:n_batches)
    history = [mean(losses)]

    for i = 1:epochs
        chain, optimiser_state, current_loss =
            MLJFlux.train_epoch(model, chain, optimiser, optimiser_state, X, y)
        make_convex!(chain; tol = optimiser.tol)
        verbosity < 2 || @info "Loss is $(round(current_loss; sigdigits=4))"
        verbosity != 1 || next!(meter)
        push!(history, current_loss)
    end

    return chain, optimiser_state, history

end

function train!(model, loss, opt_state, X, Y; _batchsize = 32, shuffle = true)
    batchsize = min(size(X, 2), _batchsize)
    X = X |> gpu
    Y = Y |> gpu
    data = Flux.DataLoader((X, Y), batchsize = batchsize, shuffle = shuffle)
    for d in data
        ∇model, _ = Flux.gradient(model, d...) do m, x, y  # calculate the gradients
            loss(m(x), y)
        end
        # insert what ever code you want here that needs gradient
        # E.g. logging with TensorBoardLogger.jl as histogram so you can see if it is becoming huge
        opt_state, model = Optimisers.update(opt_state, model, ∇model)
        # Here you might like to check validation set accuracy, and break out to do early stopping
    end
    return loss(model(X), Y)
end
