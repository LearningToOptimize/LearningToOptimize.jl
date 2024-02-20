"""
    function line_sampler(
        original_parameters::Vector{T},
        parameter_indexes::Vector{F},
        range_p::AbstractVector{T},
    ) where {T<:Real,F<:Integer}

This sampler returns a set of parameters that for a line in one dimension of the parameter space. 
The idea is to change the value of one parameter and keep the rest constant.
"""
function line_sampler(
    original_parameters::Vector{T},
    parameter_indexes::AbstractVector{F},
    range_p::AbstractVector{T}=1.01:0.01:1.25,
) where {T<:Real,F<:Integer}
    parameters = hcat(fill(original_parameters, length(range_p))...)

    for parameter_index in parameter_indexes
        parameters[parameter_index, :] = [original_parameters[parameter_index] * mul for mul in range_p]
    end

    return parameters
end

function line_sampler(
    original_parameters::Vector{T},
    range_p::AbstractVector{T}=1.01:0.01:1.25,
) where {T<:Real}
    parameters = zeros(T, length(original_parameters), length(range_p) * (1 + length(original_parameters)))
    parameters[:, 1:length(range_p)] = line_sampler(original_parameters, 1:length(original_parameters), range_p)

    for parameter_index=1:length(original_parameters)
        parameters[:, length(range_p) * parameter_index + 1:length(range_p) * (parameter_index + 1)] = line_sampler(original_parameters, [parameter_index], range_p)
    end

    return parameters
end

"""
    function box_sampler(
        original_parameter::T,
        num_s::F,
        range_p::AbstractVector{T}=0.8:0.01:1.25,
    ) where {T<:Real,F<:Integer}
    
Uniformly sample values around the original parameter value over a discrete range inside a box.
"""
function box_sampler(
    original_parameter::T,
    num_s::F,
    range_p::AbstractVector{T}=0.8:0.01:1.25,
) where {T<:Real,F<:Integer}
    parameter_samples =
        original_parameter * rand(range_p, num_s)
    return parameter_samples
end

function box_sampler(
    original_parameters::Vector{T},
    num_s::F,
    range_p::AbstractVector{T}=0.8:0.01:1.25,
) where {T<:Real,F<:Integer}
    return vcat([box_sampler(p, num_s, range_p)' for p in original_parameters]...)
end

function scaled_distribution_sampler(
    original_parameters::Vector{T},
    num_s::F;
    rng::AbstractRNG=Random.GLOBAL_RNG,
    scaler_multiplier::Distribution=Uniform(0.8, 1.25),
    distribution::Distribution=MvLogNormal(fill(-(1.05 .^ 2) ./ 2.0, length(original_parameters)), 1.05)
) where {T<:Real,F<:Integer}
    column_scales = rand(rng, scaler_multiplier, num_s)
    parameter_samples = rand(rng, distribution, num_s)
    
    for n in 1:num_s
        parameter_samples[:, n] = original_parameters .* parameter_samples[:, n] .* column_scales[n]
    end
    return parameter_samples
end

function general_sampler(
    original_parameters::Vector{T};
    samplers::Vector{Function}=[
        (original_parameters) -> scaled_distribution_sampler(original_parameters, 1000),
        line_sampler, 
        (original_parameters) -> box_sampler(original_parameters, 10),
    ]
) where {T<:Real}
    return hcat([sampler(original_parameters) for sampler in samplers]...)
end

function load_parameters(file::AbstractString)
    model = read_from_file(file)
    cons = constraint_object.(all_constraints(model, VariableRef, MOI.Parameter{Float64}))
    parameters = [cons[i].func for i in 1:length(cons)]
    vals = [cons[i].set.value for i in 1:length(cons)]
    return parameters, vals
end

function general_sampler(file::AbstractString)
    return general_sampler(load_parameters(file))
end
