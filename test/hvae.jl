##
println("Testing HVAEs module:\n")
##

# Import AutoEncode.jl module to be tested
import AutoEncode.HVAEs
import AutoEncode.VAEs
import AutoEncode.regularization: l2_regularization
# Import Flux library
import Flux

# Import basic math
import LinearAlgebra
import Random
import StatsBase
import Distributions

Random.seed!(42)

## =============================================================================

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 5
# Define number of hidden layers
n_hidden = 2
# Define number of neurons in non-linear hidden layers
n_neuron = 3
# Define dimensionality of latent space
n_latent = 2

## =============================================================================

println("Generating synthetic data...")

# Define function
f(x₁, x₂) = 10 * exp(-(x₁^2 + x₂^2))

# Defien radius
radius = 3

# Sample random radius
r_rand = radius .* sqrt.(Random.rand(n_data))

# Sample random angles
θ_rand = 2π .* Random.rand(n_data)

# Convert form polar to cartesian coordinates
x_rand = Float32.(r_rand .* cos.(θ_rand))
y_rand = Float32.(r_rand .* sin.(θ_rand))
# Feed numbers to function
z_rand = f.(x_rand, y_rand)

# Compile data into matrix
data = Matrix(hcat(x_rand, y_rand, z_rand)')

# Build single and batch data
x_vector = @view data[:, 1]
x_matrix = data

## =============================================================================

println("Defining layers structure...")

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity
# Define output layer activation function
output_activations = [Flux.identity, Flux.softplus]

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder split layers and activation functions
µ_neurons = repeat([n_neuron], n_hidden)
µ_activation = repeat([Flux.swish], n_hidden)

logσ_neurons = repeat([n_neuron], n_hidden)
logσ_activation = repeat([Flux.swish], n_hidden)

σ_neurons = repeat([n_neuron], n_hidden)
σ_activation = [repeat([Flux.swish], n_hidden - 1); Flux.softplus]

## =============================================================================


println("Defining encoders...")
# Initialize JointLogEncoder
joint_log_encoder = VAEs.JointLogEncoder(
    n_input,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation,
)

# -----------------------------------------------------------------------------

println("Defining decoders...")

# Initialize SimpleDecoder
simple_decoder = VAEs.SimpleDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize JointLogDecoder
joint_log_decoder = VAEs.JointLogDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize SplitLogDecoder
split_log_decoder = VAEs.SplitLogDecoder(
    n_input,
    n_latent,
    µ_neurons,
    µ_activation,
    logσ_neurons,
    logσ_activation,
)

# Initialize JointDecoder
joint_decoder = VAEs.JointDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activations
)

# Initialize SplitDecoder
split_decoder = VAEs.SplitDecoder(
    n_input,
    n_latent,
    µ_neurons,
    µ_activation,
    σ_neurons,
    σ_activation,
)

# Collect all decoders
decoders = [
    joint_log_decoder,
    split_log_decoder,
    joint_decoder,
    split_decoder,
    simple_decoder,
]

## =============================================================================

# Generate latent variables
z_matrix = (joint_log_encoder * simple_decoder)(data, latent=true).z
# Define single latent variable
z_vector = @view z_matrix[:, 1]

@testset "MvDiagGaussianDecoder function" begin
    # Loop through decoders
    for decoder in decoders
        # Test with default standard deviation
        @testset "default standard deviation" begin
            # Run latent variable through decoder
            latent = decoder(z_vector)

            # Build MvNormal distribution
            result = HVAEs.MvDiagGaussianDecoder(decoder, z_vector)
            @test isa(
                result,
                Distributions.MvNormal
            )
            # Check decoder type
            if typeof(decoder) <: VAEs.SimpleDecoder
                # Test mean
                @test result.μ == latent
                @test LinearAlgebra.diag(result.Σ) == ones(length(latent))
            elseif typeof(decoder) <: VAEs.AbstractVariationalLogDecoder
                @test result.μ == latent[1]
                @test LinearAlgebra.diag(result.Σ) == exp.(latent[2]) .^ 2
            elseif typeof(decoder) <: VAEs.AbstractVariationalLinearDecoder
                @test result.μ == latent[1]
                @test LinearAlgebra.diag(result.Σ) == latent[2] .^ 2
            end # if typeof(decoder) 
        end # @testset "default standard deviation"
    end # for decoder in decoders
end # @testset "MvDiagGaussianDecoder function"

## =============================================================================

@testset "SphericalPrior function" begin
    # Test with default standard deviation
    @testset "default standard deviation" begin
        prior = HVAEs.SphericalPrior(z_vector)
        @test isa(
            prior,
            Distributions.MvNormal
        )
        @test prior.μ == zeros(Float32, length(z_vector))
        @test LinearAlgebra.diag(prior.Σ) == ones(Float32, length(z_vector))
    end # @testset "default standard deviation"

    # Test with custom standard deviation
    @testset "custom standard deviation" begin
        σ = 0.5f0
        prior = HVAEs.SphericalPrior(z_vector, σ)
        @test isa(
            prior,
            Distributions.MvNormal
        )
        @test prior.μ == zeros(Float32, length(z_vector))
        @test LinearAlgebra.diag(prior.Σ) == σ .^ 2 .* ones(Float32, length(z_vector))
    end # @testset "custom standard deviation"
end # @testset "SphericalPrior function"

## =============================================================================

@testset "potential_energy function" begin

    # Loop through decoders
    for decoder in decoders
        # Build HVAE
        hvae = HVAEs.HVAE(joint_log_encoder * decoder)

        # Test with default decoder distribution and prior
        @testset "default decoder distribution and prior" begin
            # Build potential energy function
            U = HVAEs.potential_energy(hvae)
            # Check type of potential energy function
            @test typeof(U) <: Function
            # Evaluate potential energy function
            energy = U(x_vector, z_vector)
            @test isa(energy, Float32)
        end # @testset "default decoder distribution and prior"

        # Test with custom decoder distribution and prior
        @testset "custom decoder distribution and prior" begin
            # Define custom decoder distribution
            decoder_dist = HVAEs.MvDiagGaussianDecoder
            # Define custom prior
            prior_dist(z) = HVAEs.SphericalPrior(z, 0.5f0)
            # Build potential energy function
            U = HVAEs.potential_energy(
                hvae; decoder_dist=decoder_dist, prior=prior_dist
            )
            # Check type of potential energy function
            @test typeof(U) <: Function
            # Evaluate potential energy function
            energy = U(x_vector, z_vector)
            @test isa(energy, Float32)
        end
    end # for decoder in decoders
end # @testset "potential_energy function"

## =============================================================================

@testset "leapfrog_step function" begin
    # Define default arguments
    ϵ = 0.1f0
    ρ_matrix = z_matrix
    ρ_vector = z_vector

    # Loop through decoders
    for decoder in decoders
        # Build HVAE
        hvae = HVAEs.HVAE(joint_log_encoder * decoder)

        # Test with default potential energy function and its arguments
        @testset "default potential energy function and its arguments" begin
            @testset "single data point" begin
                # Compute leapfrog step
                z̄, ρ̄ = HVAEs.leapfrog_step(
                    hvae, x_vector, z_vector, ρ_vector, ϵ
                )
                @test isa(z̄, AbstractVector{Float32})
                @test isa(ρ̄, AbstractVector{Float32})
            end # @testset "single data point"

            @testset "multiple data points" begin
                # Compute leapfrog step
                z̄, ρ̄ = HVAEs.leapfrog_step(
                    hvae, x_matrix, z_matrix, ρ_matrix, ϵ
                )
                @test isa(z̄, AbstractMatrix{Float32})
                @test isa(ρ̄, AbstractMatrix{Float32})
            end # @testset "multiple data points

        end # @testset "default potential energy function and its arguments"

        # Test with custom potential energy function and its arguments
        @testset "custom potential energy function and its arguments" begin
            # Define custom potential energy function kwargs
            prior_dist(z) = HVAEs.SphericalPrior(z, 0.5f0)
            potential_energy_kwargs = (
                decoder_dist=HVAEs.MvDiagGaussianDecoder, prior=prior_dist
            )
            # Compute leapfrog step
            z̄, ρ̄ = HVAEs.leapfrog_step(
                hvae, x_vector, z_vector, ρ_vector, ϵ;
                potential_energy_kwargs=potential_energy_kwargs
            )
            @test isa(z̄, AbstractVector{Float32})
            @test isa(ρ̄, AbstractVector{Float32})
        end
    end # for decoder in decoders
end # @testset "leapfrog_step function"

## =============================================================================

@testset "quadratic_tempering function" begin
    # Define the initial inverse temperature, current stage, and total stages
    βₒ = 0.5
    k = 5
    K = 10

    # Compute the inverse temperature at stage k
    βₖ = HVAEs.quadratic_tempering(βₒ, k, K)

    # Test that βₖ is a float and within the expected range
    @test isa(βₖ, AbstractFloat)
    @test βₖ >= βₒ
    @test βₖ <= 1.0

    # Test that βₖ is correct when k = 0 and k = K
    @test HVAEs.quadratic_tempering(βₒ, 0, K) ≈ βₒ
    @test HVAEs.quadratic_tempering(βₒ, K, K) ≈ 1.0
end # @testset "quadratic_tempering function"

## =============================================================================

using Test

@testset "leapfrog_tempering_step function" begin
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    # Loop through decoders
    for decoder in decoders
        # Build HVAE
        hvae = HVAEs.HVAE(joint_log_encoder * decoder)

        # Test the version of the function that accepts AbstractVector inputs
        @testset "single input" begin
            result = HVAEs.leapfrog_tempering_step(
                hvae, x_vector, z_vector, K, ϵ, βₒ
            )
            @test result.z_init == z_vector
            @test isa(result.ρ_init, AbstractVector)
            @test isa(result.z_final, AbstractVector)
            @test isa(result.ρ_final, AbstractVector)
        end # @testset "single input"

        # Test the version of the function that accepts AbstractMatrix inputs
        @testset "multiple inputs" begin
            result = HVAEs.leapfrog_tempering_step(
                hvae, x_matrix, z_matrix, K, ϵ, βₒ
            )
            @test result.z_init == z_matrix
            @test isa(result.ρ_init, AbstractMatrix)
            @test isa(result.z_final, AbstractMatrix)
            @test isa(result.ρ_final, AbstractMatrix)
        end
    end # for decoder in decoders
end # @testset "leapfrog_tempering_step function"

## =============================================================================

@testset "HVAE Forward Pass" begin
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    # Test with latent=true
    @testset "latent=false" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
            hvae = HVAEs.HVAE(joint_log_encoder * decoder)

            @testset "single input" begin
                # Test with single data point
                result = hvae(x_vector, K, ϵ, βₒ; latent=false)
                # Check type of decoder
                if isa(decoder, VAEs.SimpleDecoder)
                    @test typeof(result) <: AbstractVector{Float32}
                else
                    @test typeof(result) <: Tuple{
                        <:AbstractVector{Float32},<:AbstractVector{Float32}
                    }
                end # if isa(decoder, VAEs.SimpleDecoder)
            end # @testset "single input"

            @testset "multiple inputs" begin
                # Test with multiple data points
                result = hvae(x_matrix, K, ϵ, βₒ; latent=false)
                # Check type of decoder
                if isa(decoder, VAEs.SimpleDecoder)
                    @test typeof(result) <: AbstractMatrix{Float32}
                else
                    @test typeof(result) <: Tuple{
                        <:AbstractMatrix{Float32},<:AbstractMatrix{Float32}
                    }
                end # if
            end # @testset "multiple inputs"
        end # for decoder in decoders
    end # @testset "latent=false"

    @testset "latent=true" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
            hvae = HVAEs.HVAE(joint_log_encoder * decoder)

            @testset "single input" begin
                # Test with single data point
                result = hvae(x_vector, K, ϵ, βₒ; latent=true)
                @test isa(result, NamedTuple)
                @test all(isa.(values(result), AbstractVector{Float32}))
            end # @testset "single input"

            @testset "multiple inputs" begin
                # Test with multiple data points
                result = hvae(x_matrix, K, ϵ, βₒ; latent=true)
                @test isa(result, NamedTuple)
                @test all(isa.(values(result), AbstractMatrix{Float32}))
            end # @testset "multiple inputs"
        end # for decoder in decoders
    end # @testset "latent=true"
end # @testset "HVAE Forward Pass"

## =============================================================================

@testset "hamiltonian_elbo" begin
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    # Loop through decoders
    for decoder in decoders
        # Define HVAE
        hvae = HVAEs.HVAE(joint_log_encoder * decoder)

        # Test with vector input
        @testset "vector input" begin
            result = HVAEs.hamiltonian_elbo(hvae, x_vector)
            @test isa(result, Float32)
        end # vector input

        # Test with matrix input
        @testset "matrix input" begin
            result = HVAEs.hamiltonian_elbo(hvae, x_matrix)
            @test isa(result, Float32)
        end # matrix input
    end # for decoder in decoders
end # @testset "hamiltonian_elbo"

## =============================================================================

@testset "loss" begin
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    # Define regularization function and its arguments
    reg_function = l2_regularization
    reg_kwargs = Dict(:reg_terms => [:encoder_logσ])

    # Loop through decoders
    for decoder in decoders
        # Define HVAE
        hvae = HVAEs.HVAE(joint_log_encoder * decoder)

        # Test with vector input without regularization
        @testset "vector input without regularization" begin
            result = HVAEs.loss(hvae, x_vector; K=K, ϵ=ϵ, βₒ=βₒ)
            @test isa(result, Float32)
        end # vector input without regularization

        # Test with matrix input without regularization
        @testset "matrix input without regularization" begin
            result = HVAEs.loss(hvae, x_matrix; K=K, ϵ=ϵ, βₒ=βₒ)
            @test isa(result, Float32)
        end # matrix input without regularization

        # Test with vector input with regularization
        @testset "vector input with regularization" begin
            result = HVAEs.loss(hvae, x_vector; K=K, ϵ=ϵ, βₒ=βₒ, reg_function=reg_function, reg_kwargs=reg_kwargs)
            @test isa(result, Float32)
        end # vector input with regularization

        # Test with matrix input with regularization
        @testset "matrix input with regularization" begin
            result = HVAEs.loss(hvae, x_matrix; K=K, ϵ=ϵ, βₒ=βₒ, reg_function=reg_function, reg_kwargs=reg_kwargs)
            @test isa(result, Float32)
        end # matrix input with regularization
    end # for decoder in decoders
end # @testset "loss"

## =============================================================================

@testset "HVAE training" begin
    # Define number of epochs
    n_epochs = 3
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    @testset "without regularization" begin
        # Define loss_kwargs
        loss_kwargs = Dict(:K => K, :ϵ => ϵ, :βₒ => βₒ)

        # Loop through decoders
        for decoder in decoders
            # Define HVAE with any decoder
            hvae = HVAEs.HVAE(deepcopy(joint_log_encoder) * deepcopy(decoder))

            # Explicit setup of optimizer
            opt_state = Flux.Train.setup(
                Flux.Optimisers.Adam(1E-3),
                hvae
            )

            # Extract parameters
            params_init = deepcopy(Flux.params(hvae))

            # Loop through a couple of epochs
            losses = Float32[]  # Track the loss
            for epoch = 1:n_epochs
                Random.seed!(42)
                # Test training function
                HVAEs.train!(hvae, data, opt_state; loss_kwargs=loss_kwargs)
                push!(losses, HVAEs.loss(hvae, data; loss_kwargs...))
            end

            # Check if loss is decreasing
            @test all(diff(losses) ≠ 0)

            # Extract modified parameters
            params_end = deepcopy(Flux.params(hvae))

            # Check that parameters have significantly changed
            threshold = 1e-5
            # Check if any parameter has changed significantly
            @test all([
                all(abs.(x .- y) .> threshold)
                for (x, y) in zip(params_init, params_end)
            ])
        end # for decoder in decoders
    end # @testset "without regularization"

    @testset "with regularization" begin
        reg_function = l2_regularization
        reg_kwargs = Dict(:reg_terms => [:encoder_μ, :encoder_logσ])

        # Define loss_kwargs
        loss_kwargs = Dict(
            :K => K, :ϵ => ϵ, :βₒ => βₒ,
            :reg_function => reg_function, :reg_kwargs => reg_kwargs
        )

        # Loop through decoders
        for decoder in decoders
            # Define HVAE with any decoder
            hvae = HVAEs.HVAE(deepcopy(joint_log_encoder) * deepcopy(decoder))

            # Explicit setup of optimizer
            opt_state = Flux.Train.setup(
                Flux.Optimisers.Adam(1E-3),
                hvae
            )

            # Extract parameters
            params_init = deepcopy(Flux.params(hvae))

            # Loop through a couple of epochs
            losses = Float32[]  # Track the loss
            for epoch = 1:n_epochs
                Random.seed!(42)
                # Test training function
                HVAEs.train!(hvae, data, opt_state; loss_kwargs=loss_kwargs)
                push!(
                    losses,
                    HVAEs.loss(hvae, data; loss_kwargs...)
                )
            end

            # Check if loss is decreasing
            @test all(diff(losses) ≠ 0)

            # Extract modified parameters
            params_end = deepcopy(Flux.params(hvae))

            # Check that parameters have significantly changed
            threshold = 1e-5
            # Check if any parameter has changed significantly
            @test all([
                all(abs.(x .- y) .> threshold)
                for (x, y) in zip(params_init, params_end)
            ])
        end # for decoder in decoders
    end # @testset "with regularization"
end # @testset "HVAE training"