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

@testset "decoder_loglikelihood tests" begin
    # Loop through decoders
    for decoder in decoders
        # Test 1: Check that the function returns a Float32
        result = HVAEs.decoder_loglikelihood(decoder, x_vector, z_vector)
        @test isa(result, Float32)
    end # for decoder in decoders
end # @testset "decoder_loglikelihood tests"

## =============================================================================

@testset "spherical_logprior" begin
    # Test 1: Check that the function returns a Float32
    result = HVAEs.spherical_logprior(z_vector)
    @test isa(result, Float32)
end # @testset "spherical_logprior"

## =============================================================================

@testset "potential_energy function" begin
    # Test with default decoder distribution and prior
    @testset "default decoder distribution and prior" begin
        # Loop through decoders
        for decoder in decoders
            # Build HVAE
            hvae = HVAEs.HVAE(joint_log_encoder * decoder)
            # Build potential energy function
            result = HVAEs.potential_energy(hvae, x_vector, z_vector)
            @test isa(result, Float32)
        end # @testset "default decoder distribution and prior"
    end # for decoder in decoders
end # @testset "potential_energy function"

## =============================================================================

@testset "∇potential_energy function" begin
    # Test with default decoder distribution and prior
    @testset "default decoder distribution and prior" begin
        # Loop through decoders
        for decoder in decoders
            # Build HVAE
            hvae = HVAEs.HVAE(joint_log_encoder * decoder)
            # Build potential energy function
            result = HVAEs.∇potential_energy(hvae, x_vector, z_vector)
            @test isa(result, Vector{Float32})
        end # @testset "default decoder distribution and prior"
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

@testset "leapfrog_tempering_step function" begin
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    # Loop through decoders
    for decoder in decoders
        # Build HVAE
        hvae = HVAEs.HVAE(joint_log_encoder * decoder)
        # Test with default potential energy function and its arguments
        @testset "default potential energy function and its arguments" begin
            @testset "single data point" begin
                # Compute leapfrog step
                result = HVAEs.leapfrog_tempering_step(
                    hvae, x_vector, z_vector;
                    K=K, ϵ=ϵ, βₒ=βₒ
                )
                @test isa(result, NamedTuple)
                @test all(isa.(values(result), AbstractVector{Float32}))
            end # @testset "single data point"

            @testset "multiple data points" begin
                # Compute leapfrog step
                result = HVAEs.leapfrog_tempering_step(
                    hvae, x_matrix, z_matrix;
                    K=K, ϵ=ϵ, βₒ=βₒ
                )
                @test isa(result, NamedTuple)
                @test all(isa.(values(result), AbstractMatrix{Float32}))

            end # @testset "multiple data points
        end # @testset "default potential energy function and its arguments"
    end # for decoder in decoders
end # @testset "leapfrog_tempering_step function"

## =============================================================================

@testset "HVAE Forward Pass" begin
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    @testset "Without ∇U provided" begin
        @testset "latent=false" begin
            # Loop through decoders
            for decoder in decoders
                # Define VAE
                hvae = HVAEs.HVAE(joint_log_encoder * decoder)

                @testset "single input" begin
                    # Test with single data point
                    result = hvae(
                        x_vector;
                        K=K, ϵ=ϵ, βₒ=βₒ, latent=false
                    )
                    @test isa(result, NamedTuple)
                    @test all(isa.(values(result), AbstractVector{Float32}))
                end # @testset "single input"

                @testset "multiple inputs" begin
                    # Test with single data point
                    result = hvae(
                        x_matrix;
                        K=K, ϵ=ϵ, βₒ=βₒ, latent=false
                    )
                    @test isa(result, NamedTuple)
                    @test all(isa.(values(result), AbstractMatrix{Float32}))

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
                    result = hvae(
                        x_vector;
                        K=K, ϵ=ϵ, βₒ=βₒ, latent=true
                    )
                    @test isa(result, NamedTuple)
                end # @testset "single input"

                @testset "multiple inputs" begin
                    # Test with single data point
                    result = hvae(
                        x_matrix;
                        K=K, ϵ=ϵ, βₒ=βₒ, latent=true
                    )
                    @test isa(result, NamedTuple)
                end # @testset "multiple inputs"
            end # for decoder in decoders
        end # @testset "latent=true"
    end # @testset "Without ∇U provided"
end # @testset "HVAE Forward Pass"

## =============================================================================

@testset "hamiltonian_elbo" begin
    # Define the number of steps, step size, and initial inverse temperature
    K = 2
    ϵ = 0.01f0
    βₒ = 1.0f0

    @testset "Without ∇U provided" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
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
    end # @testset "Without U and ∇U provided"
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

        # # Test with vector input with regularization
        # @testset "vector input with regularization" begin
        #     result = HVAEs.loss(hvae, x_vector; K=K, ϵ=ϵ, βₒ=βₒ, reg_function=reg_function, reg_kwargs=reg_kwargs)
        #     @test isa(result, Float32)
        # end # vector input with regularization

        # # Test with matrix input with regularization
        # @testset "matrix input with regularization" begin
        #     result = HVAEs.loss(hvae, x_matrix; K=K, ϵ=ϵ, βₒ=βₒ, reg_function=reg_function, reg_kwargs=reg_kwargs)
        #     @test isa(result, Float32)
        # end # matrix input with regularization
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

    # @testset "with regularization" begin
    #     reg_function = l2_regularization
    #     reg_kwargs = Dict(:reg_terms => [:encoder_μ, :encoder_logσ])

    #     # Define loss_kwargs
    #     loss_kwargs = Dict(
    #         :K => K, :ϵ => ϵ, :βₒ => βₒ,
    #         :reg_function => reg_function, :reg_kwargs => reg_kwargs
    #     )

    #     # Loop through decoders
    #     for decoder in decoders
    #         # Define HVAE with any decoder
    #         hvae = HVAEs.HVAE(deepcopy(joint_log_encoder) * deepcopy(decoder))

    #         # Explicit setup of optimizer
    #         opt_state = Flux.Train.setup(
    #             Flux.Optimisers.Adam(1E-3),
    #             hvae
    #         )

    #         # Extract parameters
    #         params_init = deepcopy(Flux.params(hvae))

    #         # Loop through a couple of epochs
    #         losses = Float32[]  # Track the loss
    #         for epoch = 1:n_epochs
    #             Random.seed!(42)
    #             # Test training function
    #             HVAEs.train!(hvae, data, opt_state; loss_kwargs=loss_kwargs)
    #             push!(
    #                 losses,
    #                 HVAEs.loss(hvae, data; loss_kwargs...)
    #             )
    #         end

    #         # Check if loss is decreasing
    #         @test all(diff(losses) ≠ 0)

    #         # Extract modified parameters
    #         params_end = deepcopy(Flux.params(hvae))

    #         # Check that parameters have significantly changed
    #         threshold = 1e-5
    #         # Check if any parameter has changed significantly
    #         @test all([
    #             all(abs.(x .- y) .> threshold)
    #             for (x, y) in zip(params_init, params_end)
    #         ])
    #     end # for decoder in decoders
    # end # @testset "with regularization"
end # @testset "HVAE training"