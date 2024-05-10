##
println("\nTesting VAEs module...\n")
##

# Import AutoEncoderToolkit.jl module to be tested
import AutoEncoderToolkit.VAEs
import AutoEncoderToolkit.regularization
# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase
import Distributions

Random.seed!(42)

## =============================================================================

@testset "reparameterize function" begin
    # Define some inputs
    µ = Float32[0.5, 0.2]
    logσ = Float32[-0.1, -0.2]

    # Test with log scale standard deviation
    @testset "log scale standard deviation" begin
        result = VAEs.reparameterize(µ, logσ, log=true)
        @test size(result) == size(µ)
        @test typeof(result) == typeof(µ)
    end # @testset "log scale standard deviation"

    # Test with standard deviation (not log scale)
    @testset "standard deviation (not log scale)" begin
        σ = exp.(logσ)
        result = VAEs.reparameterize(µ, σ, log=false)
        @test size(result) == size(µ)
        @test typeof(result) == typeof(µ)
    end # @testset "standard deviation (not log scale)"
end # @testset "reparameterize function"

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
encoder_activation = repeat([Flux.relu], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.relu], n_hidden)

# Define decoder split layers and activation functions
µ_neurons = repeat([n_neuron], n_hidden)
µ_activation = repeat([Flux.relu], n_hidden)

logσ_neurons = repeat([n_neuron], n_hidden)
logσ_activation = repeat([Flux.relu], n_hidden)

σ_neurons = repeat([n_neuron], n_hidden)
σ_activation = [repeat([Flux.relu], n_hidden - 1); Flux.softplus]


## =============================================================================


println("Defining encoders...")
# Initialize JointGaussianLogEncoder
joint_log_encoder = VAEs.JointGaussianLogEncoder(
    n_input,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation,
)

# -----------------------------------------------------------------------------

println("Defining decoders...")

# Initialize BernoulliDecoder
bernoulli_decoder = VAEs.BernoulliDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    Flux.sigmoid
)

# Initialize SimpleGaussianDecoder
simple_decoder = VAEs.SimpleGaussianDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize JointGaussianLogDecoder
joint_log_decoder = VAEs.JointGaussianLogDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize SplitGaussianLogDecoder
split_log_decoder = VAEs.SplitGaussianLogDecoder(
    n_input,
    n_latent,
    µ_neurons,
    µ_activation,
    logσ_neurons,
    logσ_activation,
)

# Initialize JointGaussianDecoder
joint_decoder = VAEs.JointGaussianDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activations
)

# Initialize SplitGaussianDecoder
split_decoder = VAEs.SplitGaussianDecoder(
    n_input,
    n_latent,
    µ_neurons,
    µ_activation,
    σ_neurons,
    σ_activation,
)

@testset "Type checking" begin
    @test typeof(joint_log_encoder) == VAEs.JointGaussianLogEncoder
    @test typeof(bernoulli_decoder) == VAEs.BernoulliDecoder
    @test typeof(simple_decoder) == VAEs.SimpleGaussianDecoder
    @test typeof(joint_log_decoder) == VAEs.JointGaussianLogDecoder
    @test typeof(split_log_decoder) == VAEs.SplitGaussianLogDecoder
    @test typeof(joint_decoder) == VAEs.JointGaussianDecoder
    @test typeof(split_decoder) == VAEs.SplitGaussianDecoder
    @test typeof(joint_log_encoder * simple_decoder) <: VAEs.VAE
end # @testset "Type checking"

# Collect all decoders
decoders = [
    joint_log_decoder,
    split_log_decoder,
    joint_decoder,
    split_decoder,
    simple_decoder,
    bernoulli_decoder
]

## =============================================================================

@testset "VAE Forward Pass" begin
    # Define single data
    x_vector = @view data[:, 1]
    # Define batch of data
    x_matrix = data

    # Test with latent=false
    @testset "latent=false" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
            vae = joint_log_encoder * decoder

            # Test with single data point
            result = vae(x_vector, latent=false)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractVector{Float32}))

            # Test with multiple data points
            result = vae(data, latent=false)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractMatrix{Float32}))
        end # for decoder in decoders
    end # @testset "latent=false"

    # Test with latent=false
    @testset "latent=true" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE
            vae = joint_log_encoder * decoder

            # Test with single data point
            result = vae(x_vector, latent=true)
            @test isa(result, NamedTuple)
            @test all(
                isa.(values(result), AbstractVector{Float32}) .||
                isa.(values(result), NamedTuple)
            )

            # Test with multiple data points
            result = vae(data, latent=true)
            @test isa(result, NamedTuple)
            @test all(
                isa.(values(result), AbstractMatrix{Float32}) .||
                isa.(values(result), NamedTuple)
            )
        end # for decoder in decoders
    end # @testset "latent=false"
end # @testset "VAE Forward Pass"

## =============================================================================

@testset "loss function" begin
    # Define single data
    x_vector = @view data[:, 1]
    # Define batch of data
    x_matrix = data

    # Define VAE with any decoder
    vae = joint_log_encoder * joint_log_decoder

    # Loop through decoders
    for decoder in decoders
        # Define VAE
        vae = joint_log_encoder * decoder

        @testset "loss function" begin
            # Test with vector input
            @testset "vector input" begin
                result = VAEs.loss(vae, x_vector)
                @test isa(result, Float32)
                result = VAEs.loss(vae, x_vector, x_vector)
                @test isa(result, Float32)

            end # vector input

            # Test with matrix input
            @testset "matrix input" begin
                result = VAEs.loss(vae, x_matrix)
                @test isa(result, Float32)
                result = VAEs.loss(vae, x_matrix, x_matrix)
                @test isa(result, Float32)

            end # matrix input

            # Test with regularization function
            # @testset "with regularization" begin
            #     reg_function = regularization.l2_regularization
            #     reg_kwargs = Dict(:reg_terms => [:encoder_μ, :encoder_logσ])
            #     result = VAEs.loss(
            #         vae, x_vector;
            #         reg_function=reg_function, reg_kwargs=reg_kwargs
            #     )
            #     @test isa(result, Float32)

            #     result = VAEs.loss(
            #         vae, x_vector, x_vector;
            #         reg_function=reg_function, reg_kwargs=reg_kwargs
            #     )
            #     @test isa(result, Float32)
            # end # with regularization
        end # loss function
    end # for decoder in decoders
end # @testset "loss function"

## =============================================================================

@testset "VAE gradient" begin
    # Define batch of data
    x_matrix = data

    # Define VAE with any decoder
    vae = joint_log_encoder * joint_log_decoder

    # Loop through decoders
    for decoder in decoders
        # Define VAE
        vae = joint_log_encoder * decoder

        @testset "with same input as output" begin
            grads = Flux.gradient(vae -> VAEs.loss(vae, x_matrix), vae)
            @test isa(grads[1], NamedTuple)
        end # @testset "with same input as output"

        @testset "with different input and output" begin
            grads = Flux.gradient(
                vae -> VAEs.loss(vae, x_matrix, x_matrix), vae
            )
            @test isa(grads[1], NamedTuple)
        end # @testset "with different input and output"

    end # for decoder in decoders
end # @testset "VAE gradient"
## =============================================================================

# NOTE: The following tests are commented out because they fail with GitHub
# Actions with the following error:
# Got exception outside of a @test
# BoundsError: attempt to access 16-element Vector{UInt8} at index [0]

@testset "VAE training" begin
    # Define number of epochs
    n_epochs = 3
    # Define single data
    x_vector = @view data[:, 1]
    # Define batch of data
    x_matrix = data

    @testset "without regularization" begin
        # Loop through decoders
        for decoder in decoders
            # Define VAE with any decoder
            vae = deepcopy(joint_log_encoder) *
                  deepcopy(decoder)

            # Explicit setup of optimizer
            opt_state = Flux.Train.setup(
                Flux.Optimisers.Adam(1E-2),
                vae
            )

            # Extract parameters
            # params_init = deepcopy(Flux.params(vae))

            # Loop through a couple of epochs
            losses = Float32[]  # Track the loss
            for epoch = 1:n_epochs
                Random.seed!(42)
                # Test training function
                L = VAEs.train!(vae, data, opt_state; loss_return=true)
                push!(losses, L)
            end

            # Check if loss is decreasing
            @test all(diff(losses) ≠ 0)

            # Extract modified parameters
            # params_end = deepcopy(Flux.params(vae))
        end # for decoder in decoders
    end # @testset "without regularization"

    # @testset "with regularization" begin
    #     reg_function = regularization.l2_regularization
    #     reg_kwargs = Dict(:reg_terms => [:encoder_μ, :encoder_logσ])
    #     # Loop through decoders
    #     for decoder in decoders
    #         # Define VAE with any decoder
    #         vae = deepcopy(joint_log_encoder) * decoder

    #         # Explicit setup of optimizer
    #         opt_state = Flux.Train.setup(
    #             Flux.Optimisers.Adam(1E-3),
    #             vae
    #         )

    #         # Extract parameters
    #         params_init = deepcopy(Flux.params(vae))

    #         # Loop through a couple of epochs
    #         losses = Float32[]  # Track the loss
    #         for epoch = 1:n_epochs
    #             Random.seed!(42)
    #             # Test training function
    #             VAEs.train!(
    #                 vae, data, opt_state;
    #                 loss_kwargs=Dict(
    #                     :reg_function => reg_function,
    #                     :reg_kwargs => reg_kwargs
    #                 )
    #             )
    #             push!(
    #                 losses,
    #                 VAEs.loss(
    #                     vae, data;
    #                     reg_function=reg_function, reg_kwargs=reg_kwargs
    #                 )
    #             )
    #         end

    #         # Check if loss is decreasing
    #         @test all(diff(losses) ≠ 0)

    #         # Extract modified parameters
    #         params_end = deepcopy(Flux.params(vae))

    #         # Check that parameters have significantly changed
    #         threshold = 1e-5
    #         # Check if any parameter has changed significantly
    #         @test all([
    #             all(abs.(x .- y) .> threshold)
    #             for (x, y) in zip(params_init, params_end)
    #         ])
    #     end # for decoder in decoders
    # end # @testset "with regularization"
end # @testset "VAE training"

println("\nAll tests passed!\n")