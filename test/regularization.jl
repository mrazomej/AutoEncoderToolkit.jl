##
println("Testing regularization module:\n")
##

# Import AutoEncoderToolkit.jl module to be tested
import AutoEncoderToolkit.regularization
import AutoEncoderToolkit.VAEs
# Import Flux library
import Flux

# Import basic math
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

## =============================================================================

@testset "l2_regularization function" begin
    # Define VAE
    vae = joint_log_encoder * simple_decoder
    # Get outputs
    outputs = vae(data; latent=true)

    # Test with all keys
    @testset "all keys" begin
        reg_terms = [:encoder_μ, :encoder_logσ, :z, :decoder_μ]
        reg_term = regularization.l2_regularization(
            outputs; reg_terms=reg_terms
        )
        @test isa(reg_term, Float32)
    end

    # Test with some keys
    @testset "some keys" begin
        reg_terms = [:encoder_μ, :z]
        reg_term = regularization.l2_regularization(
            outputs, reg_terms=reg_terms
        )
        @test isa(reg_term, Float32)
    end

    # Test with no keys
    @testset "no keys" begin
        reg_terms = Symbol[]
        reg_term = regularization.l2_regularization(
            outputs, reg_terms=reg_terms
        )
        @test reg_term == 0.0f0
    end

    # Test with a key not in outputs
    @testset "key not in outputs" begin
        reg_terms = [:encoder_μ, :not_a_key]
        @test_throws ErrorException regularization.l2_regularization(
            outputs, reg_terms=reg_terms
        )
    end
end

## =============================================================================

@testset "min_variance_regularization function" begin
    # Test with decoder_logσ and logσ=true
    @testset "decoder_logσ" begin
        # Compute VAE outputs
        vae_outputs_logσ = (joint_log_encoder * joint_log_decoder)(
            data; latent=true
        )
        @testset "log=true" begin
            reg_term = regularization.min_variance_regularization(
                vae_outputs_logσ, σ_min=0.1f0, logσ=true
            )
            @test isa(reg_term, Float32)
        end # log=true

        @testset "logσ=false" begin
            reg_term = regularization.min_variance_regularization(
                vae_outputs_logσ, σ_min=0.1f0, logσ=false
            )
            @test isa(reg_term, Float32)
        end # logσ=false
    end # decoder_logσ

    # Test with decoder_σ and logσ=true
    @testset "decoder_logσ" begin
        # Compute VAE outputs
        vae_outputs_σ = (joint_log_encoder * joint_decoder)(
            data; latent=true
        )
        @testset "log=true" begin
            reg_term = regularization.min_variance_regularization(
                vae_outputs_σ, σ_min=0.1f0, logσ=true
            )
            @test isa(reg_term, Float32)
        end # log=true

        @testset "logσ=false" begin
            reg_term = regularization.min_variance_regularization(
                vae_outputs_σ, σ_min=0.1f0, logσ=false
            )
            @test isa(reg_term, Float32)
        end # logσ=false
    end # decoder_σ

    # Test with missing decoder standard deviation
    @testset "missing decoder standard deviation" begin
        vae_outputs_missing = NamedTuple()
        @test_throws ArgumentError regularization.min_variance_regularization(
            vae_outputs_missing, σ_min=0.1f0, logσ=true
        )
    end
end

## =============================================================================

@testset "entropy_regularization function" begin
    # Test with encoder_logσ
    @testset "encoder_logσ" begin
        # Compute VAE outputs
        vae_outputs_logσ = (joint_log_encoder * joint_log_decoder)(
            data; latent=true
        )
        entropy_reg = regularization.entropy_regularization(
            vae_outputs_logσ, reg_terms=[:encoder_logσ]
        )
        @test isa(entropy_reg, Float32)
    end # encoder_logσ

    # Test with decoder_logσ
    @testset "decoder_logσ" begin
        # Compute VAE outputs
        vae_outputs_logσ = (joint_log_encoder * joint_log_decoder)(
            data; latent=true
        )
        entropy_reg = regularization.entropy_regularization(
            vae_outputs_logσ, reg_terms=[:decoder_logσ]
        )
        @test isa(entropy_reg, Float32)
    end # decoder_logσ

    # Test with decoder_σ
    @testset "decoder_σ" begin
        # Compute VAE outputs
        vae_outputs_σ = (joint_log_encoder * joint_decoder)(
            data; latent=true
        )
        entropy_reg = regularization.entropy_regularization(
            vae_outputs_σ, reg_terms=[:decoder_σ]
        )
        @test isa(entropy_reg, Float32)
    end # decoder_σ

    # Test with missing decoder standard deviation
    @testset "missing decoder standard deviation" begin
        vae_outputs_missing = NamedTuple()
        @test_throws ArgumentError regularization.entropy_regularization(
            vae_outputs_missing, reg_terms=[:encoder_logσ]
        )
    end
end