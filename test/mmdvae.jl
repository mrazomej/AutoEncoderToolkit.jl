println("\nTesting MMDVAEs.jl module...\n")
# Import AutoEncoderToolkit.jl module to be tested
import AutoEncoderToolkit.MMDVAEs
import AutoEncoderToolkit.VAEs
import AutoEncoderToolkit.regularization

# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase
import Distributions
import Distances

Random.seed!(42)

## =============================================================================

@testset "MMDVAE struct" begin
    # Define dimensionality of data
    data_dim = 10
    # Define dimensionality of latent space 
    latent_dim = 2
    # Define number of hidden layers in encoder/decoder
    n_hidden = 2
    # Define number of neurons in encoder/decoder hidden layers
    n_neuron = 10
    # Define activation function for encoder/decoder hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)
    # Define activation function for output of encoder
    output_activation = Flux.identity

    # Define encoder and decoder
    encoder = VAEs.JointGaussianLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = VAEs.SimpleGaussianDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = encoder * decoder

    # Initialize MMDVAE
    mmdvae = MMDVAEs.MMDVAE(vae)

    @testset "Type checking" begin
        @test typeof(mmdvae) <: MMDVAEs.MMDVAE{VAEs.VAE{VAEs.JointGaussianLogEncoder,VAEs.SimpleGaussianDecoder}}
    end

    @testset "Forward pass" begin
        # Define batch of data
        x = randn(Float32, data_dim, 10)

        @testset "latent=false" begin
            # Test forward pass
            result = mmdvae(x; latent=false)
            @test isa(result, NamedTuple)
            @test all(isa.(values(result), AbstractMatrix{Float32}))
        end # @testset "latent=false"

        @testset "latent=true" begin
            # Test forward pass
            result = mmdvae(x; latent=true)
            @test isa(result, NamedTuple)
            @test result.encoder isa NamedTuple
            @test result.z isa AbstractMatrix{Float32}
            @test result.decoder isa NamedTuple
        end # @testset "latent=true"
    end # @testset "Forward pass"
end # @testset "MMDVAE struct"

## =============================================================================

@testset "Kernel functions" begin
    # Define input data
    x = randn(Float32, 10, 10)
    y = randn(Float32, 10, 20)

    @testset "gaussian_kernel" begin
        result = MMDVAEs.gaussian_kernel(x, x)
        @test isa(result, Matrix{Float32})
        @test size(result) == (10, 10)

        result = MMDVAEs.gaussian_kernel(x, y)
        @test isa(result, Matrix{Float32})
        @test size(result) == (10, 20)
    end # @testset "gaussian_kernel"

    @testset "mmd_div" begin
        result = MMDVAEs.mmd_div(x, x)
        @test isa(result, Float32)

        result = MMDVAEs.mmd_div(x, y)
        @test isa(result, Float32)
    end # @testset "mmd_div"
end # @testset "Kernel functions"

## =============================================================================

@testset "logP_mmd_ratio" begin
    # Define dimensionality of data
    data_dim = 10
    # Define dimensionality of latent space 
    latent_dim = 2
    # Define number of hidden layers in encoder/decoder
    n_hidden = 2
    # Define number of neurons in encoder/decoder hidden layers
    n_neuron = 10
    # Define activation function for encoder/decoder hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)
    # Define activation function for output of encoder
    output_activation = Flux.identity

    # Define encoder and decoder
    encoder = VAEs.JointGaussianLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = VAEs.SimpleGaussianDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = VAEs.VAE(encoder, decoder)

    # Initialize MMDVAE
    mmdvae = MMDVAEs.MMDVAE(vae)

    # Define batch of data
    x = randn(Float32, data_dim, 10)

    result = MMDVAEs.logP_mmd_ratio(mmdvae, x)
    @test isa(result, Float32)
end # @testset "logP_mmd_ratio"

## =============================================================================

@testset "loss functions" begin
    # Define dimensionality of data
    data_dim = 10
    # Define dimensionality of latent space 
    latent_dim = 2
    # Define number of hidden layers in encoder/decoder
    n_hidden = 2
    # Define number of neurons in encoder/decoder hidden layers
    n_neuron = 10
    # Define activation function for encoder/decoder hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)
    # Define activation function for output of encoder
    output_activation = Flux.identity

    # Define encoder and decoder
    encoder = VAEs.JointGaussianLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = VAEs.SimpleGaussianDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = VAEs.VAE(encoder, decoder)

    # Initialize MMDVAE
    mmdvae = MMDVAEs.MMDVAE(vae)

    # Define batch of data
    x = randn(Float32, data_dim, 10)

    @testset "loss (same input and output)" begin
        result = MMDVAEs.loss(mmdvae, x)
        @test isa(result, Float32)
    end # @testset "loss (same input and output)"

    @testset "loss (different input and output)" begin
        x_out = randn(Float32, data_dim, 10)
        result = MMDVAEs.loss(mmdvae, x, x_out)
        @test isa(result, Float32)
    end # @testset "loss (different input and output)"
end # @testset "loss functions"

## =============================================================================

@testset "MMDVAE gradient" begin
    # Define dimensionality of data
    data_dim = 10
    # Define dimensionality of latent space 
    latent_dim = 2
    # Define number of hidden layers in encoder/decoder
    n_hidden = 2
    # Define number of neurons in encoder/decoder hidden layers
    n_neuron = 10
    # Define activation function for encoder/decoder hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)
    # Define activation function for output of encoder
    output_activation = Flux.identity

    # Define encoder and decoder
    encoder = VAEs.JointGaussianLogEncoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    decoder = VAEs.SimpleGaussianDecoder(
        data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
    )
    # Define VAE
    vae = VAEs.VAE(encoder, decoder)

    # Initialize MMDVAE
    mmdvae = MMDVAEs.MMDVAE(vae)

    # Define batch of data
    x = randn(Float32, data_dim, 10)

    @testset "with same input and output" begin
        grads = Flux.gradient(mmdvae -> MMDVAEs.loss(mmdvae, x), mmdvae)
        @test isa(grads[1], NamedTuple)
    end # @testset "with same input and output"

    @testset "with different input and output" begin
        x_out = randn(Float32, data_dim, 10)
        grads = Flux.gradient(mmdvae -> MMDVAEs.loss(mmdvae, x, x_out), mmdvae)
        @test isa(grads[1], NamedTuple)
    end # @testset "with different input and output"
end # @testset "MMDVAE gradient"

## =============================================================================

# NOTE: The following tests are commented out because they fail with GitHub
# Actions with the following error:
# Got exception outside of a @test
# BoundsError: attempt to access 16-element Vector{UInt8} at index [0]

# @testset "MMDVAE training" begin
#     # Define dimensionality of data
#     data_dim = 10
#     # Define dimensionality of latent space 
#     latent_dim = 2
#     # Define number of hidden layers in encoder/decoder
#     n_hidden = 2
#     # Define number of neurons in encoder/decoder hidden layers
#     n_neuron = 10
#     # Define activation function for encoder/decoder hidden layers
#     hidden_activation = repeat([Flux.relu], n_hidden)
#     # Define activation function for output of encoder
#     output_activation = Flux.identity

#     # Define encoder and decoder
#     encoder = VAEs.JointGaussianLogEncoder(
#         data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
#     )
#     decoder = VAEs.SimpleGaussianDecoder(
#         data_dim, latent_dim, repeat([n_neuron], n_hidden), hidden_activation, output_activation
#     )
#     # Define VAE
#     vae = VAEs.VAE(encoder, decoder)

#     # Initialize MMDVAE
#     mmdvae = MMDVAEs.MMDVAE(vae)

#     # Define batch of data
#     x = randn(Float32, data_dim, 10)

#     # Explicit setup of optimizer
#     opt = Flux.Train.setup(Flux.Optimisers.Adam(), mmdvae)

#     @testset "with same input and output" begin
#         L = MMDVAEs.train!(mmdvae, x, opt; loss_return=true)
#         @test isa(L, Float32)
#     end # @testset "with same input and output"

#     @testset "with different input and output" begin
#         x_out = randn(Float32, data_dim, 10)
#         L = MMDVAEs.train!(mmdvae, x, x_out, opt; loss_return=true)
#         @test isa(L, Float32)
#     end # @testset "with different input and output"
# end # @testset "MMDVAE training"

println("\nAll tests passed!\n")