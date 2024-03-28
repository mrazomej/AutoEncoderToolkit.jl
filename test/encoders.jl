# Import AutoEncode.jl module to be tested
using AutoEncode

# Import necessary libraries for testing
import Flux
import LinearAlgebra
import Random
import StatsBase

## =============================================================================

@testset "Encoder" begin
    # Create an Encoder instance with sample parameters
    encoder = AutoEncode.Encoder(784, 64, [256, 128], [relu, relu], tanh)

    # Generate random input data
    x = randn(Float32, 784, 10)

    # Pass the input through the encoder
    z = encoder(x)

    # Check if the output has the correct size
    @test size(z) == (64, 10)
end

## =============================================================================

@testset "JointLogEncoder" begin
    @testset "with single activation function" begin
        # Create a JointLogEncoder instance with sample parameters
        encoder = AutoEncode.JointLogEncoder(
            784, 64, [256, 128], [relu, relu], tanh
        )

        # Generate random input data
        x = randn(Float32, 784, 10)

        # Pass the input through the encoder
        encoder_output = encoder(x)

        # Check if the output is a NamedTuple with the correct keys
        @test encoder_output isa NamedTuple{(:μ, :logσ)}

        # Check if the output has the correct size
        @test size(encoder_output.μ) == (64, 10)
        @test size(encoder_output.logσ) == (64, 10)
    end

    @testset "with separate activation functions" begin
        # Create a JointLogEncoder instance with sample parameters
        encoder = AutoEncode.JointLogEncoder(
            784, 64, [256, 128], [relu, relu], [tanh, identity]
        )

        # Generate random input data
        x = randn(Float32, 784, 10)

        # Pass the input through the encoder
        encoder_output = encoder(x)

        # Check if the output is a NamedTuple with the correct keys
        @test encoder_output isa NamedTuple{(:μ, :logσ)}

        # Check if the output has the correct size
        @test size(encoder_output.μ) == (64, 10)
        @test size(encoder_output.logσ) == (64, 10)
    end
end

## =============================================================================

@testset "JointEncoder" begin
    # Create a JointEncoder instance with sample parameters
    encoder = AutoEncode.JointEncoder(
        784, 64, [256, 128], [relu, relu], [tanh, softplus]
    )

    # Generate random input data
    x = randn(Float32, 784, 10)

    # Pass the input through the encoder
    encoder_output = encoder(x)

    # Check if the output is a NamedTuple with the correct keys
    @test encoder_output isa NamedTuple{(:μ, :σ)}

    # Check if the output has the correct size
    @test size(encoder_output.μ) == (64, 10)
    @test size(encoder_output.σ) == (64, 10)
end

## =============================================================================

@testset "spherical_logprior" begin
    @testset "with vector input" begin
        # Generate random latent variable
        z = randn(Float32, 64)

        # Compute the log-prior
        logprior = AutoEncode.spherical_logprior(z)

        # Check if the log-prior is a scalar
        @test logprior isa Float32
    end

    @testset "with matrix input" begin
        # Generate random latent variables
        z = randn(Float32, 64, 10)

        # Compute the log-prior
        logprior = AutoEncode.spherical_logprior(z)

        # Check if the log-prior is a vector
        @test logprior isa Vector{Float32}
        @test length(logprior) == 10
    end
end

## =============================================================================

@testset "encoder_logposterior" begin
    @testset "with vector input" begin
        # Create a JointLogEncoder instance with sample parameters
        encoder = AutoEncode.JointLogEncoder(
            784, 64, [256, 128], [relu, relu], tanh
        )

        # Generate random input data
        x = randn(Float32, 784)

        # Pass the input through the encoder
        encoder_output = encoder(x)

        # Generate random latent variable
        z = randn(Float32, 64)

        # Compute the log-posterior
        logposterior = AutoEncode.encoder_logposterior(
            z, encoder, encoder_output
        )

        # Check if the log-posterior is a scalar
        @test logposterior isa Float32
    end

    @testset "with matrix input" begin
        # Create a JointLogEncoder instance with sample parameters
        encoder = AutoEncode.JointLogEncoder(
            784, 64, [256, 128], [relu, relu], tanh
        )

        # Generate random input data
        x = randn(Float32, 784, 10)

        # Pass the input through the encoder
        encoder_output = encoder(x)

        # Generate random latent variables
        z = randn(Float32, 64, 10)

        # Compute the log-posterior
        logposterior = AutoEncode.encoder_logposterior(
            z, encoder, encoder_output
        )

        # Check if the log-posterior is a vector
        @test logposterior isa Vector{Float32}
        @test length(logposterior) == 10
    end

    @testset "with single index" begin
        # Create a JointLogEncoder instance with sample parameters
        encoder = AutoEncode.JointLogEncoder(
            784, 64, [256, 128], [relu, relu], tanh
        )

        # Generate random input data
        x = randn(Float32, 784, 10)

        # Pass the input through the encoder
        encoder_output = encoder(x)

        # Generate random latent variable
        z = randn(Float32, 64)

        # Compute the log-posterior for a single index
        logposterior = AutoEncode.encoder_logposterior(
            z, encoder, encoder_output, 5
        )

        # Check if the log-posterior is a scalar
        @test logposterior isa Float32
    end
end

## =============================================================================

@testset "encoder_kl" begin
    # Create a JointLogEncoder instance with sample parameters
    encoder = AutoEncode.JointLogEncoder(
        784, 64, [256, 128], [relu, relu], tanh
    )

    # Generate random input data
    x = randn(Float32, 784, 10)

    # Pass the input through the encoder
    encoder_output = encoder(x)

    # Compute the KL divergence
    kl_div = AutoEncode.encoder_kl(encoder, encoder_output)

    # Check if the KL divergence is a vector
    @test kl_div isa Vector{Float32}
    @test length(kl_div) == 10
end