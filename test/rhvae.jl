##
println("\nTesting RHVAEs module:\n")
##

# Import AutoEncoderToolkit.jl module to be tested
using AutoEncoderToolkit.RHVAEs
import AutoEncoderToolkit.VAEs
import AutoEncoderToolkit.utils
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
x_vector = data[:, 1]
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

metric_neurons = repeat([n_neuron], n_hidden)
metric_activation = repeat([Flux.relu], n_hidden)
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

# Collect all decoders
decoders = [
    joint_log_decoder,
    split_log_decoder,
    joint_decoder,
    split_decoder,
    simple_decoder,
]

## =============================================================================

# Define data and parameters to use througout the tests
x_mat = data
x_vec = data[:, 1]
z_mat = joint_log_encoder(x_mat).µ
z_vec = z_mat[:, 1]
ρ_mat = z_mat
ρ_vec = ρ_mat[:, 1]
T = 0.5f0
λ = 0.1f0

## =============================================================================

# Define Metric MLP
metric_chain = RHVAEs.MetricChain(
    n_input,
    n_latent,
    metric_neurons,
    metric_activation,
    Flux.identity
)

@testset "MetriChain" begin
    @testset "Type checking" begin
        @test typeof(metric_chain) == RHVAEs.MetricChain
    end # type checking

    @testset "Forward Pass" begin
        @testset "Vector input" begin
            @testset "NamedTuple output" begin
                result = metric_chain(x_vector)

                @test isa(result, NamedTuple{(:diag, :lower)})
                @test all(isa.(values(result), AbstractVector{Float32}))
            end # NamedTuple output
            @testset "matrix output" begin
                result = metric_chain(x_vector, matrix=true)

                @test isa(result, AbstractMatrix{Float32})
                @test size(result) == (n_latent, n_latent)
            end # matrix output
        end # vector input

        @testset "Matrix input" begin
            @testset "NamedTuple output" begin
                result = metric_chain(x_matrix)

                @test isa(result, NamedTuple{(:diag, :lower)})
                @test all(isa.(values(result), AbstractMatrix{Float32}))
            end # NamedTuple output
            @testset "Matrix output" begin
                result = metric_chain(x_matrix, matrix=true)

                @test isa(result, AbstractArray{Float32,3})
                @test size(result) == (n_latent, n_latent, n_data)
            end # matrix output
        end # matrix input
    end # Forward Pass
end # MetricChain

## =============================================================================

@testset "update_metric function" begin
    for decoder in decoders
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )

        @testset "for decoder $decoder" begin
            result = RHVAEs.update_metric(rhvae)

            @test result.centroids_latent isa Matrix{Float32}
            @test result.M isa Array{Float32,3}
            @test result.T == rhvae.T
            @test result.λ == rhvae.λ

            # Add more specific tests if necessary
            @test size(result.centroids_latent, 2) == size(rhvae.centroids_data, 2)
        end # for decoder
    end # for decoders
end # update_metric function

## =============================================================================

# Build RHVAE example to be used in the following tests
exrhvae = RHVAEs.RHVAE(
    joint_log_encoder * joint_log_decoder,
    metric_chain,
    data,
    T,
    λ,
)
# Compute metric_param to be used in the following tests
metric_param = RHVAEs.update_metric(exrhvae)

## =============================================================================

@testset "G_inv function" begin
    @testset "vector input" begin
        # Compute inverse of metric tensor giving all parameters explicitly
        result = RHVAEs.G_inv(
            z_vec,
            exrhvae.centroids_latent,
            exrhvae.M,
            T,
            λ
        )

        @test isa(result, Matrix{Float32})
        @test size(result) == (length(z_vec), length(z_vec))

        # Check that the diagonal elements are greater than or equal to λ
        # This is because the function adds λI to the result, so the diagonal elements should be at least λ
        @test all(LinearAlgebra.diag(result) .≥ λ)

        # Check that the result is symmetric
        # This is because the metric tensor G should be symmetric
        @test result == transpose(result)
    end # vector input

    @testset "matrix input" begin
        # Compute inverse of metric tensor giving all parameters explicitly
        result = RHVAEs.G_inv(
            z_mat,
            exrhvae.centroids_latent,
            exrhvae.M,
            T,
            λ
        )

        @test isa(result, Array{Float32,3})
        @test size(result) == (size(z_mat, 1), size(z_mat, 1), n_data)

        # Check that the diagonal elements are greater than or equal to λ
        # This is because the function adds λI to the result, so the diagonal elements should be at least λ
        @test all(
            [all(LinearAlgebra.diag(x) .≥ λ) for x in eachslice(result, dims=3)]
        )

        # Check that the result is symmetric
        # This is because the metric tensor G should be symmetric
        @test result == Flux.batched_transpose(result)
    end # matrix input

    @testset "RHVAE as input" begin
        # Compute inverse of metric tensor giving an RHVAE
        result = RHVAEs.G_inv(
            z_vec,
            exrhvae,
        )

        @test isa(result, Matrix{Float32})
        @test size(result) == (length(z_vec), length(z_vec))

        # Check that the diagonal elements are greater than or equal to λ
        # This is because the function adds λI to the result, so the diagonal elements should be at least λ
        @test all(LinearAlgebra.diag(result) .≥ λ)

        # Check that the result is symmetric
        # This is because the metric tensor G should be symmetric
        @test result == transpose(result)
    end # RHVAE as input
end # G_inv function

## =============================================================================

@testset "metric_tensor function" begin
    @testset "vector input" begin
        # Compute metric tensor giving all parameters explicitly
        result = RHVAEs.metric_tensor(
            z_vec,
            exrhvae,
        )

        # Check that the result is a matrix
        @test isa(result, Matrix{Float32})
        # Check that the result has the correct dimensions
        @test size(result) == (length(z_vec), length(z_vec))
        # Check that the diagonal elements are less than or equal to 1/λ
        @test all(LinearAlgebra.diag(result) .≤ 1 / λ)
        # Check that the result is symmetric
        @test result == transpose(result)
    end

    @testset "matrix input" begin
        result = RHVAEs.metric_tensor(
            z_mat,
            exrhvae
        )

        # Check that the result is a 3D array
        @test isa(result, Array{Float32,3})
        # Check that the result has the correct dimensions
        @test size(result) == (size(z_mat, 1), size(z_mat, 1), n_data)
        # Check that the diagonal elements are less than or equal to 1/λ
        @test all(
            [all(LinearAlgebra.diag(x) .≤ 1 / λ) for x in eachslice(result, dims=3)]
        )
        # Check that the result is symmetric
        @test result == Flux.batched_transpose(result)
    end
end

## =============================================================================

@testset "riemannian_logprior function" begin
    @testset "vector input" begin
        # Compute G⁻¹ 
        Ginv = RHVAEs.G_inv(z_vec, exrhvae)
        # Compute logdetG
        logdetG = utils.slogdet(Ginv)
        # Compute Riemannian log-prior giving all parameters explicitly
        result = RHVAEs.riemannian_logprior(
            ρ_vec,
            Ginv,
            logdetG,
        )

        # Check that the result is a scalar
        @test isa(result, Float32)
        # Check that the result is not NaN
        @test !isnan(result)
        # Check that the result is not Inf
        @test !isinf(result)
    end

    @testset "matrix input" begin
        # Compute G⁻¹ 
        Ginv = RHVAEs.G_inv(z_mat, exrhvae)
        # Compute logdetG
        logdetG = utils.slogdet(Ginv)
        # Compute Riemannian log-prior giving all parameters explicitly
        result = RHVAEs.riemannian_logprior(
            ρ_mat,
            Ginv,
            logdetG,
        )

        # Check that the result is a vector
        @test isa(result, Vector{Float32})
        # Check that the result is not NaN
        @test all(!isnan, result)
        # Check that the result is not Inf
        @test all(!isinf, result)
    end
end

## =============================================================================

@testset "hamiltonian function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders

        @testset "vector input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute Hamiltonian giving all parameters explicitly
            result = RHVAEs.hamiltonian(
                x_vec, z_vec, ρ_vec, Ginv, logdetG, decoder, decoder(z_vec)
            )

            # Check that the result is a scalar
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end

        @testset "matrix input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute Hamiltonian giving all parameters explicitly
            result = RHVAEs.hamiltonian(
                x_mat, z_mat, ρ_mat, Ginv, logdetG, decoder, decoder(z_mat)
            )

            # Check that the result is a vector
            @test isa(result, Vector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute Hamiltonian using RHVAE
            result = RHVAEs.hamiltonian(x_vec, z_vec, ρ_vec, rhvae)

            # Check that the result is a scalar
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end # RHVAE as input
    end # for decoder
end # hamiltonian function

## =============================================================================

@testset "∇hamiltonian_finite function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders

        @testset "vector input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)

            # Test for var = :z
            result = RHVAEs.∇hamiltonian_finite(
                x_vec,
                z_vec,
                ρ_vec,
                Ginv,
                logdetG,
                decoder,
                decoder(z_vec),
                :z
            )

            # Check if the result is a vector of Float32
            @test isa(result, Vector{Float32})

            # Test for var = :ρ
            result = RHVAEs.∇hamiltonian_finite(
                x_vec,
                z_vec,
                ρ_vec,
                Ginv,
                logdetG,
                decoder,
                decoder(z_vec),
                :ρ
            )


            # Check if the result is a vector of Float32
            @test isa(result, Vector{Float32})
        end

        @testset "matrix input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)

            # Test for var = :z
            result = RHVAEs.∇hamiltonian_finite(
                x_mat,
                z_mat,
                ρ_mat,
                Ginv,
                logdetG,
                decoder,
                decoder(z_mat),
                :z
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})

            # Test for var = :ρ
            result = RHVAEs.∇hamiltonian_finite(
                x_mat,
                z_mat,
                ρ_mat,
                Ginv,
                logdetG,
                decoder,
                decoder(z_mat),
                :ρ
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})
        end

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Test for var = :z
            result = RHVAEs.∇hamiltonian_finite(
                x_mat,
                z_mat,
                ρ_mat,
                rhvae,
                :z
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})

            # Test for var = :ρ
            result = RHVAEs.∇hamiltonian_finite(
                x_mat,
                z_mat,
                ρ_mat,
                rhvae,
                :ρ
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})
        end # matrix input
    end # for decoder
end # ∇hamiltonian_finite function

## =============================================================================

@testset "∇hamiltonian_ForwardDiff function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders

        @testset "vector input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)

            # Test for var = :z
            result = RHVAEs.∇hamiltonian_ForwardDiff(
                x_vec,
                z_vec,
                ρ_vec,
                Ginv,
                logdetG,
                decoder,
                decoder(z_vec),
                :z
            )

            # Check if the result is a vector of Float32
            @test isa(result, Vector{Float32})

            # Test for var = :ρ
            result = RHVAEs.∇hamiltonian_ForwardDiff(
                x_vec,
                z_vec,
                ρ_vec,
                Ginv,
                logdetG,
                decoder,
                decoder(z_vec),
                :ρ
            )


            # Check if the result is a vector of Float32
            @test isa(result, Vector{Float32})
        end

        @testset "matrix input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)

            # Test for var = :z
            result = RHVAEs.∇hamiltonian_ForwardDiff(
                x_mat,
                z_mat,
                ρ_mat,
                Ginv,
                logdetG,
                decoder,
                decoder(z_mat),
                :z
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})

            # Test for var = :ρ
            result = RHVAEs.∇hamiltonian_ForwardDiff(
                x_mat,
                z_mat,
                ρ_mat,
                Ginv,
                logdetG,
                decoder,
                decoder(z_mat),
                :ρ
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})
        end

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Test for var = :z
            result = RHVAEs.∇hamiltonian_ForwardDiff(
                x_mat,
                z_mat,
                ρ_mat,
                rhvae,
                :z
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})

            # Test for var = :ρ
            result = RHVAEs.∇hamiltonian_ForwardDiff(
                x_mat,
                z_mat,
                ρ_mat,
                rhvae,
                :ρ
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})
        end # matrix input
    end # for decoder
end # ∇hamiltonian_finite function

## =============================================================================

# @testset "∇hamiltonian_TaylorDiff function" begin
#     # Loop over all decoders
#     @testset "$(decoder)" for decoder in decoders

#         @testset "vector input" begin
#             # Compute G⁻¹ 
#             Ginv = RHVAEs.G_inv(z_vec, exrhvae)
#             # Compute logdetG
#             logdetG = utils.slogdet(Ginv)

#             # Test for var = :z
#             result = RHVAEs.∇hamiltonian_TaylorDiff(
#                 x_vec,
#                 z_vec,
#                 ρ_vec,
#                 Ginv,
#                 logdetG,
#                 decoder,
#                 decoder(z_vec),
#                 :z
#             )

#             # Check if the result is a vector of Float32
#             @test isa(result, Vector{Float32})

#             # Test for var = :ρ
#             result = RHVAEs.∇hamiltonian_TaylorDiff(
#                 x_vec,
#                 z_vec,
#                 ρ_vec,
#                 Ginv,
#                 logdetG,
#                 decoder,
#                 decoder(z_vec),
#                 :ρ
#             )


#             # Check if the result is a vector of Float32
#             @test isa(result, Vector{Float32})
#         end

#         @testset "matrix input" begin
#             # Compute G⁻¹ 
#             Ginv = RHVAEs.G_inv(z_mat, exrhvae)
#             # Compute logdetG
#             logdetG = utils.slogdet(Ginv)

#             # Test for var = :z
#             result = RHVAEs.∇hamiltonian_TaylorDiff(
#                 x_mat,
#                 z_mat,
#                 ρ_mat,
#                 Ginv,
#                 logdetG,
#                 decoder,
#                 decoder(z_mat),
#                 :z
#             )

#             # Check if the result is a matrix of Float32
#             @test isa(result, Matrix{Float32})

#             # Test for var = :ρ
#             result = RHVAEs.∇hamiltonian_TaylorDiff(
#                 x_mat,
#                 z_mat,
#                 ρ_mat,
#                 Ginv,
#                 logdetG,
#                 decoder,
#                 decoder(z_mat),
#                 :ρ
#             )

#             # Check if the result is a matrix of Float32
#             @test isa(result, Matrix{Float32})
#         end

#         @testset "RHVAE as input" begin
#             # Build temporary RHVAE
#             rhvae = RHVAEs.RHVAE(
#                 joint_log_encoder * decoder,
#                 metric_chain,
#                 data,
#                 T,
#                 λ,
#             )

#             # Test for var = :z
#             result = RHVAEs.∇hamiltonian_TaylorDiff(
#                 x_mat,
#                 z_mat,
#                 ρ_mat,
#                 rhvae,
#                 :z
#             )

#             # Check if the result is a matrix of Float32
#             @test isa(result, Matrix{Float32})

#             # Test for var = :ρ
#             result = RHVAEs.∇hamiltonian_TaylorDiff(
#                 x_mat,
#                 z_mat,
#                 ρ_mat,
#                 rhvae,
#                 :ρ
#             )

#             # Check if the result is a matrix of Float32
#             @test isa(result, Matrix{Float32})
#         end # matrix input
#     end # for decoder
# end # ∇hamiltonian_finite function

## =============================================================================

@testset "_leapfrog_first_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute _leapfrog_first_step giving all parameters explicitly
            result = RHVAEs._leapfrog_first_step(
                x_vec, z_vec, ρ_vec, Ginv, logdetG, decoder, decoder(z_vec)
            )
            # Test that the result is a vector
            @test isa(result, AbstractVector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # vector input

        @testset "matrix input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute _leapfrog_first_step giving all parameters explicitly
            result = RHVAEs._leapfrog_first_step(
                x_mat, z_mat, ρ_mat, Ginv, logdetG, decoder, decoder(z_mat)
            )
            # Test that the result is a matrix
            @test isa(result, AbstractMatrix{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # matrix input

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )
            # Compute _leapfrog_first_step using RHVAE
            result = RHVAEs._leapfrog_first_step(x_vec, z_vec, ρ_vec, rhvae)
            # Test that the result is a vector
            @test isa(result, AbstractVector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # RHVAE as input
    end # for decoder
end # _leapfrog_first_step function

## =============================================================================

@testset "_leapfrog_second_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute _leapfrog_first_step giving all parameters explicitly
            result = RHVAEs._leapfrog_second_step(
                x_vec, z_vec, ρ_vec, Ginv, logdetG, decoder, decoder(z_vec)
            )
            # Test that the result is a vector
            @test isa(result, AbstractVector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # vector input

        @testset "matrix input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute _leapfrog_first_step giving all parameters explicitly
            result = RHVAEs._leapfrog_second_step(
                x_mat, z_mat, ρ_mat, Ginv, logdetG, decoder, decoder(z_mat)
            )
            # Test that the result is a matrix
            @test isa(result, AbstractMatrix{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # matrix input

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )
            # Compute _leapfrog_first_step using RHVAE
            result = RHVAEs._leapfrog_second_step(x_vec, z_vec, ρ_vec, rhvae)
            # Test that the result is a vector
            @test isa(result, AbstractVector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # RHVAE as input
    end # for decoder
end # _leapfrog_second_step function

## =============================================================================

@testset "_leapfrog_third_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute _leapfrog_first_step giving all parameters explicitly
            result = RHVAEs._leapfrog_third_step(
                x_vec, z_vec, ρ_vec, Ginv, logdetG, decoder, decoder(z_vec)
            )
            # Test that the result is a vector
            @test isa(result, AbstractVector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # vector input

        @testset "matrix input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute _leapfrog_first_step giving all parameters explicitly
            result = RHVAEs._leapfrog_third_step(
                x_mat, z_mat, ρ_mat, Ginv, logdetG, decoder, decoder(z_mat)
            )
            # Test that the result is a matrix
            @test isa(result, AbstractMatrix{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # matrix input

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )
            # Compute _leapfrog_first_step using RHVAE
            result = RHVAEs._leapfrog_third_step(x_vec, z_vec, ρ_vec, rhvae)
            # Test that the result is a vector
            @test isa(result, AbstractVector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # RHVAE as input
    end # for decoder
end # _leapfrog_third_step function

## =============================================================================

@testset "general_leapfrog_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector inputs" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute general_leapfrog_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_step(
                x_vec, z_vec, ρ_vec, Ginv, logdetG, decoder, decoder(z_vec),
                metric_param
            )
            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    AbstractVector{Float32},
                    AbstractVector{Float32},
                    AbstractMatrix{Float32},
                    Float32,
                    NamedTuple,
                }
            )

            # Check that the result is not NaN
            @test all(!isnan, result[1])
            @test all(!isnan, result[2])

            # Check that the result is not Inf
            @test all(!isinf, result[1])
            @test all(!isinf, result[2])
        end # vector inputs

        @testset "matrix inputs" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute general_leapfrog_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_step(
                x_mat, z_mat, ρ_mat, Ginv, logdetG, decoder, decoder(z_mat),
                metric_param
            )
            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    AbstractMatrix{Float32},
                    AbstractMatrix{Float32},
                    AbstractArray{Float32,3},
                    AbstractVector{Float32},
                    NamedTuple,
                }
            )

            # Check that the result is not NaN
            @test all(!isnan, result[1])
            @test all(!isnan, result[2])

            # Check that the result is not Inf
            @test all(!isinf, result[1])
            @test all(!isinf, result[2])
        end # matrix inputs

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute general_leapfrog_step using RHVAE
            result = RHVAEs.general_leapfrog_step(x_vec, z_vec, ρ_vec, rhvae)

            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    AbstractVector{Float32},
                    AbstractVector{Float32},
                    AbstractMatrix{Float32},
                    Float32,
                    NamedTuple,
                }
            )

            # Check that the result is not NaN
            @test all(!isnan, result[1])
            @test all(!isnan, result[2])

            # Check that the result is not Inf
            @test all(!isinf, result[1])
            @test all(!isinf, result[2])
        end
    end # for decoder
end # general_leapfrog_step function

## =============================================================================

@testset "general_leapfrog_tempering_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector inputs" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_vec, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute general_leapfrog_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_tempering_step(
                x_vec, z_vec, Ginv, logdetG, decoder, decoder(z_vec),
                metric_param
            )
            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    NamedTuple,
                    NamedTuple,
                }
            )
        end # vector inputs

        @testset "matrix input" begin
            # Compute G⁻¹ 
            Ginv = RHVAEs.G_inv(z_mat, exrhvae)
            # Compute logdetG
            logdetG = utils.slogdet(Ginv)
            # Compute general_leapfrog_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_tempering_step(
                x_mat, z_mat, Ginv, logdetG, decoder, decoder(z_mat),
                metric_param
            )
            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    NamedTuple,
                    NamedTuple,
                }
            )
        end # matrix inputs

        @testset "RHVAE as input" begin
            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute general_leapfrog_step using RHVAE
            result = RHVAEs.general_leapfrog_tempering_step(x_vec, z_vec, rhvae)

            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    NamedTuple,
                    NamedTuple,
                }
            )
        end
    end # for decoder
end # general_leapfrog_tempreing_step function

## =============================================================================

@testset "RHVAE forward pass" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary RHVAE
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )

        @testset "explicit metric_param" begin
            @testset "latent = false" begin
                @testset "vector inputs" begin
                    # Run RHVAE with vector inputs
                    result = rhvae(x_vec, metric_param)
                    # Check that the result is a NamedTuple
                    @test isa(result, NamedTuple)
                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat, metric_param)
                    # Check that the result is a NamedTuple
                    @test isa(result, NamedTuple)
                end # matrix inputs
            end # latent = false

            @testset "latent = true" begin
                @testset "vector inputs" begin
                    # Run RHVAE with vector inputs
                    result = rhvae(x_vec, metric_param; latent=true)
                    # Check that the result is a NamedTuple
                    @test isa(
                        result, NamedTuple{(:encoder, :decoder, :phase_space),}
                    )
                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat, metric_param; latent=true)
                    # Check that the result is a NamedTuple
                    @test isa(
                        result, NamedTuple{(:encoder, :decoder, :phase_space)}
                    )
                end # matrix inputs
            end # latent = false
        end # explicit metric_param

        @testset "implicit metric_param" begin
            @testset "latent = false" begin
                @testset "vector inputs" begin
                    # Run RHVAE with vector inputs
                    result = rhvae(x_vec)
                    # Check that the result is a NamedTuple
                    @test isa(result, NamedTuple)
                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat)
                    # Check that the result is a NamedTuple
                    @test isa(result, NamedTuple)
                end # matrix inputs
            end # latent = false

            @testset "latent = true" begin
                @testset "vector inputs" begin
                    # Run RHVAE with vector inputs
                    result = rhvae(x_vec; latent=true)
                    # Check that the result is a NamedTuple
                    @test isa(
                        result, NamedTuple{(:encoder, :decoder, :phase_space)}
                    )
                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat; latent=true)
                    # Check that the result is a NamedTuple
                    @test isa(
                        result, NamedTuple{(:encoder, :decoder, :phase_space)}
                    )
                end # matrix inputs
            end # latent = false
        end # implicit metric_param
    end # for decoder
end # RHVAE function

## =============================================================================

@testset "_log_p̄ function" begin
    # Loop through all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary RHVAE
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )

        @testset "vector input" begin
            # Define hvae_outputs
            rhvae_outputs = rhvae(x_vec, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_p̄(x_vec, rhvae, rhvae_outputs)
            # Check that the result is a Float32
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Define hvae_outputs
            rhvae_outputs = rhvae(x_mat, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_p̄(x_mat, rhvae, rhvae_outputs)
            # Check that the result is a Vector{Float32}
            @test isa(result, Vector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # matrix input
    end # for decoder
end # _log_p̄ function

## =============================================================================

@testset "_log_q̄ function" begin
    # Loop through all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary RHVAE
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )
        # Define βₒ
        β = 0.5f0

        @testset "vector input" begin
            # Define rhvae_outputs
            rhvae_outputs = rhvae(x_vec, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_q̄(rhvae, rhvae_outputs, β)
            # Check that the result is a Float32
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Define rhvae_outputs
            rhvae_outputs = rhvae(x_mat, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_q̄(rhvae, rhvae_outputs, β)
            # Check that the result is a Vector{Float32}
            @test isa(result, Vector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end # matrix input
    end # for decoder
end # _log_q̄ function

## =============================================================================

@testset "riemannian_hamiltonian_elbo function" begin
    # Loop through all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary RHVAE
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )

        @testset "single x_in" begin
            @testset "vector input with pre-computed metric_param" begin
                # Compute riemannian_hamiltonian_elbo
                result = RHVAEs.riemannian_hamiltonian_elbo(
                    rhvae, metric_param, x_vec
                )
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input with pre-computed metric_param" begin
                # Compute riemannian_hamiltonian_elbo
                result = RHVAEs.riemannian_hamiltonian_elbo(
                    rhvae, metric_param, x_mat
                )
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # matrix input

            @testset "without pre-computed metric_param" begin
                # Compute riemannian_hamiltonian_elbo
                result = RHVAEs.riemannian_hamiltonian_elbo(rhvae, x_mat)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)

            end # without pre-computed metric_param
        end # single x_in

        @testset "x_in and x_out" begin
            @testset "vector input with pre-computed metric_param" begin
                # Compute riemannian_hamiltonian_elbo
                result = RHVAEs.riemannian_hamiltonian_elbo(
                    rhvae, metric_param, x_vec, x_vec,
                )
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input with pre-computed metric_param" begin
                # Compute riemannian_hamiltonian_elbo
                result = RHVAEs.riemannian_hamiltonian_elbo(
                    rhvae, metric_param, x_mat, x_mat
                )
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # matrix input

            @testset "without pre-computed metric_param" begin
                # Compute riemannian_hamiltonian_elbo
                result = RHVAEs.riemannian_hamiltonian_elbo(rhvae, x_mat, x_mat)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)

            end # without pre-computed metric_param
        end # x_in and x_out
    end # for decoder
end # riemannian_hamiltonian_elbo function

## =============================================================================

@testset "loss function" begin
    # Loop through all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary RHVAE
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )

        @testset "single x_in" begin
            @testset "vector input" begin
                # Compute loss
                result = RHVAEs.loss(rhvae, x_vec)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input" begin
                # Compute loss
                result = RHVAEs.loss(rhvae, x_mat)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # matrix input
        end # single x_in

        @testset "x_in and x_out" begin
            @testset "vector input" begin
                # Compute loss
                result = RHVAEs.loss(rhvae, x_vec, x_vec)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input" begin
                # Compute loss
                result = RHVAEs.loss(rhvae, x_mat, x_mat)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # matrix input
        end # single x_in
    end # for decoder
end # loss function

## =============================================================================

@testset "RHVAE gradient" begin
    @testset "without regularization" begin
        # Loop through decoders
        for decoder in decoders
            # Build temporary HVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            @testset "gradient (same input and output)" begin
                grads = Flux.gradient(rhvae -> RHVAEs.loss(rhvae, x_vec), rhvae)
                @test isa(grads[1], NamedTuple)
            end # @testset "gradient (same input and output)"

            @testset "gradient (different input and output)" begin
                grads = Flux.gradient(
                    rhvae -> RHVAEs.loss(rhvae, x_vec, x_vec), rhvae
                )
                @test isa(grads[1], NamedTuple)
            end # @testset "gradient (different input and output)"

        end # for decoder in decoders
    end # @testset "without regularization"
end # @testset "HVAE grad"

## =============================================================================

# NOTE: The following tests are commented out because they fail with GitHub
# Actions with the following error:
# Got exception outside of a @test
# BoundsError: attempt to access 16-element Vector{UInt8} at index [0]

# @testset "RHVAE training" begin
#     @testset "without regularization" begin
#         # Loop through decoders
#         for decoder in decoders
#             # Build temporary RHVAE
#             rhvae = RHVAEs.RHVAE(
#                 joint_log_encoder * decoder,
#                 metric_chain,
#                 data,
#                 T,
#                 λ,
#             )

#             # Explicit setup of optimizer
#             opt_state = Flux.Train.setup(
#                 Flux.Optimisers.Adam(1E-2),
#                 rhvae
#             )

#             # Test training function
#             L = RHVAEs.train!(rhvae, data, opt_state; loss_return=true)
#             @test isa(L, Float32)
#         end # for decoder in decoders
#     end # @testset "without regularization"
# end # @testset "RHVAE training"

println("\nAll tests passed!\n")