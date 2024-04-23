##
println("\nTesting HVAEs module:\n")
##

# Import AutoEncode.jl module to be tested
using AutoEncode.HVAEs
import AutoEncode.VAEs
import AutoEncode.utils
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

# Define data and parameters to use througout the tests
x_mat = data
x_vec = @view data[:, 1]
z_mat = joint_log_encoder(x_mat).µ
z_vec = @view z_mat[:, 1]
ρ_mat = z_mat
ρ_vec = @view ρ_mat[:, 1]
T = 0.5f0
λ = 0.1f0


## =============================================================================

# Build HVAE example to be used in the following tests
exhvae = HVAEs.HVAE(
    joint_log_encoder * joint_log_decoder,
)

## =============================================================================

@testset "potential energy function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders

        @testset "vector input" begin

            # Compute Hamiltonian giving all parameters explicitly
            result = HVAEs.potential_energy(
                x_vec, z_vec, decoder, decoder(z_vec)
            )

            # Check that the result is a scalar
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end

        @testset "matrix input" begin
            # Compute Hamiltonian giving all parameters explicitly
            result = HVAEs.potential_energy(
                x_mat, z_mat, decoder, decoder(z_mat)
            )

            # Check that the result is a vector
            @test isa(result, Vector{Float32})
            # Check that the result is not NaN
            @test all(!isnan, result)
            # Check that the result is not Inf
            @test all(!isinf, result)
        end

        @testset "HVAE as input" begin
            # Build temporary HVAE
            hvae = HVAEs.HVAE(
                joint_log_encoder * decoder,
            )

            # Compute Hamiltonian using HVAE
            result = HVAEs.potential_energy(x_vec, z_vec, hvae)

            # Check that the result is a scalar
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end # HVAE as input
    end # for decoder
end # hamiltonian function

## =============================================================================

@testset "∇potential_energy_finite function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders

        @testset "vector input" begin
            result = HVAEs.∇potential_energy_finite(
                x_vec,
                z_vec,
                decoder,
                decoder(z_vec),
            )

            # Check if the result is a vector of Float32
            @test isa(result, Vector{Float32})
        end

        @testset "matrix input" begin
            result = HVAEs.∇potential_energy_finite(
                x_mat,
                z_mat,
                decoder,
                decoder(z_mat),
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})
        end

        @testset "HVAE as input" begin
            # Build temporary HVAE
            hvae = HVAEs.HVAE(
                joint_log_encoder * decoder,
            )
            result = HVAEs.∇potential_energy_finite(
                x_mat,
                z_mat,
                hvae,
            )

            # Check if the result is a matrix of Float32
            @test isa(result, Matrix{Float32})
        end # matrix input
    end # for decoder
end # ∇potential_energy_finite function

## =============================================================================

# @testset "∇potential_energy_TaylorDiff function" begin
#     # Loop over all decoders
#     @testset "$(decoder)" for decoder in decoders

#         @testset "vector input" begin
#             # Test for var = :z
#             result = HVAEs.∇potential_energy_TaylorDiff(
#                 x_vec,
#                 z_vec,
#                 decoder,
#                 decoder(z_vec),
#             )

#             # Check if the result is a vector of Float32
#             @test isa(result, Vector{Float32})
#         end

#         @testset "matrix input" begin
#             # Test for var = :z
#             result = HVAEs.∇potential_energy_TaylorDiff(
#                 x_mat,
#                 z_mat,
#                 decoder,
#                 decoder(z_mat),
#             )

#             # Check if the result is a matrix of Float32
#             @test isa(result, Matrix{Float32})
#         end

#         @testset "HVAE as input" begin
#             # Build temporary HVAE
#             hvae = HVAEs.HVAE(
#                 joint_log_encoder * decoder,
#             )

#             result = HVAEs.∇potential_energy_TaylorDiff(
#                 x_mat,
#                 z_mat,
#                 hvae,
#             )

#             # Check if the result is a matrix of Float32
#             @test isa(result, Matrix{Float32})
#         end # matrix input
#     end # for decoder
# end # ∇potential_energy_finite function

## =============================================================================

@testset "leapfrog_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary HVAE
        hvae = HVAEs.HVAE(
            joint_log_encoder * decoder,
        )
        @testset "vector inputs" begin
            # Compute leapfrog_step giving all parameters explicitly
            result = HVAEs.leapfrog_step(
                x_vec, z_vec, ρ_vec, hvae
            )
            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    AbstractVector{Float32},
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
        end # vector inputs

        @testset "matrix inputs" begin

            # Compute leapfrog_step giving all parameters explicitly
            result = HVAEs.leapfrog_step(
                x_mat, z_mat, ρ_mat, hvae
            )
            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    AbstractMatrix{Float32},
                    AbstractMatrix{Float32},
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

        @testset "HVAE as input" begin
            # Build temporary HVAE
            hvae = HVAEs.HVAE(
                joint_log_encoder * decoder,
            )

            # Compute leapfrog_step using HVAE
            result = HVAEs.leapfrog_step(x_vec, z_vec, ρ_vec, hvae)

            # Test that the result is a tuple of vectors
            @test isa(
                result,
                Tuple{
                    AbstractVector{Float32},
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
        end
    end # for decoder
end # leapfrog_step function

## =============================================================================

@testset "quadratic_tempering tests" begin
    @test HVAEs.quadratic_tempering(0.3f0, 1, 3) isa Float32
end

@testset "null_tempering tests" begin
    @test HVAEs.null_tempering(0.3f0, 1, 3) isa Float32
end

## =============================================================================

@testset "leapfrog_tempering_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector inputs" begin
            # Compute leapfrog_step giving all parameters explicitly
            result = HVAEs.leapfrog_tempering_step(
                x_vec, z_vec, decoder, decoder(z_vec),
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
            # Compute leapfrog_step giving all parameters explicitly
            result = HVAEs.leapfrog_tempering_step(
                x_mat, z_mat, decoder, decoder(z_mat),
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

        @testset "HVAE as input" begin
            # Build temporary HVAE
            hvae = HVAEs.HVAE(
                joint_log_encoder * decoder,
            )

            # Compute leapfrog_step using HVAE
            result = HVAEs.leapfrog_tempering_step(x_vec, z_vec, hvae)

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
end # leapfrog_tempreing_step function

## =============================================================================

@testset "HVAE forward pass" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary HVAE
        hvae = HVAEs.HVAE(
            joint_log_encoder * decoder,
        )

        @testset "latent = false" begin
            @testset "vector inputs" begin
                # Run HVAE with vector inputs
                result = hvae(x_vec)
                # Check that the result is a NamedTuple
                @test isa(result, NamedTuple)
            end # vector inputs

            @testset "matrix inputs" begin
                # Run HVAE with matrix inputs
                result = hvae(x_mat)
                # Check that the result is a NamedTuple
                @test isa(result, NamedTuple)
            end # matrix inputs
        end # latent = false

        @testset "latent = true" begin
            @testset "vector inputs" begin
                # Run HVAE with vector inputs
                result = hvae(x_vec; latent=true)
                # Check that the result is a NamedTuple
                @test isa(
                    result, NamedTuple{(:encoder, :decoder, :phase_space),}
                )
            end # vector inputs

            @testset "matrix inputs" begin
                # Run HVAE with matrix inputs
                result = hvae(x_mat; latent=true)
                # Check that the result is a NamedTuple
                @test isa(
                    result, NamedTuple{(:encoder, :decoder, :phase_space)}
                )
            end # matrix inputs
        end # latent = false
    end # for decoder
end # HVAE function

## =============================================================================

@testset "_log_p̄ function" begin
    # Loop through all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary HVAE
        hvae = HVAEs.HVAE(
            joint_log_encoder * decoder,
        )

        @testset "vector input" begin
            # Define hvae_outputs
            hvae_outputs = hvae(x_vec; latent=true)

            # Compute _log_p̄
            result = HVAEs._log_p̄(x_vec, hvae, hvae_outputs)
            # Check that the result is a Float32
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Define hvae_outputs
            hvae_outputs = hvae(x_mat; latent=true)

            # Compute _log_p̄
            result = HVAEs._log_p̄(x_mat, hvae, hvae_outputs)
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
        # Build temporary HVAE
        hvae = HVAEs.HVAE(
            joint_log_encoder * decoder,
        )
        # Define βₒ
        β = 0.5f0

        @testset "vector input" begin
            # Define hvae_outputs
            hvae_outputs = hvae(x_vec; latent=true)

            # Compute _log_p̄
            result = HVAEs._log_q̄(hvae, hvae_outputs, β)
            # Check that the result is a Float32
            @test isa(result, Float32)
            # Check that the result is not NaN
            @test !isnan(result)
            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Define hvae_outputs
            hvae_outputs = hvae(x_mat; latent=true)

            # Compute _log_p̄
            result = HVAEs._log_q̄(hvae, hvae_outputs, β)
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

@testset "hamiltonian_elbo function" begin
    # Loop through all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary HVAE
        hvae = HVAEs.HVAE(
            joint_log_encoder * decoder,
        )

        @testset "single x_in" begin
            @testset "vector input" begin
                # Compute hamiltonian_elbo
                result = HVAEs.hamiltonian_elbo(
                    hvae, x_vec
                )
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input" begin
                # Compute hamiltonian_elbo
                result = HVAEs.hamiltonian_elbo(
                    hvae, x_mat
                )
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
                # Compute hamiltonian_elbo
                result = HVAEs.hamiltonian_elbo(
                    hvae, x_vec, x_vec,
                )
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input" begin
                # Compute hamiltonian_elbo
                result = HVAEs.hamiltonian_elbo(
                    hvae, x_mat, x_mat
                )
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # matrix input
        end # x_in and x_out
    end # for decoder
end # hamiltonian_elbo function

## =============================================================================

@testset "loss function" begin
    # Loop through all decoders
    @testset "$(decoder)" for decoder in decoders
        # Build temporary HVAE
        hvae = HVAEs.HVAE(
            joint_log_encoder * decoder,
        )

        @testset "single x_in" begin
            @testset "vector input" begin
                # Compute loss
                result = HVAEs.loss(hvae, x_vec)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input" begin
                # Compute loss
                result = HVAEs.loss(hvae, x_mat)
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
                result = HVAEs.loss(hvae, x_vec, x_vec)
                # Check that the result is a Float32
                @test isa(result, Float32)
                # Check that the result is not NaN
                @test !isnan(result)
                # Check that the result is not Inf
                @test !isinf(result)
            end # vector input

            @testset "matrix input" begin
                # Compute loss
                result = HVAEs.loss(hvae, x_mat, x_mat)
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

@testset "HVAE training" begin
    @testset "without regularization" begin
        # Loop through decoders
        for decoder in decoders
            # Build temporary HVAE
            hvae = HVAEs.HVAE(
                joint_log_encoder * decoder,
            )

            # Explicit setup of optimizer
            opt_state = Flux.Train.setup(
                Flux.Optimisers.Adam(1E-2),
                hvae
            )

            # Test training function
            L = HVAEs.train!(hvae, data, opt_state; loss_return=true)
            @test isa(L, Float32)
        end # for decoder in decoders
    end # @testset "without regularization"
end # @testset "HVAE training"

println("\nAll tests passed!\n")