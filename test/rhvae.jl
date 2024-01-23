##
println("Testing RHVAEs module:\n")
##

# Import AutoEncode.jl module to be tested
using AutoEncode.RHVAEs
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

metric_neurons = repeat([n_neuron], n_hidden)
metric_activation = repeat([Flux.swish], n_hidden)
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

# Define Metric MLP
metric_chain = RHVAEs.MetricChain(
    n_input,
    n_latent,
    metric_neurons,
    metric_activation,
    Flux.identity
)

@testset "Type checking" begin
    @test typeof(metric_chain) == RHVAEs.MetricChain
end # type checking

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

@testset "G_inv function" begin
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
    # @test all(diag(result) .≥ λ)

    # Check that the result is symmetric
    # This is because the metric tensor G should be symmetric
    @test result == transpose(result)

    # Compute inverse of metric tensor giving an RHVAE
    result = RHVAEs.G_inv(
        z_vec,
        exrhvae,
    )

    @test isa(result, Matrix{Float32})
    @test size(result) == (length(z_vec), length(z_vec))

    # Check that the diagonal elements are greater than or equal to λ
    # This is because the function adds λI to the result, so the diagonal elements should be at least λ
    # @test all(diag(result) .≥ λ)

    # Check that the result is symmetric
    # This is because the metric tensor G should be symmetric
    @test result == transpose(result)
end # G_inv function

## =============================================================================

@testset "riemannian_logprior function" begin
    # Compute Riemannian log-prior 
    result = RHVAEs.riemannian_logprior(z_vec, ρ_vec, metric_param)

    @test isa(result, Float32)

    # Check that the result is less than or equal to 0
    # This is because the log-prior of a Gaussian distribution should be less than or equal to 0
    @test result ≤ 0

    # Check that the result is not NaN
    @test !isnan(result)

    # Check that the result is not Inf
    @test !isinf(result)
end

## =============================================================================

@testset "hamiltonian function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        # Compute Hamiltonian giving all parameters explicitly
        result = RHVAEs.hamiltonian(x_vec, z_vec, ρ_vec, decoder, metric_param)

        @test isa(result, Float32)

        # Check that the result is not NaN
        @test !isnan(result)

        # Check that the result is not Inf
        @test !isinf(result)

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

        @test isa(result, Float32)

        # Check that the result is not NaN
        @test !isnan(result)

        # Check that the result is not Inf
        @test !isinf(result)
    end # for decoder
end # hamiltonian function

## =============================================================================

@testset "∇hamiltonian function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders

        @testset "Derivatives with respect to :z" begin
            # Compute ∇hamiltonian giving all parameters explicitly
            result = RHVAEs.∇hamiltonian(
                x_vec, z_vec, ρ_vec, decoder, metric_param, :z
            )

            @test isa(result, AbstractVector{Float32})

            # Check that the result is not NaN
            @test all(!isnan, result)

            # Check that the result is not Inf
            @test all(!isinf, result)

            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute ∇hamiltonian using RHVAE
            result = RHVAEs.∇hamiltonian(x_vec, z_vec, ρ_vec, rhvae, :z)

            @test isa(result, AbstractVector{Float32})

            # Check that the result is not NaN
            @test all(!isnan, result)

            # Check that the result is not Inf
            @test all(!isinf, result)
        end # Derivatives with respect to :z

        @testset "Derivatives with respect to :ρ" begin
            # Compute ∇hamiltonian giving all parameters explicitly
            result = RHVAEs.∇hamiltonian(
                x_vec, z_vec, ρ_vec, decoder, metric_param, :ρ
            )

            @test isa(result, AbstractVector{Float32})

            # Check that the result is not NaN
            @test all(!isnan, result)

            # Check that the result is not Inf
            @test all(!isinf, result)

            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute ∇hamiltonian using RHVAE
            result = RHVAEs.∇hamiltonian(x_vec, z_vec, ρ_vec, rhvae, :ρ)

            @test isa(result, AbstractVector{Float32})

            # Check that the result is not NaN
            @test all(!isnan, result)

            # Check that the result is not Inf
            @test all(!isinf, result)
        end # Derivatives with respect to :ρ
    end # for decoder
end # ∇hamiltonian function

## =============================================================================

@testset "_leapfrog_first_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        # Compute _leapfrog_first_step giving all parameters explicitly
        result = RHVAEs._leapfrog_first_step(
            x_vec, z_vec, ρ_vec, decoder, metric_param
        )

        @test isa(result, AbstractVector{Float32})

        # Check that the result is not NaN
        @test all(!isnan, result)

        # Check that the result is not Inf
        @test all(!isinf, result)

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

        @test isa(result, AbstractVector{Float32})

        # Check that the result is not NaN
        @test all(!isnan, result)

        # Check that the result is not Inf
        @test all(!isinf, result)
    end # for decoder
end # _leapfrog_first_step function

## =============================================================================

@testset "_leapfrog_second_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        # Compute _leapfrog_second_step giving all parameters explicitly
        result = RHVAEs._leapfrog_second_step(
            x_vec, z_vec, ρ_vec, decoder, metric_param
        )

        @test isa(result, AbstractVector{Float32})

        # Check that the result is not NaN
        @test all(!isnan, result)

        # Check that the result is not Inf
        @test all(!isinf, result)

        # Build temporary RHVAE
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )

        # Compute _leapfrog_second_step using RHVAE
        result = RHVAEs._leapfrog_second_step(x_vec, z_vec, ρ_vec, rhvae)

        @test isa(result, AbstractVector{Float32})

        # Check that the result is not NaN
        @test all(!isnan, result)

        # Check that the result is not Inf
        @test all(!isinf, result)
    end # for decoder
end # _leapfrog_second_step function

## =============================================================================

@testset "_leapfrog_third_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        # Compute _leapfrog_third_step giving all parameters explicitly
        result = RHVAEs._leapfrog_third_step(
            x_vec, z_vec, ρ_vec, decoder, metric_param
        )

        @test isa(result, AbstractVector{Float32})

        # Check that the result is not NaN
        @test all(!isnan, result)

        # Check that the result is not Inf
        @test all(!isinf, result)

        # Build temporary RHVAE
        rhvae = RHVAEs.RHVAE(
            joint_log_encoder * decoder,
            metric_chain,
            data,
            T,
            λ,
        )

        # Compute _leapfrog_third_step using RHVAE
        result = RHVAEs._leapfrog_third_step(x_vec, z_vec, ρ_vec, rhvae)

        @test isa(result, AbstractVector{Float32})

        # Check that the result is not NaN
        @test all(!isnan, result)

        # Check that the result is not Inf
        @test all(!isinf, result)
    end # for decoder
end # _leapfrog_third_step function

## =============================================================================

@testset "general_leapfrog_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector inputs" begin
            # Compute general_leapfrog_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_step(
                x_vec, z_vec, ρ_vec, decoder, metric_param
            )

            @test isa(
                result, Tuple{AbstractVector{Float32},AbstractVector{Float32}}
            )

            # Check that the result is not NaN
            @test all(!isnan, result[1])
            @test all(!isnan, result[2])

            # Check that the result is not Inf
            @test all(!isinf, result[1])
            @test all(!isinf, result[2])

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

            @test isa(
                result, Tuple{AbstractVector{Float32},AbstractVector{Float32}}
            )

            # Check that the result is not NaN
            @test all(!isnan, result[1])
            @test all(!isnan, result[2])

            # Check that the result is not Inf
            @test all(!isinf, result[1])
            @test all(!isinf, result[2])
        end # vector inputs

        @testset "matrix inputs" begin
            # Compute general_leapfrog_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_step(
                x_mat, z_mat, ρ_mat, decoder, metric_param
            )

            @test isa(
                result, Tuple{AbstractMatrix{Float32},AbstractMatrix{Float32}}
            )

            # Check that the result is not NaN
            @test all(!isnan, result[1])
            @test all(!isnan, result[2])

            # Check that the result is not Inf
            @test all(!isinf, result[1])
            @test all(!isinf, result[2])

            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute general_leapfrog_step using RHVAE
            result = RHVAEs.general_leapfrog_step(x_mat, z_mat, ρ_mat, rhvae)

            @test isa(
                result, Tuple{AbstractMatrix{Float32},AbstractMatrix{Float32}}
            )

            # Check that the result is not NaN
            @test all(!isnan, result[1])
            @test all(!isnan, result[2])

            # Check that the result is not Inf
            @test all(!isinf, result[1])
            @test all(!isinf, result[2])
        end # matrix inputs
    end # for decoder
end # general_leapfrog_step function

## =============================================================================

@testset "general_leapfrog_tempering_step function" begin
    # Loop over all decoders
    @testset "$(decoder)" for decoder in decoders
        @testset "vector inputs" begin
            # Compute general_leapfrog_tempering_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_tempering_step(
                x_vec, z_vec, decoder, metric_param
            )

            @test isa(result, NamedTuple)

            # Check that the result is not NaN
            @test all(!isnan, result.z_init)
            @test all(!isnan, result.ρ_init)
            @test all(!isnan, result.z_final)
            @test all(!isnan, result.ρ_final)

            # Check that the result is not Inf
            @test all(!isinf, result.z_init)
            @test all(!isinf, result.ρ_init)
            @test all(!isinf, result.z_final)
            @test all(!isinf, result.ρ_final)

            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute general_leapfrog_tempering_step using RHVAE
            result = RHVAEs.general_leapfrog_tempering_step(x_vec, z_vec, rhvae)

            @test isa(result, NamedTuple)

            # Check that the result is not NaN
            @test all(!isnan, result.z_init)
            @test all(!isnan, result.ρ_init)
            @test all(!isnan, result.z_final)
            @test all(!isnan, result.ρ_final)

            # Check that the result is not Inf
            @test all(!isinf, result.z_init)
            @test all(!isinf, result.ρ_init)
            @test all(!isinf, result.z_final)
            @test all(!isinf, result.ρ_final)
        end # vector inputs

        @testset "matrix inputs" begin
            # Compute general_leapfrog_tempering_step giving all parameters explicitly
            result = RHVAEs.general_leapfrog_tempering_step(
                x_mat, z_mat, decoder, metric_param
            )

            @test isa(result, NamedTuple)

            # Check that the result is not NaN
            @test all(!isnan, result.z_init)
            @test all(!isnan, result.ρ_init)
            @test all(!isnan, result.z_final)
            @test all(!isnan, result.ρ_final)

            # Check that the result is not Inf
            @test all(!isinf, result.z_init)
            @test all(!isinf, result.ρ_init)
            @test all(!isinf, result.z_final)
            @test all(!isinf, result.ρ_final)

            # Build temporary RHVAE
            rhvae = RHVAEs.RHVAE(
                joint_log_encoder * decoder,
                metric_chain,
                data,
                T,
                λ,
            )

            # Compute general_leapfrog_tempering_step using RHVAE
            result = RHVAEs.general_leapfrog_tempering_step(x_mat, z_mat, rhvae)

            @test isa(result, NamedTuple)

            # Check that the result is not NaN
            @test all(!isnan, result.z_init)
            @test all(!isnan, result.ρ_init)
            @test all(!isnan, result.z_final)
            @test all(!isnan, result.ρ_final)

            # Check that the result is not Inf
            @test all(!isinf, result.z_init)
            @test all(!isinf, result.ρ_init)
            @test all(!isinf, result.z_final)
            @test all(!isinf, result.ρ_final)
        end # matrix inputs
    end # for decoder
end # general_leapfrog_tempering_step function

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

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), AbstractVector{Float32}))

                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat, metric_param)

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), AbstractMatrix{Float32}))
                end # matrix inputs
            end # latent = false

            @testset "latent = true" begin
                @testset "vector inputs" begin
                    # Run RHVAE with vector inputs
                    result = rhvae(x_vec, metric_param; latent=true)

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), NamedTuple))

                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat, metric_param; latent=true)

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), NamedTuple))
                end # matrix inputs
            end # latent = false
        end # explicit metric_param

        @testset "implicit metric_param" begin
            @testset "latent = false" begin
                @testset "vector inputs" begin
                    # Run RHVAE with vector inputs
                    result = rhvae(x_vec)

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), AbstractVector{Float32}))

                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat)

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), AbstractMatrix{Float32}))
                end # matrix inputs
            end # latent = false

            @testset "latent = true" begin
                @testset "vector inputs" begin
                    # Run RHVAE with vector inputs
                    result = rhvae(x_vec; latent=true)

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), NamedTuple))

                end # vector inputs

                @testset "matrix inputs" begin
                    # Run RHVAE with matrix inputs
                    result = rhvae(x_mat; latent=true)

                    @test isa(result, NamedTuple)

                    @test all(isa.(values(result), NamedTuple))
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
            hvae_outputs = rhvae(x_vec, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_p̄(x_vec, rhvae, metric_param, hvae_outputs)

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Define hvae_outputs
            hvae_outputs = rhvae(x_mat, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_p̄(x_mat, rhvae, metric_param, hvae_outputs)

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
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
            # Define hvae_outputs
            hvae_outputs = rhvae(x_vec, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_q̄(x_vec, rhvae, metric_param, hvae_outputs, β)

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Define hvae_outputs
            hvae_outputs = rhvae(x_mat, metric_param; latent=true)

            # Compute _log_p̄
            result = RHVAEs._log_q̄(x_mat, rhvae, metric_param, hvae_outputs, β)

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
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

        @testset "vector input" begin
            # Compute riemannian_hamiltonian_elbo
            result = RHVAEs.riemannian_hamiltonian_elbo(
                rhvae, metric_param, x_vec
            )

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Compute riemannian_hamiltonian_elbo
            result = RHVAEs.riemannian_hamiltonian_elbo(
                rhvae, metric_param, x_mat
            )

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
        end # matrix input
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

        @testset "vector input" begin
            # Compute loss
            result = RHVAEs.loss(rhvae, x_vec)

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
        end # vector input

        @testset "matrix input" begin
            # Compute loss
            result = RHVAEs.loss(rhvae, x_mat)

            @test isa(result, Float32)

            # Check that the result is not NaN
            @test !isnan(result)

            # Check that the result is not Inf
            @test !isinf(result)
        end # matrix input
    end # for decoder
end # loss function