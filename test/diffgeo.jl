println("\nTesting AutoEncode.diffgeo.NeuralGeodesics module...\n")
# Import AutoEncode.jl module to be tested
import AutoEncode.diffgeo.NeuralGeodesics
import AutoEncode.RHVAEs: RHVAE, metric_tensor

# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase
import Distributions

Random.seed!(42)

## =============================================================================

@testset "NeuralGeodesic struct" begin
    # Define dimensionality of latent space
    latent_dim = 2
    # Define number of hidden layers in the MLP
    n_hidden = 2
    # Define number of neurons in each hidden layer
    n_neuron = 10
    # Define activation function for hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)

    # Define initial and end points of the geodesic curve
    z_init = randn(Float32, latent_dim)
    z_end = randn(Float32, latent_dim)

    # Initialize the MLP
    mlp = Flux.Chain(
        Flux.Dense(1, n_neuron, hidden_activation[1]),
        [Flux.Dense(n_neuron, n_neuron, a) for a in hidden_activation[2:end]]...,
        Flux.Dense(n_neuron, latent_dim)
    )

    # Initialize the NeuralGeodesic
    curve = NeuralGeodesics.NeuralGeodesic(mlp, z_init, z_end)

    @testset "Type checking" begin
        @test typeof(curve) == NeuralGeodesics.NeuralGeodesic
    end

    @testset "Forward pass" begin
        # Define scalar time
        t = 0.5f0

        # Test forward pass with scalar time
        result = curve(t)
        @test isa(result, AbstractVector{Float32})
        @test length(result) == latent_dim

        # Define vector of times
        t_vec = collect(range(0.0f0, 1.0f0, length=10))

        # Test forward pass with vector of times
        result = curve(t_vec)
        @test isa(result, AbstractMatrix{eltype(t_vec)})
        @test size(result) == (latent_dim, 10)
    end # @testset "Forward pass"
end # @testset "NeuralGeodesic struct"

## =============================================================================

@testset "Curve velocity computation" begin
    # Define dimensionality of latent space
    latent_dim = 2
    # Define number of hidden layers in the MLP
    n_hidden = 2
    # Define number of neurons in each hidden layer
    n_neuron = 10
    # Define activation function for hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)

    # Define initial and end points of the geodesic curve
    z_init = randn(Float32, latent_dim)
    z_end = randn(Float32, latent_dim)

    # Initialize the MLP
    mlp = Flux.Chain(
        Flux.Dense(1, n_neuron, hidden_activation[1]),
        [Flux.Dense(n_neuron, n_neuron, a) for a in hidden_activation[2:end]]...,
        Flux.Dense(n_neuron, latent_dim)
    )

    # Initialize the NeuralGeodesic
    curve = NeuralGeodesics.NeuralGeodesic(mlp, z_init, z_end)

    # Define scalar time
    t = 0.5f0

    # Define vector of times
    t_vec = collect(range(0.0f0, 1.0f0, length=10))

    @testset "curve_velocity_TaylorDiff" begin
        result = NeuralGeodesics.curve_velocity_TaylorDiff(curve, t)
        @test isa(result, AbstractVector{eltype(t)})
        @test length(result) == latent_dim

        result = NeuralGeodesics.curve_velocity_TaylorDiff(curve, t_vec)
        @test isa(result, AbstractMatrix{eltype(t)})
        @test size(result) == (latent_dim, 10)
    end # @testset "curve_velocity_TaylorDiff"

    @testset "curve_velocity_finitediff" begin
        result = NeuralGeodesics.curve_velocity_finitediff(
            curve, t_vec; fdtype=:forward
        )
        @test isa(result, AbstractMatrix{Float32})
        @test size(result) == (latent_dim, 10)

        result = NeuralGeodesics.curve_velocity_finitediff(curve, t_vec; fdtype=:central)
        @test isa(result, AbstractMatrix{Float32})
        @test size(result) == (latent_dim, 10)
    end # @testset "curve_velocity_finitediff"
end # @testset "Curve velocity computation"

## =============================================================================

@testset "Curve integrals" begin
    # Define dimensionality of latent space
    latent_dim = 2
    # Define number of time points
    n_time = 10

    # Define Riemannian metric tensor
    riemannian_metric = randn(Float32, latent_dim, latent_dim, n_time)
    # Multiply by transpose to ensure positive definiteness
    riemannian_metric = Flux.batched_mul(
        riemannian_metric,
        Flux.batched_transpose(riemannian_metric)
    )
    # Define curve velocity
    curve_velocity = randn(Float32, latent_dim, n_time)
    # Define vector of times
    t = collect(range(0.0f0, 1.0f0, length=n_time))

    @testset "curve_length" begin
        result = NeuralGeodesics.curve_length(
            riemannian_metric, curve_velocity, t;
        )
        @test isa(result, Number)
    end # @testset "curve_length"

    @testset "curve_energy" begin
        result = NeuralGeodesics.curve_energy(
            riemannian_metric, curve_velocity, t;
        )
        @test isa(result, Number)
    end # @testset "curve_energy"
end # @testset "Curve integrals"

## =============================================================================

println("Defining RHVAE layers structure...")

# Define dimensionality of latent space
latent_dim = 2
# Define number of hidden layers in the MLP
n_hidden = 2
# Define number of neurons in each hidden layer
n_neuron = 10

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.relu], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.relu], n_hidden)

metric_neurons = repeat([n_neuron], n_hidden)
metric_activation = repeat([Flux.relu], n_hidden)

# Initialize JointLogEncoder
joint_log_encoder = VAEs.JointLogEncoder(
    n_input,
    latent_dim,
    encoder_neurons,
    encoder_activation,
    latent_activation,
)

# Initialize SimpleDecoder
simple_decoder = VAEs.SimpleDecoder(
    n_input,
    latent_dim,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Define Metric MLP
metric_chain = RHVAEs.MetricChain(
    n_input,
    latent_dim,
    metric_neurons,
    metric_activation,
    Flux.identity
)

# Initialize RHVAE
rhvae = AutoEncode.RHVAEs.RHVAE(
    joint_log_encoder * simple_decoder,
    metric_chain,
    randn(Float32, n_input, n_input),
    0.8f0,
    0.01f0
)
AutoEncode.RHVAEs.update_metric!(rhvae)

## =============================================================================

@testset "Loss function" begin
    # Define activation function for hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)

    # Define initial and end points of the geodesic curve
    z_init = randn(Float32, latent_dim)
    z_end = randn(Float32, latent_dim)

    # Initialize the MLP
    mlp = Flux.Chain(
        Flux.Dense(1, n_neuron, hidden_activation[1]),
        [Flux.Dense(n_neuron, n_neuron, a) for a in hidden_activation[2:end]]...,
        Flux.Dense(n_neuron, latent_dim)
    )

    # Initialize the NeuralGeodesic
    curve = NeuralGeodesics.NeuralGeodesic(mlp, z_init, z_end)

    # Define vector of times
    t = collect(range(0.0f0, 1.0f0, length=10))

    result = NeuralGeodesics.loss(curve, rhvae, t)
    @test isa(result, Number)

    result = NeuralGeodesics.loss(
        curve, rhvae, t;
        curve_velocity=NeuralGeodesics.curve_velocity_finitediff
    )
    @test isa(result, Number)

    result = NeuralGeodesics.loss(
        curve, rhvae, t; curve_integral=NeuralGeodesics.curve_energy
    )
    @test isa(result, Number)

    result = NeuralGeodesics.loss(curve, rhvae, t;)
    @test isa(result, Number)
end # @testset "Loss function"

## =============================================================================

@testset "NeuralGeodesic training" begin
    # Define activation function for hidden layers
    hidden_activation = repeat([Flux.relu], n_hidden)

    # Define initial and end points of the geodesic curve
    z_init = randn(Float32, latent_dim)
    z_end = randn(Float32, latent_dim)

    # Initialize the MLP
    mlp = Flux.Chain(
        Flux.Dense(1, n_neuron, hidden_activation[1]),
        [Flux.Dense(n_neuron, n_neuron, a) for a in hidden_activation[2:end]]...,
        Flux.Dense(n_neuron, latent_dim)
    )

    # Initialize the NeuralGeodesic
    curve = NeuralGeodesics.NeuralGeodesic(mlp, z_init, z_end)

    # Define vector of times
    t = collect(range(0.0f0, 1.0f0, length=10))

    # Define optimizer
    opt = Flux.Train.setup(Flux.Optimisers.Adam(), curve)

    L = NeuralGeodesics.train!(curve, rhvae, t, opt; loss_return=true)
    @test isa(L, Number)
end # @testset "NeuralGeodesic training"

println("\nAll tests passed!\n")