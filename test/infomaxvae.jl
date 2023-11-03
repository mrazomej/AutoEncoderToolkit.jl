##
println("Testing InfoMaxVAEs module:\n")
##

# Import AutoEncode.jl module to be tested
import AutoEncode.InfoMaxVAEs
# Import Flux library
import Flux

# Import basic math
import Random
import StatsBase

Random.seed!(42)

## =============================================================================

# Define number of inputs
n_input = 3
# Define number of synthetic data points
n_data = 50
# Define number of hidden layers
n_hidden = 2
# Define number of neurons in non-linear hidden layers
n_neuron = 10
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

# Fit model to standardize data to mean zero and standard deviation 1 on each
# environment
dt = StatsBase.fit(StatsBase.ZScoreTransform, data, dims=2)

# Center data to have mean zero and standard deviation one
data = StatsBase.transform(dt, data);

## =============================================================================

println("Testing InfoMaxVAE = JointEncoder + SimpleDecoder")

# Define latent space activation function
latent_activation = Flux.identity
# Define output layer activation function
output_activation = Flux.identity

# Define encoder layer and activation functions
encoder_neurons = repeat([n_neuron], n_hidden)
encoder_activation = repeat([Flux.swish], n_hidden)

# Define decoder layer and activation function
decoder_neurons = repeat([n_neuron], n_hidden)
decoder_activation = repeat([Flux.swish], n_hidden)

# Define MLP layer and activation function
mlp_neurons = repeat([n_neuron], n_hidden)
mlp_activation = repeat([Flux.swish], n_hidden)

# Define MLP output activation function
mlp_output_activation = Flux.identity

# Initialize encoder
encoder = InfoMaxVAEs.JointEncoder(
    n_input,
    n_latent,
    encoder_neurons,
    encoder_activation,
    latent_activation
)

# Initialize decoder
decoder = InfoMaxVAEs.SimpleDecoder(
    n_input,
    n_latent,
    decoder_neurons,
    decoder_activation,
    output_activation
)

# Initialize MLP
mlp = InfoMaxVAEs.MLP(
    n_input,
    n_latent,
    mlp_neurons,
    mlp_activation,
    mlp_output_activation
)

infomaxvae = InfoMaxVAEs.InfoMaxVAE(
    InfoMaxVAEs.VAE(encoder, decoder),
    mlp
)

# Test if it returns the right type
@test isa(infomaxvae, InfoMaxVAEs.InfoMaxVAE)

##

# Test that reconstruction works
@test isa(infomaxvae(data; latent=false), AbstractVecOrMat)

# Generate list of permutated data
data_shuffle = data[:, Random.shuffle(1:end)]

#  Test loss functions
@test isa(
    InfoMaxVAEs.loss(
        infomaxvae.vae, infomaxvae.mlp, data[:, 1], data_shuffle[:, 1]
    ),
    AbstractFloat
)

@test isa(
    InfoMaxVAEs.loss(
        infomaxvae.vae, infomaxvae.mlp,
        data[:, 1], data[:, 1], data_shuffle[:, 1]
    ),
    AbstractFloat
)

@test isa(
    InfoMaxVAEs.mlp_loss(
        infomaxvae.vae, infomaxvae.mlp, data[:, 1], data_shuffle[:, 1]
    ),
    AbstractFloat
)

## =============================================================================

# Extract parameters
params_init = deepcopy(collect(Flux.params(infomaxvae)))

# Explicit setup of optimizer
vae_opt = Flux.Train.setup(
    Flux.Optimisers.Adam(1E-1),
    infomaxvae.vae
)

mlp_opt = Flux.Train.setup(
    Flux.Optimisers.Adam(1E-1),
    infomaxvae.mlp
)

# Loop through a couple of epochs
losses = Float32[]  # Track the loss
# Loop through a couple of epochs
for epoch = 1:10
    # Test training function
    InfoMaxVAEs.train!(infomaxvae, data, vae_opt, mlp_opt)

    # Compute average loss
    push!(
        losses,
        StatsBase.mean(InfoMaxVAEs.loss.(
            Ref(infomaxvae.vae), Ref(infomaxvae.mlp),
            eachcol(data), eachcol(data_shuffle)
        ))
    )
end # for

# Check if loss is decreasing
@test all(diff(losses) ≠ 0)

# Extract modified parameters
params_end = deepcopy(collect(Flux.params(infomaxvae)))

# Check that parameters have significantly changed
threshold = 1e-5
# Check if any parameter has changed significantly
@test all([
    all(abs.(x .- y) .> threshold) for (x, y) in zip(params_init, params_end)
])

println("Passed tests for InfoMaxVAEs module!\n")
##