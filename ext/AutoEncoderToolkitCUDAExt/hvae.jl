using CUDA
using AutoEncoderToolkit.HVAEs

@doc raw"""
    train!(
        hvae::HVAE, 
        x::AbstractArray, 
        opt::NamedTuple; 
        loss_function::Function=loss, 
        loss_kwargs::Union{NamedTuple,Dict}=Dict(),
        verbose::Bool=false,
        loss_return::Bool=false,
    )

Customized training function to update parameters of a Hamiltonian Variational
Autoencoder given a specified loss function.

# Arguments
- `hvae::HVAE`: A struct containing the elements of a Hamiltonian Variational
  Autoencoder.
- `x::AbstractArray`: Input data to the HVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the HVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Dict=Dict()`: Arguments for the loss function. These might
  include parameters like `K`, `ϵ`, `βₒ`, `steps`, `∇H`, `∇H_kwargs`,
  `tempering_schedule`, `reg_function`, `reg_kwargs`, `reg_strength`, depending
  on the specific loss function in use.
- `verbose::Bool=false`: Whether to print the loss at each iteration.
- `loss_return::Bool=false`: Whether to return the loss at each iteration.

# Description
Trains the HVAE by:
1. Computing the gradient of the loss w.r.t the HVAE parameters.
2. Updating the HVAE parameters using the optimizer.
3. Updating the metric parameters.
"""
function HVAEs.train!(
    hvae::HVAEs.HVAE,
    x::CUDA.CuArray,
    opt::NamedTuple;
    loss_function::Function=HVAEs.loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(hvae) do hvae_model
            loss_function(hvae_model, x; loss_kwargs...)
        end # do block
    end

    # Update parameters
    Flux.Optimisers.update!(opt, hvae, ∇L[1])

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function

# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

@doc raw"""
    train!(
        hvae::HVAE, 
        x_in::AbstractArray,
        x_out::AbstractArray,
        opt::NamedTuple; 
        loss_function::Function=loss, 
        loss_kwargs::Union{NamedTuple,Dict}=Dict(),
        verbose::Bool=false,
        loss_return::Bool=false,
    )

Customized training function to update parameters of a Hamiltonian Variational
Autoencoder given a specified loss function.

# Arguments
- `hvae::HVAE`: A struct containing the elements of a Hamiltonian Variational
  Autoencoder.
- `x_in::AbstractArray`: Input data to the HVAE encoder. The last dimension is
  taken as having each of the samples in a batch.
- `x_out::AbstractArray`: Target data to compute the reconstruction error. The
  last dimension is taken as having each of the samples in a batch.
- `opt::NamedTuple`: State of the optimizer for updating parameters. Typically
  initialized using `Flux.Optimisers.update!`.

# Optional Keyword Arguments
- `loss_function::Function=loss`: The loss function used for training. It should
  accept the HVAE model, data `x`, and keyword arguments in that order.
- `loss_kwargs::Dict=Dict()`: Arguments for the loss function. These might
  include parameters like `K`, `ϵ`, `βₒ`, `steps`, `∇H`, `∇H_kwargs`,
  `tempering_schedule`, `reg_function`, `reg_kwargs`, `reg_strength`, depending
  on the specific loss function in use.
- `verbose::Bool=false`: Whether to print the loss at each iteration.
- `loss_return::Bool=false`: Whether to return the loss at each iteration.

# Description
Trains the HVAE by:
1. Computing the gradient of the loss w.r.t the HVAE parameters.
2. Updating the HVAE parameters using the optimizer.
3. Updating the metric parameters.
"""
function HVAEs.train!(
    hvae::HVAEs.HVAE,
    x_in::CUDA.CuArray,
    x_out::CUDA.CuArray,
    opt::NamedTuple;
    loss_function::Function=HVAEs.loss,
    loss_kwargs::Union{NamedTuple,Dict}=Dict(),
    verbose::Bool=false,
    loss_return::Bool=false,
)
    # Compute VAE gradient
    L, ∇L = CUDA.allowscalar() do
        Flux.withgradient(hvae) do hvae_model
            loss_function(hvae_model, x_in, x_out; loss_kwargs...)
        end # do block
    end

    # Update parameters
    Flux.Optimisers.update!(opt, hvae, ∇L[1])

    # Check if loss should be returned
    if loss_return
        return L
    end # if

    # Check if loss should be printed
    if verbose
        println("Loss: ", L)
    end # if
end # function