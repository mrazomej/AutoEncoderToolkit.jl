# ==============================================================================
# Discretized curve characteristics
# ==============================================================================

@doc raw"""
    curve_energy(manifold, γ)

Function to compute the (discretized) integral defining the energy of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = 1 / 2 ∫ dt ⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩,

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral. This means that we approximate the
energy as

    E ≈ 1 / 2Δt ∑ᵢ ⟨γ̲(tᵢ+1) - γ̲(tᵢ), M̲̲ (γ̲(tᵢ+1) - γ̲(tᵢ))⟩.

# Arguments

- `γ::AbstractMatrix`: `d×N` Matrix where `d` is the dimension of the manifold
  on which the curve lies and `N` is the number of points along the curve
  (without the initial point `γ̲ₒ`). Each column represents the sampled points
  along the curve. The longer the number of entries in this vector, the more
  accurate the energy estimate will be. 

# Returns
- `Energy::Number`: Value of the energy for the path on the manifold.
"""
function curve_energy(rhvae::RHVAEs, γ::AbstractMatrix)
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate Riemmanian metric at each point

    # Evaluate and return energy
    return (1 / (2 * Δt)) * sum(
        begin
            LinearAlgebra.dot(
                Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
            )
        end for i in axes(Δγ, 2)
    )
end # function

@doc raw"""
    curve_length(γ̲, manifold)

Function to compute the (discretized) integral defining the length of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = ∫ dt √(⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Length::T`: Value of the Length for the path on the manifold.
"""
function curve_length(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return sum(
        [
        sqrt(
            LinearAlgebra.dot(
                Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
            )
        )
        for i = 1:size(Δγ, 2)
    ]
    )
end # function

@doc raw"""
    curve_lengths(γ̲, manifold)

Function to compute the (discretized) integral defining the length of a curve γ
on a Riemmanina manifold. The energy is defined as

    E = ∫ dt √(⟨γ̲̇(t), M̲̲ γ̲̇(t)⟩),

where γ̲̇(t) defines the velocity of the parametric curve, and M̲̲ is the
Riemmanian metric. For this function, we use finite differences from the curve
sample points `γ` to compute this integral.

# Arguments
- `γ::AbstractMatrix{T}`: `d×N` long vector where `d` is the dimension of the
  manifold on which the curve lies and `N` is the number of points along the
  curve (without the initial point `γ̲ₒ`). This vector represents the sampled
  poinst along the curve. The longer the number of entries in this vector, the
  more accurate the energy estimate will be. NOTE: Notice that this function
  asks 
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.

# Returns
- `Lengths::Vector{T}`: Value of the lengths for each individual path on the
  manifold.
"""
function curve_lengths(γ::AbstractMatrix{T}, manifold::Function) where {T<:Real}
    # Define Δt
    Δt = 1 / size(γ, 2)

    # Compute the differences between points
    Δγ = diff(γ, dims=2)

    # Evaluate and return energy
    return [
        sqrt(
            LinearAlgebra.dot(
                Δγ[:, i], riemmanian_metric(manifold, γ[:, i+1]), Δγ[:, i]
            )
        )
        for i = 1:size(Δγ, 2)
    ]
end # function