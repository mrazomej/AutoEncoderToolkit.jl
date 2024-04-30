# Import Zygote
using Zygote
using Zygote: @adjoint
# using FillArrays: Fill
using TaylorDiff
using TaylorDiff: value
using ChainRulesCore

using LinearAlgebra

using ..utils: vec_to_ltri
# ==============================================================================
# Define Zygote.@adjoint for FillArrays.fill
# ==============================================================================

# Note: This is needed because the specific type of FillArrays.fill does not
# have a Zygote.@adjoint function defined. This causes an error when trying to
# backpropagate through the RHVAE.

# Define the Zygote.@adjoint function for the FillArrays.fill method.
# The function takes a matrix `x` of type Float32 and a size `sz` as input.
# @adjoint function (::Type{T})(
#     x::AbstractMatrix{Float32}, sz
# ) where {T<:Fill}
#     # Define the backpropagation function for the adjoint. The function takes a
#     # gradient `Δ` as input and returns the sum of the gradient and `nothing`.
#     back(Δ::AbstractArray) = (sum(Δ), nothing)
#     # Define the backpropagation function for the adjoint. The function takes a
#     # gradient `Δ` as input and returns the value of `Δ` and `nothing`.
#     back(Δ::NamedTuple) = (Δ.value, nothing)
#     # Return the result of the FillArrays.fill method and the backpropagation
#     # function.
#     return Fill(x, sz), back
# end # @adjoint


# @adjoint function fill!(A::AbstractArray, x)
#     # Define the backward pass
#     function back(Δ::AbstractArray)
#         # The gradient with respect to x is the sum of all elements in Δ
#         grad_x = sum(Δ)
#         # The gradient with respect to A is nothing because A is overwritten
#         return (nothing, grad_x)
#     end

#     # Execute the operation and return the result and the backward pass
#     return fill!(A, x), back
# end

# ==============================================================================
# Define Zygote.@adjoint for TaylorDiff.TaylorScalar
# ==============================================================================

# Note: These are supposed to be included as rrules in the TaylorDiff package.
# But, for some reason I cannot get them to work. So, I am including them here
# as I copied them from an old commit in their repo, before they migrated to
# ChainRulesCore.jl.

# @adjoint value(t::TaylorScalar) = value(t), v̄ -> (TaylorScalar(v̄),)
# @adjoint TaylorScalar(v) = TaylorScalar(v), t̄ -> (t̄.value,)
# @adjoint function getindex(t::NTuple{N,T}, i::Int) where {N,T<:Number}
#     getindex(t, i), v̄ -> (tuple(zeros(T, i - 1)..., v̄, zeros(T, N - i)...), nothing)
# end
# end

# ==============================================================================
# Define ChainRulesCore rrules for vec_to_ltri
# ==============================================================================

@doc raw"""
    rrule(::typeof(vec_to_ltri), diag::AbstractVecOrMat, lower::AbstractVecOrMat)

This function defines the reverse mode rule (rrule) for the `vec_to_ltri`
function. The `vec_to_ltri` function converts a diagonal vector and a lower
triangular vector into a lower triangular matrix. The `rrule` function computes
the gradients of the inputs `diag` and `lower` with respect to the output lower
triangular matrix.

# Arguments
- `diag::AbstractVecOrMat`: The diagonal vector.
- `lower::AbstractVecOrMat`: The lower triangular vector.

# Returns
- `ltri`: The lower triangular matrix computed by `vec_to_ltri`.
- `vec_to_ltri_pullback`: The pullback function that computes the gradients of
  `diag` and `lower`.
"""
function ChainRulesCore.rrule(
    ::typeof(vec_to_ltri), diag::AbstractVecOrMat, lower::AbstractVecOrMat
)
    # Compute the lower triangular matrix
    ltri = vec_to_ltri(diag, lower)

    # Define the pullback function
    function vec_to_ltri_pullback(ΔLtri)
        # Extract matrix dimensions and number of samples
        n, _, cols = size(ΔLtri)

        # Initialize the gradients for 'diag' and 'lower'
        Δdiag = zeros(eltype(ΔLtri), size(diag))
        Δlower = zeros(eltype(ΔLtri), size(lower))

        # Compute the gradients for 'diag' and 'lower'
        for k in 1:cols
            for i in 1:n
                # Gradient for 'diag'
                Δdiag[i, k] = ΔLtri[i, i, k]
                for j in 1:(i-1)
                    # Gradient for 'lower'
                    Δlower[(i-1)*(i-2)÷2+j+(k-1)*(n*(n-1)÷2)] = ΔLtri[i, j, k]
                end
            end
        end

        # Return the gradients for 'diag' and 'lower'
        return NoTangent(), Δdiag, Δlower
    end

    # Return the lower triangular matrix and the pullback function
    return ltri, vec_to_ltri_pullback
end