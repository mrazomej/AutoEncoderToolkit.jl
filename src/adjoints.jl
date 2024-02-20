# Import Zygote
using Zygote
using Zygote: @adjoint
# Import FillArrays
using FillArrays: Fill
# Import TaylorDiff
using TaylorDiff
using TaylorDiff: value

# ==============================================================================
# Define Zygote.@adjoint for FillArrays.fill
# ==============================================================================

# Note: This is needed because the specific type of FillArrays.fill does not
# have a Zygote.@adjoint function defined. This causes an error when trying to
# backpropagate through the RHVAE.

# Define the Zygote.@adjoint function for the FillArrays.fill method.
# The function takes a matrix `x` of type Float32 and a size `sz` as input.
@adjoint function (::Type{T})(
    x::AbstractMatrix{Float32}, sz
) where {T<:Fill}
    # Define the backpropagation function for the adjoint. The function takes a
    # gradient `Δ` as input and returns the sum of the gradient and `nothing`.
    back(Δ::AbstractArray) = (sum(Δ), nothing)
    # Define the backpropagation function for the adjoint. The function takes a
    # gradient `Δ` as input and returns the value of `Δ` and `nothing`.
    back(Δ::NamedTuple) = (Δ.value, nothing)
    # Return the result of the FillArrays.fill method and the backpropagation
    # function.
    return Fill(x, sz), back
end # @adjoint


@adjoint function fill!(A::AbstractArray, x)
    # Define the backward pass
    function back(Δ::AbstractArray)
        # The gradient with respect to x is the sum of all elements in Δ
        grad_x = sum(Δ)
        # The gradient with respect to A is nothing because A is overwritten
        return (nothing, grad_x)
    end

    # Execute the operation and return the result and the backward pass
    return fill!(A, x), back
end

# ==============================================================================
# Define Zygote.@adjoint for TaylorDiff.TaylorScalar
# ==============================================================================

# Note: These are supposed to be included as rrules in the TaylorDiff package.
# But, for some reason I cannot get them to work. So, I am including them here
# as I copied them from an old commit in their repo, before they migrated to
# ChainRulesCore.jl.

@adjoint value(t::TaylorScalar) = value(t), v̄ -> (TaylorScalar(v̄),)
@adjoint TaylorScalar(v) = TaylorScalar(v), t̄ -> (t̄.value,)
@adjoint function getindex(t::NTuple{N,T}, i::Int) where {N,T<:Number}
    getindex(t, i), v̄ -> (tuple(zeros(T, i - 1)..., v̄, zeros(T, N - i)...), nothing)
end