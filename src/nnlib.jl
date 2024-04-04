using CUDA
using TaylorDiff
using NNlib

function NNlib.batched_vec(
    M::CUDA.CuArray{S,3},
    w::CUDA.CuMatrix{T}
) where {T<:TaylorDiff.TaylorScalar{Float32,2}, S<:Number}

    # Extract the coefficients from w
    w_coeffs = reinterpret(Float32, w)[1:2:end, :]
    # Extract order from w
    w_order = reinterpret(Float32, w)[2:2:end, :]

    # Create array to store the result of M * w
    Mw = CUDA.zeros(S, size(w_coeffs))
    # Perform element-wise multiplication: v_i .* (M_i * w_i)
    CUDA.CUBLAS.gemv_strided_batched!(
        'N', CUDA.one(S), M, w_coeffs, CUDA.zero(S), Mw
    )

    # Create array to store the result of M * w
    Mw_order = CUDA.zeros(S, size(w_order))
    # Perform element-wise multiplication: v_i .* (M_i * w_i)
    CUDA.CUBLAS.gemv_strided_batched!(
        'N', CUDA.one(S), M, w_order, CUDA.zero(S), Mw_order
    )

    Mw = TaylorDiff.TaylorScalar{S, 2}.(Mw, Mw_order)

    return Mw
end # function