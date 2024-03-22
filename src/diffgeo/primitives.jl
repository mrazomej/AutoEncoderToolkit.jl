# ==============================================================================
# This file is a simple copy of the file:
# https://github.com/JuliaDiff/TaylorDiff.jl/blob/4f7b477b188ef15f6a4368285bb2c40319089a48/ext/TaylorDiffNNlibExt.jl#L4
# 
# because the authors of the package haven't updated their package in a long
# time and I need these functions to be available.
# ==============================================================================
import TaylorDiff
import NNlib: oftf
import NNlib: sigmoid_fast, tanh_fast, rrelu, leakyrelu

@inline sigmoid_fast(t::TaylorDiff.TaylorScalar) = one(t) / (one(t) + exp(-t))

@inline tanh_fast(t::TaylorDiff.TaylorScalar) = tanh(t)

@inline function rrelu(t::TaylorDiff.TaylorScalar{T,N},
    l=oftf(t, 1 / 8),
    u=oftf(t, 1 / 3)) where {T,N}
    a = (u - l) * rand(float(T)) + l
    return leakyrelu(t, a)
end