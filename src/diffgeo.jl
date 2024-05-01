# Import basic math
import LinearAlgebra
import Distances

# Import library to perform Einstein summation
# using TensorOperations: @tensor

# Import library for automatic differentiation
import Zygote

# Import ML library
import Flux

# Import function for k-medoids clustering
using Clustering: kmedoids

# Import Abstract Types
using ..AutoEncoderToolkit: JointGaussianLogEncoder, SimpleGaussianDecoder, JointGaussianLogDecoder,
    SplitGaussianLogDecoder, JointGaussianDecoder, SplitGaussianDecoder,
    AbstractDeterministicDecoder, AbstractVariationalDecoder,
    AbstractVariationalEncoder, AbstractDecoder

# Import types from RHVAEs
using ..RHVAEs: RHVAEs

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #
# Differential Geometry on Riemmanian Manifolds
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #

# ==============================================================================
# Pullback Metric computation
# ==============================================================================

# Reference
# > Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
# > Curvature of Deep Generative Models. Preprint at
# > http://arxiv.org/abs/1710.11379 (2021).

include("diffgeo/pullback.jl")

# ==============================================================================
# Geodesic differential equation
# ==============================================================================

# include("diffgeo/diffeq.jl")

# ==============================================================================
# Approximating geodesics via Splines
# ==============================================================================

# Reference
# > Hofer, M. & Pottmann, H. Energy-Minimizing Splines in Manifolds.

# include("diffgeo/splines.jl")

# ==============================================================================
# Discretized curve characteristics
# ==============================================================================

# include("diffgeo/discrete.jl")

# ==============================================================================
# Encoder Riemmanian metric
# ==============================================================================

# Reference
# > Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational 
# > Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).

# include("diffgeo/encoder_metric.jl")

# ==============================================================================
# Geodesic computation via neural networks
# ==============================================================================

# References

# > Chen, N. et al. Metrics for Deep Generative Models. in Proceedings of the
# > Twenty-First International Conference on Artificial Intelligence and
# > Statistics 1540–1550 (PMLR, 2018).

# > Chadebec, C., Mantoux, C. & Allassonnière, S. Geometry-Aware Hamiltonian
# > Variational Auto-Encoder. Preprint at http://arxiv.org/abs/2010.11518
# > (2020).

module NeuralGeodesics
include("diffgeo/neural_geodesics.jl")
end # submodule