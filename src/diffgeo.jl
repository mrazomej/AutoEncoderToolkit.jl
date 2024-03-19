# Import basic math
import LinearAlgebra
import Distances

# Import library to perform Einstein summation
using TensorOperations: @tensor

# Import library for automatic differentiation
import Zygote

# Import ML library
import Flux

# Import function for k-medoids clustering
using Clustering: kmedoids

# Import Abstract Types
using ..AutoEncode: JointLogEncoder, SimpleDecoder, JointLogDecoder,
    SplitLogDecoder, JointDecoder, SplitDecoder,
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

include("diffgeo/diffeq.jl")

# ==============================================================================
# Approximating geodesics via Splines
# ==============================================================================

# Reference
# > Hofer, M. & Pottmann, H. Energy-Minimizing Splines in Manifolds.

include("diffgeo/splines.jl")

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

