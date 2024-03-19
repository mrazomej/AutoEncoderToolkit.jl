# ==============================================================================
# Approximating geodesics via Splines
# ==============================================================================

# Reference
# > Hofer, M. & Pottmann, H. Energy-Minimizing Splines in Manifolds.

@doc raw"""
    min_energy_spline(γ_init, γ_end, manifold, n_points; step_size, stop, tol)

Function that produces an energy-minimizing spline between two points `γ_init`
and `γ_end` on a Riemmanian manifold using the algorithm by Hoffer & Pottmann.

# Arguments
- `manifold::Function`: Function defining the Riemmanian manifold on which the
  curve lies.
- `γ_init::AbstractVector`: Initial position of curve `γ` on the parameter
  space.
- `γ_end::AbstractVector`: Final position of curve `γ` on the parameter space.
- `n_points::Int`: Number of points to use for the interpolation.

## Optional arguments
- `step_size::Number=1E-3`: Step size used to update curve parameters.
- `stop::Boolean=true`: Boolean indicating if iterations should stop if the
  current point is within certain tolerance value `tol` from the desired final
  point.
- `tol::Number=1E-6`: Tolerated difference between desired and current final
  point for an early stop criteria.

# Returns
- `γ::AbstractMatrix`: `d × N` matrix where each row represents each of the
  dimensions on the manifold and `N` is the number of points needed to
  interpolate between points.

# Citation
> Hofer, M. & Pottmann, H. Energy-Minimizing Splines in Manifolds.
"""
function min_energy_spline(
    manifold::Function,
    γ_init::AbstractVector{T},
    γ_end::AbstractVector{T},
    n_points::Int;
    step_size::Number=1E-3,
    stop::Bool=true,
    tol::Number=1E-3
) where {T<:Any}
    # Initialize matrix where to store interpolation points
    γ = Matrix{T}(undef, length(γ_init), n_points)
    # Set initial value
    γ[:, 1] .= γ_init

    # Map final position to output space
    x_end = manifold(γ_end)

    # Initialize loop counter
    count = 1
    # Loop through points
    for i = 2:n_points
        # 1. Map current point to output space
        x_current = manifold(γ[:, i-1])
        # 2. Compute Jacobian. Each column represents one basis vector c̲ for
        #    the tangent space at x_current.
        J̲̲ = first(Zygote.jacobian(manifold, γ[:, i-1]))
        # 3. Compute the Riemmanian metric tensor at current point
        M̲̲ = J̲̲' * J̲̲
        # 4. Compute the difference between the current position and the desired
        #    final point
        Δx̲ = x_end .- x_current
        # 5. Compute r̲ vector where each entry is the inner product beetween
        #    Δx̲ and each of the basis vectors of the tangent space c̲. Since
        #    these vectors are stored in the Jacobian, we can simply do matrix
        #    vector multiplication.
        r̲ = J̲̲' * Δx̲
        # 6. Solve linear system to find tangential vector
        t̲ = M̲̲ \ r̲
        # 7. Update parameters given chosen step size
        γ[:, i] = γ[:, i-1] .+ (step_size .* t̲)

        # Update loop counter
        count += 1

        # Check if final point satisfies tolerance
        if (stop) .& (LinearAlgebra.norm(γ[:, i] - γ_end) ≤ tol)
            break
        end # if
    end # for

    # Return curve
    if stop
        return γ[:, 1:count]
    else
        return γ
    end # if
end # function