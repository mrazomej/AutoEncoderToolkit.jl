# Utils

`AutoEncoderToolkit.jl` offers a series of utility functions for different
tasks. 

## Training Utilities

```@docs
AutoEncoderToolkit.utils.step_scheduler
AutoEncoderToolkit.utils.cycle_anneal
AutoEncoderToolkit.utils.locality_sampler
```

## [Centroid Finding Utilities] (@id centroidutils)

Some VAE models, such as the [`RHVAE`](@ref RHVAEsmodule), require clustering
of the data. Specifically `RHVAE` can take a fixed subset of the training data
as a reference for the computation of the metric tensor. The following functions
can be used to define this reference subset to be used as centroids for the
metric tensor computation.

```@docs
AutoEncoderToolkit.utils.centroids_kmeans
AutoEncoderToolkit.utils.centroids_kmedoids
```

## Other Utilities

```@docs
AutoEncoderToolkit.utils.storage_type
AutoEncoderToolkit.utils.vec_to_ltri
AutoEncoderToolkit.utils.vec_mat_vec_batched
AutoEncoderToolkit.utils.slogdet
AutoEncoderToolkit.utils.sample_MvNormalCanon
AutoEncoderToolkit.utils.unit_vector
AutoEncoderToolkit.utils.finite_difference_gradient
AutoEncoderToolkit.utils.taylordiff_gradient
```