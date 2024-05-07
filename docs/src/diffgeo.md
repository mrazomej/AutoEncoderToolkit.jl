# Differential Geometry of Generative Models

A lot of recent research in the field of generative models has focused on the
geometry of the learned latent space (see the [references](@ref diffgeoref) at
the end of this section for examples). The non-linear nature of neural networks
makes it relevant to consider the non-Euclidean geometry of the latent space
when trying to gain insights into the structure of the learned space. In other
words, given that neural networks involve a series of non-linear transformations
of the input data, we cannot expect the latent space to be Euclidean, and thus,
we need to account for curvature and other non-Euclidean properties. For this,
we can borrow concepts and tools from Riemannian geometry, now applied to the
latent space of generative models.

`AutoEncoderToolkit.jl` aims to provide the set of necessary tools to study the geometry
of the latent space in the context of variational autoencoders generative
models.

!!! note
    This is very much work in progress. As always, contributions are welcome!

## A word on Riemannian geometry

In what follows we will give a very short primer on some relevant concepts in
differential geometry. This includes some basic definitions and concepts along
with what we consider intuitive explanations of the concepts. We trade rigor for
accessibility, so if you are looking for a more formal treatment, this is not
the place.

!!! note
    These notes are partially based on the 2022 paper by Chadebec et al. [2].

A $d$-dimensional manifold $\mathcal{M}$ is a manifold that is locally
homeomorphic to a $d$-dimensional Euclidean space. This means that the
manifold--some surface or high-dimensional shape--when observed from really
close, can be stretched or bent without tearing or gluing it to make it resemble
regular Euclidean space. 

If the manifold is differentiable, it possesses a tangent space $T_z$ at any
point $z \in \mathcal{M}$ composed of the tangent vectors of the curves passing
by $z$. 

![](./figs/diffgeo01.png)

If the manifold $\mathcal{M}$ is equipped with a smooth inner product, 

```math
g: z \rightarrow \langle \cdot \mid \cdot \rangle_z,
\tag{1}
```
defined on the tangent space $T_z$ for any $z \in \mathcal{M}$, then
$\mathcal{M}$ is a Riemannian manifold and $g$ is the associated Riemannian
metric. With this, a local representation of $g$ at any point $z$ is given by
the positive definite matrix $\mathbf{G}(z)$.

A chart (fancy name for a coordinate system) $(U, \phi)$ provides a homeomorphic
mapping between an open set $U$ of the manifold and an open set $V$ of Euclidean
space. This means that there is a way to bend and stretch any segment of the
manifold to make it look like a segment of Euclidean space. Therefore, given a
point $z \in U$, a chart--its coordinate--$\phi: (z_1, z_2, \ldots, z_d)$
induces a basis $\{\partial_{z_1}, \partial_{z_2}, \ldots, \partial_{z_d}\}$ on
the tangent space $T_z \mathcal{M}$. In other words, the partial derivatives of
the manifold with respect to the dimensions form a basis (think of $\hat{i},
\hat{j}, \hat{k}$ in 3D space) for the tangent space at that point. Hence, the
metric--a "position-dependent scale-bar"--of a Riemannian manifold can be
locally represented at $\phi$ as a positive definite matrix $\mathbf{G}(z)$
with components $g_{ij}(z)$ of the form

```math
g_{ij}(z) = \langle \partial_{z_i} \mid \partial_{z_j} \rangle_z.
\tag{2}
```

This implies that for every pair of vectors $v, w \in T_z \mathcal{M}$ and a
point $z \in \mathcal{M}$, the inner product $\langle v \mid w \rangle_z$ is
given by

```math
\langle v \mid w \rangle_z = v^T \mathbf{G}(z) w.
\tag{3}
```

If $\mathcal{M}$ is connected--a continuous shape with no breaks--a Riemannian
distance between two points $z_1, z_2 \in \mathcal{M}$ can be defined as

```math
\text{dist}(z_1, z_2) = \min_{\gamma} \int_0^1 dt
\sqrt{\langle \dot{\gamma}(t) \mid \dot{\gamma}(t) \rangle_{\gamma(t)}},
\tag{4}
```
where $\gamma$ is a 1D curve traveling from $z_1$ to $z_2$, i.e., $\gamma(0) =
z_1$ and $\gamma(1) = z_2$. Another way to state this is that the length of a
curve on the manifold $\gamma$ is given by
```math
L(\gamma) = \int_0^1 dt 
\sqrt{\langle \dot{\gamma}(t) \mid \dot{\gamma}(t) \rangle_{\gamma(t)}}.
\tag{5}
```
If $L$ minimizes the distance between the initial and final points, then
$\gamma$ is a **geodesic curve**.

The concept of geodesic is so important the study of the Riemannian manifold
learned by generative models that let's try to give another intuitive
explanation. Let us consider a curve $\gamma$ such that
```math
\gamma: [0, 1] \rightarrow \mathbb{R}^d,
\tag{6}
```
In words, $\gamma$ is a function that, without loss of generality, maps a number
between zero and one to the dimensionality of the latent space (the
dimensionality of our manifold). Let us define $f$ to be a continuous function 
that embeds any point along the curve $\gamma$ into the data space, i.e.,
```math
f : \gamma(t) \rightarrow x \in \mathbb{R}^n.
\tag{7}
```
where $n$ is the dimensionality of the data space. 

![](./figs/diffgeo02.png)

The length of this curve in the data space is given by
```math
L(\gamma) = \int_0^1 dt
\left\| \frac{d f}{dt} \right\|_2.
\tag{8}
```
After some manipulation, we can show that the length of the curve in the data
space is given by
```math
L(\gamma) = \int_0^1 dt
\sqrt{
    \dot{\gamma}(t)^T \mathbf{G}(\gamma(t)) \dot{\gamma}(t)
},
\tag{9}
```
where $\dot{\gamma}(t)$ is the derivative of $\gamma$ with respect to $t$, and
$T$ denotes the transpose of a vector. For a Euclidean space, the length of
the curve would take the same functional form, except that the metric tensor
would be given by the identity matrix. This is why the metric tensor can be 
thought of as a position-dependent scale-bar.

## [Neural Geodesic Networks] (@id neuralgeodesic)

Computing a geodesic on a Riemannian manifold is a non-trivial task, especially
when the manifold is parametrized by a neural network. Thus, knowing the 
function $\gamma$ that minimizes the distance between two points $z_1$ and $z_2$
is not straightforward. However, as first suggested by Chen et al. [1], we can
repurpose the expressivity of neural networks to approximate almost any function
to approximate the geodesic curve. This is the idea behind the Neural Geodesic
module in `AutoEncoderToolkit.jl`.

Briefly, to approximate the geodesic curve between two points $z_1$ and $z_2$
in latent space, we define a neural network $g_\omega$ such that
```math
g_\omega: \mathbb{R} \rightarrow \mathbb{R}^d,
\tag{10}
```
i.e., the neural network takes a number between zero and one and maps it to the
dimensionality of the latent space. The intention is to have $g_\omega \approx
\gamma$, where $\omega$ are the parameters of the neural network we are free to
optimize.

We approximate the integral defining the length of the curve in the latent space
with $n$ equidistantly sampled points $t_i$ between zero and one. The length of
the curve is then approximated by
```math
L(g_\gamma(t)) \approx \frac{1}{n} \sum_{i=1}^n 
\sqrt{
    \dot{g}_\omega(t_i)^T \mathbf{G}(g_\omega(t_i)) \dot{g}_\omega(t_i)
},
```
By setting the loss function to be this approximation of the length of the
curve, we can train the neural network to approximate the geodesic curve.

`AutoEncoderToolkit.jl` provides the `NeuralGeodesic` struct to implement this idea. The
struct takes three inputs:
- The multi-layer perceptron (MLP) that approximates the geodesic curve.
- The initial point in latent space.
- The final point in latent space.

### `NeuralGeodesic` struct

```@docs
AutoEncoderToolkit.diffgeo.NeuralGeodesics.NeuralGeodesic
```

### `NeuralGeodesic` forward pass

```@docs
AutoEncoderToolkit.diffgeo.NeuralGeodesics.NeuralGeodesic(::AbstractVector)
```

### `NeuralGeodesic` loss function

```@docs
AutoEncoderToolkit.diffgeo.NeuralGeodesics.loss
```

### `NeuralGeodesic` training

```@docs
AutoEncoderToolkit.diffgeo.NeuralGeodesics.train!
```

### Other functions for `NeuralGeodesic`

```@docs
AutoEncoderToolkit.diffgeo.NeuralGeodesics.curve_velocity_TaylorDiff
AutoEncoderToolkit.diffgeo.NeuralGeodesics.curve_velocity_finitediff
AutoEncoderToolkit.diffgeo.NeuralGeodesics.curve_length
AutoEncoderToolkit.diffgeo.NeuralGeodesics.curve_energy
```


## [References] (@id diffgeoref)
1. Chen, N. et al. Metrics for Deep Generative Models. in Proceedings of the
   Twenty-First International Conference on Artificial Intelligence and
   Statistics 1540–1550 (PMLR, 2018).
2. Chadebec, C. & Allassonnière, S. A Geometric Perspective on Variational
   Autoencoders. Preprint at http://arxiv.org/abs/2209.07370 (2022).
3. Chadebec, C., Mantoux, C. & Allassonnière, S. Geometry-Aware Hamiltonian
   Variational Auto-Encoder. Preprint at http://arxiv.org/abs/2010.11518 (2020).
4. Arvanitidis, G., Hauberg, S., Hennig, P. & Schober, M. Fast and Robust
   Shortest Paths on Manifolds Learned from Data. in Proceedings of the
   Twenty-Second International Conference on Artificial Intelligence and
   Statistics 1506–1515 (PMLR, 2019).
5. Arvanitidis, G., Hauberg, S. & Schölkopf, B. Geometrically Enriched Latent
   Spaces. Preprint at https://doi.org/10.48550/arXiv.2008.00565 (2020).
6. Arvanitidis, G., González-Duque, M., Pouplin, A., Kalatzis, D. & Hauberg, S.
   Pulling back information geometry. Preprint at
   http://arxiv.org/abs/2106.05367 (2022).
7. Fröhlich, C., Gessner, A., Hennig, P., Schölkopf, B. & Arvanitidis, G.
   Bayesian Quadrature on Riemannian Data Manifolds.
8. Kalatzis, D., Eklund, D., Arvanitidis, G. & Hauberg, S. Variational
   Autoencoders with Riemannian Brownian Motion Priors. Preprint at
   http://arxiv.org/abs/2002.05227 (2020).
9. Arvanitidis, G., Hansen, L. K. & Hauberg, S. Latent Space Oddity: on the
   Curvature of Deep Generative Models. Preprint at
   http://arxiv.org/abs/1710.11379 (2021).
