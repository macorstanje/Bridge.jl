# Gaussian
using Distributions
using Base.LinAlg: norm_sqr

import Base: rand
import Distributions: pdf, logpdf, sqmahal
import Base: chol, size

"""
    PSD{T}

Simple wrapper for the lower triangular Cholesky root of a positive (semi-)definite element `σ`.
"""
type PSD{T}
    σ::T
    PSD(σ::T) where {T} = istril(σ) ? new{T}(σ) : throw(ArgumentError("Argument not lower triangular"))
end
chol(P::PSD) = P.σ' 

sumlogdiag(Σ::Float64, d=1) = log(Σ)
sumlogdiag(Σ,d) = sum(log.(diag(Σ)))
sumlogdiag(J::UniformScaling, d)= log(J.λ)*d

_logdet(Σ::PSD, d) = 2*sumlogdiag(Σ.σ, d)

_logdet(Σ, d) = logdet(Σ)
_logdet(J::UniformScaling, d) = log(J.λ) * d

_symmetric(Σ) = Symmetric(Σ)
_symmetric(J::UniformScaling) = J

"""
    Gaussian(μ, Σ) -> P

Gaussian distribution with mean `μ`` and covariance `Σ`. Defines `rand(P)` and `(log-)pdf(P, x)`.
Designed to work with `Number`s, `UniformScaling`s, `StaticArrays` and `PSD`-matrices.

Implementation details: On `Σ` the functions `logdet`, `whiten` and `unwhiten`
(or `chol` as fallback for the latter two) are called.
"""
struct Gaussian{T,S}
    μ::T
    Σ::S
    Gaussian(μ::T, Σ::S) where {T,S} = new{T,S}(μ, Σ)
end
dim(P::Gaussian) = length(P.μ)
whiten(Σ::PSD, z) = Σ.σ\z
whiten(Σ, z) = chol(Σ)'\z
whiten(Σ::UniformScaling, z) = z/sqrt(Σ.λ)
sqmahal(P::Gaussian, x) = norm_sqr(whiten(P.Σ,x - P.μ))

rand(P::Gaussian) = P.μ + chol(P.Σ)'*randn(typeof(P.μ))
rand(P::Gaussian{Vector}) = P.μ + chol(P.Σ)'*randn(T, length(P.μ))

logpdf(P::Gaussian, x) = -(sqmahal(P,x) + _logdet(P.Σ, dim(P)) + dim(P)*log(2pi))/2    
pdf(P::Gaussian, x) = exp(logpdf(P::Gaussian, x))

"""
    logpdfnormal(x, Σ) 

logpdf of centered Gaussian with covariance Σ
"""
function logpdfnormal(x, Σ) 

    S = chol(_symmetric(Σ))'

    d = length(x)
     -((norm(S\x))^2 + 2sumlogdiag(S,d) + d*log(2pi))/2
end
function logpdfnormal(x::Float64, Σ) 
     -(x^2/Σ + log(Σ) + log(2pi))/2
end
