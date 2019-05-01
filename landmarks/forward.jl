# reminder, to type H*, do H\^+
#outdir="output/landmarks/"
#cd("/Users/Frank/.julia/dev/Bridge/landmarks")
#cd("landmarks")


using Bridge, StaticArrays, Distributions
using Bridge:logpdfnormal
using Test, Statistics, Random, LinearAlgebra
using Bridge.Models
using DelimitedFiles
using DataFrames
using CSV
using RCall
using Base.Iterators
using SparseArrays

models = [:ms, :ahs]
model = models[1]
TEST = false#true

discrmethods = [:ralston, :lowrank, :psd]
discrmethod = discrmethods[1]

obsschemes =[:full, :partial]
obsscheme = obsschemes[2]

const d = 2
const itostrat=true

n = 4 # nr of landmarks
ldim = 40   # dimension of low-rank approximation to H\^+

cheat = true #true#false # if cheat is true, then we initialise at x0 (true value) and
# construct the guiding term based on xT (true value)

θ = -π/6# π/6 0#π/5  # angle to rotate endpoint

ϵ = 10.0^(-4)   # parameter for initialising Hend⁺
σobs = 10.0^(-3)   # noise on observations

println(model)
println(discrmethod)
println(obsscheme)

T = 0.2#1.0#0.5
t = 0.0:0.01:T  # time grid

#Random.seed!(5)
include("state.jl")
include("models.jl")
include("patches.jl")
include("guiding.jl")
include("LowrankRiccati.jl")
using .LowrankRiccati


### Specify landmarks models
a = 3.0 # the larger, the stronger landmarks behave similarly
λ = 0.0; #= not the lambda of noise fields  =# γ = 8.0
db = 3.0 # domainbound
nfstd = .5 # tau , widht of noisefields
r1 = -db:nfstd:db
r2 = -db:nfstd:db
nfloc = PointF.(collect(product(r1, r2)))[:]
nfscales = [.1PointF(1.0, 1.0) for x in nfloc]  # intensity

nfs = [Noisefield(δ, λ, nfstd) for (δ, λ) in zip(nfloc, nfscales)]
Pms = MarslandShardlow(a, γ, λ, n)
Pahs = Landmarks(a, λ, n, nfs)
###

StateW = PointF
if model == :ms
    dwiener = n
    P = Pms
else
    dwiener = length(nfloc)
    P = Pahs
end

# specify initial landmarks configuration
q0 = [PointF(2.5cos(t), sin(t)) for t in (0:(2pi/n):2pi)[1:n]]  #q0 = circshift(q0, (1,))
p_ = 5*PointF(0.1, 0.1)
p0 = [p_ for i in 1:n]  #
#p0 = [randn(PointF) for i in 1:n]
x0 = State(q0, p0)

#Random.seed!(1234)
w0 = zeros(StateW, dwiener)
W = SamplePath(t, [copy(w0) for s in t])
X = SamplePath(t, [copy(x0) for s in t])
sample!(W, Wiener{Vector{StateW}}())
println("Sample forward process:")
@time solve!(EulerMaruyama!(), X, x0, W, P)
#@time solve!(StratonovichHeun!(), X, x0, W, P)

# compute Hamiltonian along path
ham = [hamiltonian(X.yy[i],Pms) for i in 1:length(t)]

tc(t,T) = t.*(2 .-t/T)
tt_ =  tc(t,T)#tc(t,T)# 0:dtimp:(T)



####################
if obsscheme==:partial
  L = deepmat( [(i==j)*one(Unc) for i in 1:2:2n, j in 1:2n])
  Σ = Diagonal(σobs^2*ones(n*d))

  # observe positions
  v0 = q(X.yy[1])  + σobs * randn(PointF,n)
  rot =  SMatrix{2,2}(cos(θ), sin(θ), -sin(θ), cos(θ))
  vT = [rot * X.yy[end].q[i] + σobs * randn(d)    for i in 1:P.n ]

  Pmsaux = MarslandShardlowAux(Pms, State(vT, zero(vT)))
  if cheat
      Pahsaux = LandmarksAux(Pahs, X.yy[end])
  else
      Pahsaux = LandmarksAux(Pahs, State(vT, zero(vT)))
      #Pahsaux = LandmarksAux(Pahs, State(vT, rand(PointF,Pahs.n)))
  end
end
if obsscheme==:full
  L = deepmat( [(i==j)*one(Unc) for i in 1:2n, j in 1:2n])
  Σ = Diagonal(σobs^2*ones(2n*d))
  Pmsaux = MarslandShardlowAux(Pms,X.yy[end])
  Pahsaux = LandmarksAux(Pahs, X.yy[end])
  v0 = vec(X.yy[1])
  vT = vec(X.yy[end])
end

#Paux = (model==:ms) ? Pmsaux : Pahsaux



# solve backward recursion on [0,T]
# "old" choice (wrong)
#νend = State(zero(vT), zero(vT))
#Hend⁺ = [(i==j)*one(Unc)/ϵ for i in 1:2n, j in 1:2n]  # this is the precison
# "right" choice
if obsscheme==:partial
    #νendT = State(zero(vT), zero(vT))
    νendT = State(randn(PointF,Pahs.n), randn(PointF,Pahs.n))
end
if obsscheme==:full
    νendT = X.yy[end]
end
HendT⁺ = reshape(zeros(Unc,4n^2),2n,2n)
for i in 1:n
    HendT⁺[2i-1,2i-1] = one(Unc)/ϵ  # high variance on positions
    if discrmethod==:lowrank
        HendT⁺[2i,2i] = one(Unc)*10^(-4) # still need to figure out why
    else
        HendT⁺[2i,2i] = one(Unc)/10^(-4)
    end
end
#### perform gpupdate step
νendT , HendT⁺ = gpupdate(νendT,HendT⁺, Σ, L, vT)
νend = copy(νendT)
Hend⁺  = copy(HendT⁺)
Pahsaux = LandmarksAux(Pahs, copy(νend))   # this might be a good idea
Pmsaux = MarslandShardlowAux(Pms, copy(νend))   # this might be a good idea

if model == :ms
    Paux = Pmsaux
else
    Paux = Pahsaux
end


# L, and Σ are ordinary matrices, vT an array of Points
# νend is a state , Hend⁺ a  UncMat

νt =  [copy(νend) for s in tt_]
println("Compute guiding term:")
if discrmethod==:lowrank
    M0 = eigen(deepmat(Hend⁺))
    largest = sortperm(M0.values)[end-ldim+1:end]
    Send = Matrix(Diagonal(M0.values[largest]))
    Uend = M0.vectors[:,largest]
    St = [copy(Send) for s in tt_]
    Ut = [copy(Uend) for s in tt_]
    @time νend, (Send, Uend) = bucybackwards!(LRR(), tt_, νt, (St, Ut), Paux, νend, (Send, Uend))
    Hend⁺ = deepmat2unc(Uend * Send * Uend')
    Ht = map((S,U) -> deepmat2unc(U * inv(S) * U'), St, Ut)  # directly compute Mt
    #Ht = map((S,U) -> LowRank(S,U), St,Ut)
end
if discrmethod==:ralston
    H⁺t = [copy(Hend⁺) for s in tt_]
    @time νend , Hend⁺ = bucybackwards!(Bridge.R3!(), tt_, νt, H⁺t, Paux, νend, Hend⁺)
    Ht = map(H⁺ -> InverseCholesky(lchol(H⁺)),H⁺t)
end
if discrmethod==:psd
    H⁺t = [copy(Hend⁺) for s in tt_]
    @time νend , Hend⁺ = bucybackwards!(Lyap(), tt_, νt, H⁺t, Paux, νend, Hend⁺)
#    println(map(x->isposdef(deepmat(x)),H⁺t))
    Ht = map(H⁺ -> InverseCholesky(lchol(H⁺)),H⁺t)
end

Q = GuidedProposal!(P, Paux, tt_, νt, Ht)
# careful, not a state
νstart , Hstart⁺ = gpupdate(νend , Hend⁺, Σ, L, v0)
xinit = cheat ? x0 : νstart  # or xinit ~ N(νstart, Hstart⁺)
winit = zeros(StateW, dwiener)
XX = SamplePath(tt_, [copy(xinit) for s in tt_])
WW = SamplePath(tt_, [copy(winit) for s in tt_])
sample!(WW, Wiener{Vector{StateW}}())

println("Sample guided bridge proposal:")
@time Bridge.solve!(EulerMaruyama!(), XX, xinit, WW, Q)

include("plotlandmarks.jl")

if model==:ms
    @time llikelihood(LeftRule(), XX, Q; skip = 0)  # won't work for AHS because matrix multilication for Htilde is not defined yet
end

using ForwardDiff
dual(x, i, n) = ForwardDiff.Dual(x, ForwardDiff.Chunk{n}(), Val(i))
dual(x, n) = ForwardDiff.Dual(x, ForwardDiff.Chunk{n}(), Val(0))

#using Flux
xinitv = deepvec(xinit)

xinitv = map(i->dual(xinitv[i], i <= 2 ? i : 0, 2), 1:length(xinitv))

xinitnew = deepvec2state(xinitv)
x = copy(xinitnew)

#lux.Tracker.gradient(x -> Bridge._b!((1,0.0), deepvec2state(x), deepvec2state(x), P), deepvec(xinit))
Bridge.b!(0.0, x, copy(x), P)

import Bridge;

#include(joinpath(dirname(pathof(Bridge)), "..", "landmarks/patches.jl"))
#include(joinpath(dirname(pathof(Bridge)), "..", "landmarks/models.jl"))

XX = Bridge.solve(EulerMaruyama!(), xinitnew, WW, P)



function obj(xinitv)
    xinit = deepvec2state(xinitv)
    sample!(WW, Wiener{Vector{StateW}}())
    XXᵒ = Bridge.solve(EulerMaruyama!(), xinit, WW, Q)
    llikelihood(LeftRule(), XXᵒ, Q; skip = 0)
end

let
    x = deepvec(xinit)
    ϵ = 1e-7
    for i in 1:1000
        ∇x = ForwardDiff.gradient(obj, x)
        x .+= ϵ*∇x
        println(deepvec2state(x))
    end
end