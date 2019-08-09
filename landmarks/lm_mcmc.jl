"""
    Perform mcmc or sgd for landmarks model using the LM-parametrisation
    tt_:      time grid
    (xobs0,xobsT): observations at times 0 and T
    mT: vector of momenta at time T used for constructing guiding term
    P: target process
    model: either :ms (Marsland-Shardlow) or :ahs (Arnaudon-Holm-Sommer)
    sampler: either sgd (stochastic gradient descent) or mcmc (Markov Chain Monte Carlo)
    dataset: dataset to extract xobs0 and xobsT
    xinit: initial guess on starting state
    δ: parameter for Langevin updates on initial state
    ITER: number of iterations
    subsamples: vector of indices of iterations that are to be saved
    prior_a: prior on parameter a
    prior_γ: prior on parameter γ
    σ_a: parameter determining update proposal for a [update a to aᵒ as aᵒ = a * exp(σ_a * rnorm())]
    σ_γ: parameter determining update proposal for γ [update γ to γᵒ as γᵒ = γ * exp(σ_γ * rnorm())]
    ourdir: output directory for animation
    makefig: logical flag for making figures
    showmomenta: logical flag if momenta are also drawn in figures

    Returns:
    Xsave: saved iterations of all states at all times in tt_
    parsave: saved iterations of all parameter updates ,
    objvals: saved values of stochastic approximation to loglikelihood
    perc_acc: acceptance percentages (bridgepath - inital state)

"""
function lm_mcmc(tt_, (xobs0,xobsT), mT, P,
        model, sampler, dataset,
        xinit, δ, ITER, subsamples,
        prior_a, prior_γ, σ_a, σ_γ,
        outdir; makefig=true, showmomenta=false)

    StateW = PointF
    dwiener = dimwiener(P)
    L0 = LT = [(i==j)*one(UncF) for i in 1:2:2P.n, j in 1:2P.n]
    Σ0 = ΣT = [(i==j)*σobs^2*one(UncF) for i in 1:P.n, j in 1:P.n]
    μT = zeros(PointF,P.n)
    if isa(P,Landmarks)
        Paux = LandmarksAux(P, State(xobsT, mT))
    elseif isa(P,MarslandShardlow)
        Paux = MarslandShardlowAux(P, State(xobsT, mT))
    end

    println("compute guiding term:")
    Lt, Mt⁺, μt, Ht = initLMμH(tt_,(LT,ΣT,μT))
    Q = construct_guidedproposal!(tt_, (Lt, Mt⁺ , μt, Ht), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)
    Ltᵒ, Mt⁺ᵒ, μtᵒ, Htᵒ = initLMμH(tt_,(LT,ΣT,μT)) # needed when doing parameter estimation

    println("Sample guided proposal:")
    X = initSamplePath(tt_, xinit)
    W = initSamplePath(tt_,  zeros(StateW, dwiener))
    sample!(W, Wiener{Vector{StateW}}())

    ll = simguidedlm_llikelihood!(LeftRule(), X, xinit, W, Q; skip=sk)
    if makefig
        if isa(P,Landmarks)
            plotlandmarkpositions(X,P.n,model,xobs0,xobsT,P.nfs;db=4)
        elseif isa(P,MarslandShardlow)
            plotlandmarkpositions(X,P.n,model,xobs0,xobsT,0;db=4)
        end
    end

    # saving objects
    objvals =   Float64[]  # keep track of (sgd approximation of the) loglikelihood
    acc = zeros(3) # keep track of mcmc accept probs (first comp is for CN update; 2nd component for langevin update on initial momenta, 3rd parameter updates)
    #Xsave = zeros(length(subsamples), length(tt_) * P.n * 2 * d )
    Xsave = typeof(zeros(length(tt_) * P.n * 2 * d))[]
    parsave = Vector{Float64}[]
    push!(Xsave, convert_samplepath(X))
    push!(objvals, ll)
    push!(parsave,[P.a, getγ(P)])


    mask = deepvec(State(0 .- 0*xinit.q, 1 .- 0*(xinit.p)))  # only optimize momenta
    mask_id = (mask .> 0.1) # get indices that correspond to momenta

    # initialisation
    Xᵒ = initSamplePath(tt_, xinit)
    Wᵒ = initSamplePath(tt_,  zeros(StateW, dwiener))
    Wnew = initSamplePath(tt_,  zeros(StateW, dwiener))
    x = deepvec(xinit)
    xᵒ = deepcopy(x)
    ∇x = deepcopy(x)
    ∇xᵒ = deepcopy(x)
    result = DiffResults.GradientResult(x) # allocate
    resultᵒ = DiffResults.GradientResult(xᵒ)

    if makefig
        xobs0comp1 = extractcomp(xobs0,1)
        xobs0comp2 = extractcomp(xobs0,2)
        xobsTcomp1 = extractcomp(xobsT,1)
        xobsTcomp2 = extractcomp(xobsT,2)
    end

    # start iterations
    anim =    @animate for i in 1:ITER
        if makefig
            drawpath(i,x,X,objvals,x0,(xobs0comp1,xobs0comp2,xobsTcomp1,xobsTcomp2))
        end
        println("iteration $i")

        if sampler==:sgd
            δ = 0.01*ϵstep(i)
        end
        #println("------")
        #print(Q.target.a)

        (x , W, X), ll, obj, acc  = updatepath!(X,Xᵒ,W,Wᵒ,Wnew,ll,x,xᵒ,∇x, ∇xᵒ,result, resultᵒ,
                                    sampler,Q,
                                    mask, mask_id, δ, ρ, acc)

        # HERE updatepars!, i.e. an update on (P, Paux, Q)
        updatepars = true
        if updatepars
            aᵒ = P.a * exp(σ_a * randn())
            γᵒ = getγ(P) * exp(σ_γ * randn())
            if isa(P,MarslandShardlow)
                Pᵒ = MarslandShardlow(aᵒ,γᵒ,P.λ, P.n)
                Pauxᵒ = MarslandShardlowAux(aᵒ,γᵒ,P.λ,Paux.xT,P.n)
            elseif isa(P,Landmarks)
                nfs = construct_nfs(P.db, P.nfstd, γᵒ) # need ot add db and nfstd to struct Landmarks
                Pᵒ = Landmarks(aᵒ,P.n,P.db,P.nfstd,nfs)
                Pauxᵒ = LandmarksAux(aᵒ,Paux.xT,P.n,nfs)
            end
            println("compute Qᵒ")
            Qᵒ = construct_guidedproposal!(tt_, (Ltᵒ, Mt⁺ᵒ, μtᵒ, Htᵒ), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), Pᵒ, Pauxᵒ)
            llᵒ = simguidedlm_llikelihood!(LeftRule(), Xᵒ, deepvec2state(x), W, Qᵒ; skip=sk)
            A = logpdf(prior_a,aᵒ) - logpdf(prior_a,P.a) + logpdf(prior_γ,γᵒ) - logpdf(prior_γ,getγ(P)) +
                    llᵒ - ll +
                    logpdf(LogNormal(log(aᵒ),σ_a),P.a)- logpdf(LogNormal(log(P.a),σ_a),aᵒ)+
                    logpdf(LogNormal(log(γᵒ),σ_γ),getγ(P))- logpdf(LogNormal(log(getγ(P)),σ_γ),γᵒ)
            println("logaccept for parameter update ", round(A;digits=4))
            if log(rand()) <= A  # assume symmetric proposal and uniform prior, adjust later
                println("parameter update accepted")
                P, Pᵒ = Pᵒ, P
                X, Xᵒ = Xᵒ, X
                Paux, Pauxᵒ = Pauxᵒ, Paux
                Q, Qᵒ = Qᵒ, Q
                Ltᵒ, Mt⁺ᵒ, μtᵒ, Htᵒ, Lt, Mt⁺, μt, Ht = Lt, Mt⁺, μt, Ht, Ltᵒ, Mt⁺ᵒ, μtᵒ, Htᵒ
                acc[3] +=1
            end
        end


        println()
        # save some of the results
        if i in subsamples
            #push!(Xsave, copy(X))
            push!(Xsave, convert_samplepath(X))
            push!(parsave, [P.a, getγ(P)])
        end
        push!(objvals, obj)
        if makefig && (i==ITER)
            drawpath(ITER,x,X,objvals,x0,(xobs0comp1,xobs0comp2,xobsTcomp1,xobsTcomp2))
        end
    end

    fn = "me"*"_" * string(model) * "_" * string(sampler) *"_" * string(dataset)
    gif(anim, outdir*fn*".gif", fps = 20)
    mp4(anim, outdir*fn*".mp4", fps = 20)

    # drawobjective(objvals)

    perc_acc = 100acc/ITER
    println("Acceptance percentages (bridgepath - inital state - parameters): ",perc_acc)
    Xsave, parsave, objvals, perc_acc
end



function drawpath(i,x,X,objvals,x0,(xobs0comp1,xobs0comp2,xobsTcomp1,xobsTcomp2);showmomenta=false)
        s = deepvec2state(x).p
        s0 = x0.p # true momenta

        # plot initial and final shapes
        pp = Plots.plot(xobs0comp1, xobs0comp2,seriestype=:scatter, color=:black,label="q0", title="Landmark evolution")
        Plots.plot!(pp, repeat(xobs0comp1,2), repeat(xobs0comp2,2),seriestype=:path, color=:black,label="")
        Plots.plot!(pp, xobsTcomp1, xobsTcomp2,seriestype=:scatter , color=:orange,label="qT") # points move from black to orange
        Plots.plot!(pp, repeat(xobsTcomp1,2), repeat(xobsTcomp2,2),seriestype=:path, color=:orange,label="")

        if showmomenta
            Plots.plot!(pp, extractcomp(s,1), extractcomp(s,2),seriestype=:scatter ,
             color=:blue,label="p0 est") # points move from black to orange)
            Plots.plot!(pp, extractcomp(s0,1), extractcomp(s0,2),seriestype=:scatter ,
              color=:red,label="p0",markersize=5) # points move from black to orange)
              xlims!(-3,3)
              ylims!(-4,3)
        else
            xlims!(-3,3)
            ylims!(-4,2)
        end

        outg = [Any[X.tt[i], [X.yy[i][CartesianIndex(c, k)][l] for l in 1:d, c in 1:2]..., "point$k"] for k in 1:n, i in eachindex(X.tt) ][:]
        dfg = DataFrame(time=extractcomp(outg,1),pos1=extractcomp(outg,2),pos2=extractcomp(outg,3),mom1=extractcomp(outg,4),mom2=extractcomp(outg,5),pointID=extractcomp(outg,6))
        for j in 1:n
            #global pp
            el1 = dfg[:pointID].=="point"*"$j"
            dfg1 = dfg[el1,:]
            Plots.plot!(pp,dfg1[:pos1], dfg1[:pos2],label="")
        end

        pp2 = Plots.plot(collect(1:i), objvals[1:i],seriestype=:scatter ,color=:blue,markersize=1.5,label="",title="Loglikelihood approximation")
        Plots.plot!(pp2, collect(1:i), objvals[1:i] ,color=:blue,label="")
        xlabel!(pp2,"iteration")
        ylabel!(pp2,"stoch log likelihood")
        xlims!(0,ITER)

        l = @layout [a  b]
        Plots.plot(pp,pp2,background_color = :ivory,layout=l , size = (900, 500) )
end


function drawobjective(objvals)
    ITER = length(objvals)
    sc2 = Plots.plot(collect(1:ITER), objvals,seriestype=:scatter ,color=:blue,markersize=1.2,label="",title="Loglikelihood approximation")
    Plots.plot!(sc2, collect(1:ITER), objvals ,color=:blue,label="")
    xlabel!(sc2,"iteration")
    ylabel!(sc2,"stoch log likelihood")
    xlims!(0,ITER)
    display(sc2)
    png(sc2,"stochlogp.png")
end

"""
    Useful for storage of a samplepath of states
    Ordering is as follows:
    1) time
    2) landmark nr
    3) for each landmark: q1, q2 p1, p2

    With m time-steps, n landmarks, this entails a vector of length m * n * 2 * d
"""
function convert_samplepath(X)
    VA = VectorOfArray(map(x->deepvec(x),X.yy))
    vec(convert(Array,VA))
end


if TEST
    xinit = State(xobs0, zeros(PointF,n))
    xinit = State(xobs0, [Point(-1.0,3.0)/P.n for i in 1:P.n])
    ITER = 10
    lm_mcmc(tt_, (LT,ΣT,μT), (L0,Σ0), (xobs0,xobsT), P, Paux, model, sampler, dataset, xinit, δ, ITER; makefig=true)
end

# change parameter values and update
function updateguidedproposal!((α,γ), tt_, (Lt, Mt⁺ , μt, Ht), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P::MarslandShardlow, Paux, Q)
    P = MarslandShardlow(α, γ, P.λ, P.n)
    if model == :ms
        Paux = MarslandShardlowAux(P, State(xobsT, mT))
    else
        Paux = LandmarksAux(P, State(xobsT, mT))
    end
    Q .= construct_guidedproposal!(tt_, (Lt, Mt⁺ , μt, Ht), (LT,ΣT,μT), (L0, Σ0), (xobs0, xobsT), P, Paux)
end