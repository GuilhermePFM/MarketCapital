using HMMBase, Distributions
using CSV
using DataFrames
using Statistics
using JuMP
using Ipopt
using Plots

cd("D:\\Repositories\\PUC\\Mercado de Capitais\\Code")

# Reading Data
fname = "stock_returns.csv"
hist_stocks = CSV.read(fname; types = [Float64 for i in 1:10])
hist_stocks = dropmissing(hist_stocks)

# 1) Estimating expected return
desc = describe(hist_stocks)
μ = aggregate(hist_stocks, mean)
μ = desc[:mean]


# 2) Estimating covariance matrix
Σ = cov(Matrix(hist_stocks); dims=1, corrected=true)

h = [ String(split(String(i),".")[1]) for i in names(hist_stocks)]
i=1
h
plot(Matrix(hist_stocks), label = ["AAPL"  "MSFT"  "GOOG"  "MCD"  "GE"  "GSPC"  "VALE"  "FB"  "AMZN"  "IBM"])
xlabel!("Time")
ylabel!("Return")
title!("Stock returns")

function simulate(hmm, nsim, v)
    nstates, obs_dim = size(hmm)
    nstocks =obs_dim
    m = zeros(nsim, nstates, nstocks)
    p = zeros(nsim, nstates, nstates)
    hid_state = zeros(nsim, nstates)
    for i in 1:nstates
        for s in 1:nsim
            α=[NaN]
            while any(isnan.(α))
                y = rand(hmm, 1; init=i) 


                v = y#vcat(Matrix(hist_stocks), y) 
                α , _ = forward(hmm, v)

                m[s, i, :] = y

                p[s, i, :] = α #[end,:] #/nsim
                # using Distributions, HMMBase
                # hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
                # for i in 1:10
                #     y = rand(hmm, 10)
                #     probs, tot = HMMBase.forward(hmm, y)
                #     @show probs
                # end
            end
         end
    end
    return m,p,hid_state
end

# HMM
μ = [i for i in μ]
nobs = length(μ)

n=3
a = [1/n for i in 1:n]
A = ones(n,n).*1/n
B = [MvNormal(μ, Σ) for j in 1:n]
hmm = HMM(A, B)

# Fitting
v = Matrix(hist_stocks)
hmm, hist = fit_mle(hmm, v, display = :iter, init = :kmeans)

# simulation
nsim = 1000
scenarios, probabilities, hid_state = simulate(hmm, nsim, v)

# optimization problem
using Xpress
η = 0.5
γ = (maximum(v) - minimum(v) )/maximum(v)
W=1
optim(μ, Σ, 0.25, scenarios, probabilities, 0.9, hmm)
optim(μ, Σ, 0.5, scenarios, probabilities, 0.9, hmm)
optim(μ, Σ, 0.65, scenarios, probabilities, 0.9, hmm)
optim(μ, Σ, 0.85, scenarios, probabilities, 0.9, hmm)
optim(μ, Σ, 0.95, scenarios, probabilities, 0.9, hmm)
function optim(μ, Σ, η, scenarios, probabilities, γ, hmm)
    nvar = size(μ,1)
    ncen = size(scenarios,2)
    nstates, obs_dim = size(hmm)

    # sets
    J = nstates
    S = nsim
    K = nstates

    # probabilities
    # init_prob = posteriors(hmm, v)[end,:]
    transition_mat = hmm.A
    init_prob = [1/K for i in 1:K]
    # probabilities = ones(S,K,K) .*1/S
    # transition_mat = [1/3 1/3 1/3;1/3 1/3 1/3;1/3 1/3 1/3]

    m = JuMP.Model(with_optimizer(Xpress.Optimizer));
    @variable(m, x[1:nvar] >=0);
    @variable(m, θ);
    @variable(m, z);
    @variable(m, δ[s = 1:S] >= 0);
    @constraint(m, sum(x) == W);
    @constraint(m, θ  <= γ*W);
    @constraint(m, θ == z + sum(δ[s] * probabilities[s,j,k] * transition_mat[j,k] * init_prob[j] * 1/S for s = 1:S, k = 1:K, j=1:J)/(1-η)) ;
    

    @constraint(m,[s = 1:S], δ[s]  >= sum(scenarios[s, k, i]*x[i]* probabilities[s,j,k] * transition_mat[j,k] *init_prob[j] for i=1:nvar, k = 1:K, j=1:J ) - z);

    # vetor com renda por cenario para estado ini j = scenarios[s,j,:] * x
    # prob estado ini j = init_prob[j]
    # prob transiçao de j para k = transition_mat[j,k]
    # probabilidiade do cenário s dado que estado eh k = probabilitis[s,k]
    @objective(m, Max, sum(x' * scenarios[s,j,:] * probabilities[s,j,k] * transition_mat[j,k] * init_prob[j]  * 1/S for s in 1:S,j=1:J,k=1:K));
    optimize!(m);

    return sum(value(x[i]) * scenarios[s,k,i] * probabilities[s,j,k] * transition_mat[j,k]* init_prob[j]  * 1/S  for s in 1:S, i in 1:nvar, k = 1:K, j=1:J), value.(θ), value.(x), termination_status(m)
end
function optim3(μ, Σ, η, scenarios, probabilities, γ, hmm)
    nvar = size(μ,1)
    ncen = size(scenarios,2)
    nstates, obs_dim = size(hmm)

    # sets
    J = nstates
    S = nsim
    K = nstates

    # probabilities
    init_prob = posteriors(hmm, v)[end,:]
    transition_mat = hmm.A
    # init_prob = [1/K for i in 1:K]
    # probabilities = ones(S,K,K) .*1/S
    # transition_mat = [1/3 1/3 1/3;1/3 1/3 1/3;1/3 1/3 1/3]

    m = JuMP.Model(with_optimizer(Xpress.Optimizer));
    @variable(m, x[1:nvar] >=0);
    @variable(m, θ);
    @variable(m, z);
    @variable(m, δ[s = 1:S] >= 0);
    @constraint(m, sum(x) == W);
    @constraint(m, θ  <= γ*W);
    @constraint(m, θ == z + sum(δ[s] * probabilities[s,j,k] * transition_mat[j,k] * init_prob[j] * 1/S for s = 1:S, k = 1:K, j=1:J)/(1-η)) ;

    @constraint(m,[s = 1:S], δ[s]  >= sum(scenarios[s, k, i]*x[i]* probabilities[s,j,k] * transition_mat[j,k] *init_prob[j] for i=1:nvar, k = 1:K, j=1:J ) - z);

    # vetor com renda por cenario para estado ini j = scenarios[s,j,:] * x
    # prob estado ini j = init_prob[j]
    # prob transiçao de j para k = transition_mat[j,k]
    # probabilidiade do cenário s dado que estado eh k = probabilitis[s,k]
    @objective(m, Max, sum(x' * scenarios[s,j,:] * probabilities[s,j,k] * transition_mat[j,k] * init_prob[j]  * 1/S for s in 1:S,j=1:J,k=1:K));
    optimize!(m);

    return sum(value(x[i]) * scenarios[s,k,i] * probabilities[s,j,k] * transition_mat[j,k]* init_prob[j]  * 1/S  for s in 1:S, i in 1:nvar, k = 1:K, j=1:J), value.(θ), value.(x), termination_status(m)
end
function optim4(μ, Σ, η, scenarios, probabilities, γ, hmm)
    nvar = size(μ,1)
    ncen = size(scenarios,2)
    nstates, obs_dim = size(hmm)

    # sets
    J = nstates
    S = nsim
    K = nstates

    # probabilities
    # init_prob = posteriors(hmm, v)[end,:]
    transition_mat = hmm.A
    init_prob = [1/K for i in 1:K]
    probabilities = ones(S,K,K) .*1/K
    # transition_mat = [1/3 1/3 1/3;1/3 1/3 1/3;1/3 1/3 1/3]

    m = JuMP.Model(with_optimizer(Xpress.Optimizer));
    @variable(m, x[1:nvar] >=0);
    @variable(m, θ);
    @variable(m, z);
    @variable(m, δ[s = 1:S] >= 0);
    @constraint(m, sum(x) == W);
    @constraint(m, θ  <= γ*W);
    @constraint(m, θ == z + sum(δ[s] * probabilities[s,j,k] * transition_mat[j,k] * init_prob[j] * 1/S for s = 1:S, k = 1:K, j=1:J)/(1-η)) ;

    @constraint(m,[s = 1:S], δ[s]  >= sum(scenarios[s, k, i]*x[i]* probabilities[s,j,k] * transition_mat[j,k] *init_prob[j] for i=1:nvar, k = 1:K, j=1:J ) - z);

    # vetor com renda por cenario para estado ini j = scenarios[s,j,:] * x
    # prob estado ini j = init_prob[j]
    # prob transiçao de j para k = transition_mat[j,k]
    # probabilidiade do cenário s dado que estado eh k = probabilitis[s,k]
    @objective(m, Max, sum(x' * scenarios[s,j,:] * probabilities[s,j,k] * transition_mat[j,k] * init_prob[j]  * 1/S for s in 1:S,j=1:J,k=1:K));
    optimize!(m);

    return sum(value(x[i]) * scenarios[s,k,i] * probabilities[s,j,k] * transition_mat[j,k]* init_prob[j]  * 1/S  for s in 1:S, i in 1:nvar, k = 1:K, j=1:J), value.(θ), value.(x), termination_status(m)
end
function optim2(μ, Σ, η, scenarios, probabilities, γ, hmm)
    nvar = size(μ,1)
    ncen = size(scenarios,2)
    nstates, obs_dim = size(hmm)

    # sets
    J = nstates
    S = nsim
    K = nstates

    m = JuMP.Model(with_optimizer(Xpress.Optimizer));
    @variable(m, x[1:nvar] >=0);
    @variable(m, θ);
    @variable(m, z);
    @variable(m, δ[s = 1:S, k=1:K] >= 0);
    @constraint(m, sum(x) == W);
    @constraint(m, θ  <= γ*W);
    @constraint(m, θ == z + sum(δ[s,k] * 1/S * 1/K for s = 1:S, k=1:K)/(1-η)) ;
    

    @constraint(m,[s = 1:S, k=1:K], δ[s,k]  >= scenarios[s, k, :]' * x - z);

    # vetor com renda por cenario para estado ini j = scenarios[s,j,:] * x
    # prob estado ini j = init_prob[j]
    # prob transiçao de j para k = transition_mat[j,k]
    # probabilidiade do cenário s dado que estado eh k = probabilitis[s,k]
    @objective(m, Max, sum(x' * scenarios[s,j,:] * 1/J * 1/S for s in 1:S,j=1:J));
    optimize!(m);

    return sum(value(x[i]) * scenarios[s,k,i] * 1/S * 1/K for s in 1:S, i in 1:nvar, k = 1:K), value.(θ), value.(x), termination_status(m)
end

function plot1_variando_gamma()
    Η = collect(0.01:0.05:0.99)
    eta_vec = []
    return_vec = []
    alpha_vec = []
    x_vec = []
    eta_vec2 = []
    return_vec2 = []
    alpha_vec2 = []
    x_vec2 = []
    γ = 0.9
    for η in Η
        @show η
        r, θ, x, stat = optim(μ, Σ, η, scenarios, probabilities, γ, hmm)
        r2, θ2, x2, stat = optim2(μ, Σ, η, scenarios, probabilities, γ, hmm)
        push!(eta_vec, η)
        push!(return_vec, r)
        push!(x_vec, x)
        push!(eta_vec2, η)
        push!(return_vec2, r2)
        push!(x_vec2, x2)
    end
    plot(eta_vec, return_vec, label = "HMM")
    # plot!(eta_vec, return_vec, label = "HMM 5 estados")
    plot(eta_vec2, return_vec2, label = "HMM expected return")
    xlabel!("Risk aversion")
    ylabel!("Return")
    title!("Risk-return")

    # compare HHM numbers
    # HMM

    for n in 3:5
        a = [1/n for i in 1:n]
        A = ones(n,n).*1/n
        B = [MvNormal(μ, Σ) for j in 1:n]
        hmm = HMM(A, B)

        # Fitting
        hmm, hist = fit_mle(hmm, v, display = :iter, init = :kmeans)

        # simulation
        nsim = 1000
        scenarios, probabilities, hid_state = simulate(hmm, nsim, v)

        Η = collect(0.01:0.05:0.99)
        eta_vec = []
        return_vec = []
        alpha_vec = []
        x_vec = []
        eta_vec2 = []
        return_vec2 = []
        alpha_vec2 = []
        x_vec2 = []
        γ = 0.9
        for η in Η
            @show η
            r, θ, x, stat = optim(μ, Σ, η, scenarios, probabilities, γ, hmm)
            r2, θ2, x2, stat = optim2(μ, Σ, η, scenarios, probabilities, γ, hmm)
            push!(eta_vec, η)
            push!(return_vec, r)
            push!(x_vec, x)
            push!(eta_vec2, η)
            push!(return_vec2, r2)
            push!(x_vec2, x2)
        end
        plot!(eta_vec, return_vec, label = "HMM $(n) estados")
        # plot(eta_vec, return_vec, label = "HMM 2 estados")
    end
#GRAFICO COM X MUdando em funcao da aversao ao risco
    Γ = collect(0.8:0.05:2)
    return_vec = []
    gamma_vec = []
    return_vec2 = []
    gamma_vec2 = []
    x_vec = []
    for γ  in Γ
        @show γ
        r, θ, x, stat = optim(μ, Σ, 0.95, scenarios, probabilities, γ, hmm)
        r2, θ2, x, stat = optim2(μ, Σ, 0.95, scenarios, probabilities, γ, hmm)
        push!(gamma_vec, θ)
        push!(return_vec, r)
        push!(gamma_vec2, θ2)
        push!(return_vec2, r2)
        push!(x_vec, x)
    end
    plot(gamma_vec, return_vec, label = "HMM return")
    plot!(gamma_vec2, return_vec2, label = "Equiprobable return")
    xlabel!("Gamma")
    ylabel!("Return")
    title!("CVaR limit")

    npontos = length(x_vec2)
    nagentes = 10
    vecvec = [zeros(npontos) for i in 1:nagentes]
    for i in 1:npontos, j in 1:nagentes
        vecvec[j][i] = x_vec2[i][j]
    end
    plot(eta_vec2, vecvec,  label = ["AAPL"  "MSFT"  "GOOG"  "MCD"  "GE"  "GSPC"  "VALE"  "FB"  "AMZN"  "IBM"])
    # plot(gamma_vec, vecvec)
    for i in 2:nagentes
        plot!(eta_vec2, vecvec[i])
        # plot!(gamma_vec, vecvec[i])
    end
    xlabel!("CVar-alpha")
    ylabel!("Percentage of portfolio")
    title!("Portfolio allocation HMM")
end

using MathOptFormat
using MathOptInterface
lp_model = MathOptFormat.LP.Model()
MathOptInterface.copy_to( lp_model , backend( m ) , copy_names = true )
MathOptInterface.write_to_file( lp_model , joinpath( pwd(), "prb.lp" ) )