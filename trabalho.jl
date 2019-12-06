using HMMBase, Distributions
using CSV
using DataFrames
using Statistics
using JuMP
using Ipopt
# using Plots

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

function simulate(hmm, nsim, v)
    nstates, obs_dim = size(hmm)
    nstocks =obs_dim
    m = zeros(nsim, nstates, nstocks)
    p = zeros(nsim, nstates, nstates)
    hid_state = zeros(nsim, nstates)
    for i in 1:nstates
         z,y = rand(hmm, nsim; init=i, seq=true) 
         α , _ = forward(hmm, y)
         for s in 1:nsim
            m[s, i, :] = y[s,:]


            p[s, i, :] = α[s,:]#/nsim

            hid_state[s,i] =z[1]

            # using Distributions, HMMBase
            # hmm = HMM([0.9 0.1; 0.1 0.9], [Normal(0,1), Normal(10,1)])
            # y = rand(hmm, 1)
            # probs, tot = HMMBase.forward(hmm, y)
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
nsim = 10
scenarios, probabilities, hid_state = simulate(hmm, nsim, v)

# optimization problem
using Xpress
η = 0.5
γ = (maximum(v) - minimum(v) )/maximum(v)
W=1
function optim(μ, Σ, η, scenarios, probabilities, γ, hmm)
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

    m = JuMP.Model(with_optimizer(Xpress.Optimizer));
    @variable(m, x[1:nvar] >=0);
    @variable(m, θ);
    @variable(m, z);
    @variable(m, δ[s = 1:S] >= 0);
    @constraint(m, sum(x) == 1*W);
    @constraint(m, θ  <= 1*W);
    @constraint(m, θ == z + sum(δ[s] * probabilities[s,j,k] * transition_mat[j,k] * init_prob[j] for s = 1:S, k = 1:K, j=1:J)/(1-η)) ;
    
    @constraint(m,[s = 1:S], δ[s]  >= sum(scenarios[s, k, i]*x[i]* probabilities[s,j,k] * transition_mat[j,k] *init_prob[j] for i=1:nvar, k = 1:K, j=1:J ) - z);

    @objective(m, Max, sum(x[i] * scenarios[s,k,i] * probabilities[s,j,k] * transition_mat[j,k]* init_prob[j] for s in 1:S, i in 1:nvar, k = 1:K, j=1:J));
    optimize!(m);
    termination_status(m);
    value.(x)
    value.(θ)
    value.(δ)
    value.(z)
    sum(value.(x)[i] * scenarios[s,1,i] *1/10 for s in 1:S, i=1:nvar)
    value.(z) - sum(value.(x)[i] * scenarios[s,1,i] *1/10 for s in 1:S, i=1:nvar)
    scenarios[s,1,:]'*value.(x)
    # # portfolio variance
    σ = sqrt(sum(value(x[i])*value(x[j])*Σ[i,j] for i in 1:nvar, j in 1:nvar))
    return x, σ, termination_status(m), value(θ)
end
using MathOptFormat
using MathOptInterface
lp_model = MathOptFormat.LP.Model()
MathOptInterface.copy_to( lp_model , backend( m ) , copy_names = true )
MathOptInterface.write_to_file( lp_model , joinpath( pwd(), "prb.lp" ) )