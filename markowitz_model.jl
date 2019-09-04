using CSV
using DataFrames
using Statistics
using JuMP
using Ipopt
using Plots

cd("D:\\Repositories\\PUC\\Mercado de Capitais\\Code")

# Reading Data
fname = "stocks.csv"
hist_stocks = CSV.read(fname; types = [String, Float64, Float64, Float64, Float64, Float64, Float64, Float64])
hist_stocks = dropmissing(hist_stocks)

# 1) Estimating expected return
desc = describe(hist_stocks)
μ = aggregate(hist_stocks[2:end], mean)
μ = desc[:mean][2:end]

# 2) Estimating covariance matrix
Σ = cov(Matrix(hist_stocks[2:end]); dims=1, corrected=true)

# 3) Get the efficient frontier with markowitz analytical expression
function markowitz_frontier(μ, Σ)
    min_mu = minimum(μ)
    max_mu = maximum(μ)
    R_vec = [ i for i in min_mu:5*max_mu]
    sigma_vec = []
    for R in R_vec
        sigma, status = markowitz_variance(μ, Σ, R)
        #if status == OPTIMAL::TerminationStatusCode
            push!(sigma_vec, sigma)
        #end
    end
    plot(sigma_vec, R_vec.-1, labels=["efficient frontier"])
    xlabel!("sigma")
    ylabel!("r")
end
function markowitz_variance(μ, Σ, R)
    nvar = size(μ,1)
    m = JuMP.Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, x[1:nvar])
    @constraint(m, sum(x) == 1)
    @constraint(m, sum(μ[i]*x[i] for i in 1:nvar) == R)
    @objective(m, Min, x'*Σ*x)
    optimize!(m)

    # portfolio variance
    σ = sum(value(x[i])*value(x[j])*Σ[i,j] for i in 1:nvar, j in 1:nvar)
    return σ, termination_status(m)
end