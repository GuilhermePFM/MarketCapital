using CSV
using DataFrames
using Statistics
using JuMP
using Ipopt
using Plots
using HypothesisTests

cd("D:\\Repositories\\PUC\\Mercado de Capitais\\Code")

# Reading Data
fname = "stocks.csv"
fmarkt= "market_index.csv"
hist_stocks = CSV.read(fname; types = [String, Float64, Float64, Float64, Float64, Float64, Float64])
hist_stocks = dropmissing(hist_stocks)

hist_market = CSV.read(fmarkt; types = [String, Float64])
hist_market = dropmissing(hist_market)

# 1) Estimating expected return
desc = describe(hist_stocks)
μ = aggregate(hist_stocks[2:end], mean)
μ = desc[:mean][2:end]


# 2) Estimating covariance matrix
Σ = cov(Matrix(hist_stocks[2:end]); dims=1, corrected=true)

hist_stocks = CSV.read(fname; types = [String, Float64, Float64, Float64, Float64, Float64, Float64])
hist_market = CSV.read(fmarkt; types = [String, Float64])
joint_df = hcat(hist_stocks[:,2:end-1], hist_market[:,2])
joint_df = dropmissing(joint_df)
rm = describe(joint_df)[:mean][end]

Σm = cov(Matrix(joint_df); dims=1, corrected=true)

# 3) Get the efficient frontier with markowitz analytical expression
function markowitz_frontier(μ, Σ)
    min_mu = minimum(μ)
    max_mu = maximum(μ)
    R_vec = collect(min_mu:0.1:2*max_mu)
    sigma_vec = []
    sigma_vec2 = []
    for R in R_vec
        # sigma, status = markowitz_variance(μ, Σ, R)
        sigma, status = markowitz_analytical(μ, Σ, R)
        sigma2, status = markowitz_variance_modified(μ, Σ, R)
        #if status == OPTIMAL::TerminationStatusCode
            push!(sigma_vec, sigma2)
            push!(sigma_vec2, sigma2)
        #end
    end
    #@show sigma_vec - sigma_vec2
    plot(sigma_vec, R_vec.-1)
    plot!(sigma_vec2, R_vec.-1)
    xlabel!("sigma")
    ylabel!("r")
    title!("Comparação")
end
function markowitz_analytical(μ, Σ, R)
    nvar = length(μ)
    M = [Σ -ones(nvar,1) -μ;
        ones(1,nvar) 0 0;
        μ' 0 0]
    RHS = [zeros(1, nvar) 1 R]'
    sol = M \ RHS
    x=sol[1:end-2]
    π=sol[end-1] 
    λ=sol[end]
    # portfolio variance
    σ = sum(x[i]*x[j]*Σ[i,j] for i in 1:nvar, j in 1:nvar)
    return σ, 1
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
    σ = sqrt(sum(value(x[i])*value(x[j])*Σ[i,j] for i in 1:nvar, j in 1:nvar))
    return σ, termination_status(m)
end
function markowitz_variance_modified(μ, Σ, R)
    nvar = size(μ,1)
    m = JuMP.Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, x[1:nvar])
    @constraint(m, sum(x) == 1)
    @constraint(m, sum(μ[i]*x[i] for i in 1:nvar) >= R)
    @objective(m, Min, x'*Σ*x)
    optimize!(m)

    # portfolio variance
    σ = sum(value(x[i])*value(x[j])*Σ[i,j] for i in 1:nvar, j in 1:nvar)
    return σ, termination_status(m)
end
function return_excess_maximization(μ, Σ, R, rf)
    nvar = length(μ)
    v = Σ \ (μ .-rf)
    x = v / sum(v)
    return x
end

# 4) Calculate the analytical maximum expected return under variance portfolio
function max_exp_ret(r, Σ, rf)
    #μk = rf + Σk'v
    v = (μ .-rf) \ Σ
    x = v / sum(v)
    return x
end

# 5) Calculate the CAPM of the assets
function CAPM(r, Σm, rm, rf,i)
    beta = Σm[1:end-1,end] / Σm[end,1]

    # estimates the return of each asset
    # r = rf .+ beta * (rm - rf)
    alpha = r .- rf .- beta[i] * (rm - rf)
    a =OneSampleTTest(alpha)
    println(a)
end
for i in 1:5
    r = dropmissing(hist_stocks)[:,i+1]
    CAPM(r, Σm, rm, 0,i)
end


function CVAR()

end

function Var(l, alpha)
    ls = sort(l)
    n = length(l)
    maior_que = alpha * n
    
end