include("markowitz_model.jl")
using Distributions

μ = [i for i in μ]

function simulate(μ, Σ, nsim)
    d = MvNormal(μ, Σ)
    return rand(d, nsim)
end
# Simulate cenarios
nsim = 1000
r = simulate(μ, Σ, nsim)

function questao1(μ, Σ, α, γ)
    d = Normal()
    phi_inv = quantile(d, α)

    nvar = size(μ,1)
    m = JuMP.Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, x[1:nvar] >=0)
    @constraint(m, sum(x) == 1)
    @constraint(m, (γ+μ'x)^2 >= phi_inv * (x'*Σ*x))
    @objective(m, Max, x'μ)
    optimize!(m)

    # # portfolio variance
    σ = sqrt(sum(value(x[i])*value(x[j])*Σ[i,j] for i in 1:nvar, j in 1:nvar))
    return x,  σ, termination_status(m)
end

# run code
γ = -0.5 * maximum(μ)
α = 0.95
μ = μ[1:(end-1)]
Σ= Σ[1:(end-1), 1:(end-1)]
x, sigma, stat = questao1(μ, Σ, α, γ)
value.(x)

function questao2(μ, Σ, α, r, R)
    nvar = size(μ,1)
    ncen = size(r,2)
    p = ones(ncen) .* (1/ncen)
    m = JuMP.Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, z)
    @variable(m, y[1:ncen] >=0)
    @variable(m, x[1:nvar] >=0)
    @constraints(m, begin
        CVAR1[s in 1:ncen], y[s] >= z - x' * r[:,s]
        sum(x) == 1
        returns, sum(μ .* x) == R
    end)
    @objective(m, Min, -(z - sum(p .* y)/(1-α)) )
    optimize!(m)
    JuMP.value.(x)
    σ = sqrt(sum(value(x[i]) * value(x[j]) * Σ[i,j] for i in 1:nvar, j in 1:nvar))
    return x, σ, termination_status(m)
end

α = 0.9
R = 1.3
x1, σ, stat = questao2(μ, Σ, 0.9, r, R)
value.(x1)
x2, σ, stat = questao2(μ, Σ, 0.5, r, R)
value.(x2)
x3, σ, stat = questao2(μ, Σ, 0.1, r, R)
value.(x)
value.(x3) - value.(x2)
value.(x3) - value.(x1)
value.(x2) - value.(x1)

function markowitz_variance_modified2(μ, Σ, R)
    nvar = size(μ,1)
    m = JuMP.Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, x[1:nvar] >= 0)
    @constraint(m, sum(x) == 1)
    @constraint(m, sum(μ .* x) == R)
    @objective(m, Min, x' * Σ * x / 2)
    optimize!(m)
    JuMP.value.(x)
    # portfolio variance
    σ = sqrt(sum(value(x[i])*value(x[j]) * Σ[i,j] for i in 1:nvar, j in 1:nvar))
    return σ, termination_status(m)
end
function markowitz_frontier2(average, aux_covar, r, α)
    min_mu = minimum(average)
    max_mu = maximum(average)
    R_vec = collect(min_mu:0.1:max_mu)
    sigma_vec = []
    sigma_vec2 = []
    for R in R_vec
        sigma, status = markowitz_variance_modified2(average, aux_covar, R)
        x, sigma2, status = questao2(average, aux_covar, α, r, R)
        push!(sigma_vec, sigma)
        push!(sigma_vec2, sigma2)
    end
    #@show sigma_vec - sigma_vec2
    plot(sigma_vec, R_vec.-1)
    plot!(sigma_vec2, R_vec.-1)
    xlabel!("sigma")
    ylabel!("r")
    title!("Comparação")
end
markowitz_frontier2(μ, Σ, r, 0.5)

function questao3(μ, Σ, α, r, γ)
    nvar = size(μ,1)
    ncen = size(r,2)
    p = ones(ncen).*(1/ncen)
    m = JuMP.Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, x[1:nvar] >=0)
    @variable(m, θ)
    @variable(m, z)
    @variable(m, δ[1:ncen] >= 0)
    @constraint(m, sum(x) == 1)
    @constraint(m, θ <= γ)
    @constraint(m, θ >= - (z - sum(p[i]*δ[i]/(1-α) for i in 1:ncen)) )
    @constraint(m,[i=1:ncen], δ[i] >= z - r[:,i]'x )
    @objective(m, Max, μ'x)
    optimize!(m)

    # # portfolio variance
    σ = sqrt(sum(value(x[i])*value(x[j])*Σ[i,j] for i in 1:nvar, j in 1:nvar))
    return x, σ, termination_status(m)
end

γ = -0.5 * maximum(μ)
α = 0.95
x1, σ1, stat = questao1(μ, Σ, α, γ)
value.(x1)
x2, σ2, stat = questao2(μ, Σ, α, r, value.(x1)'μ)
value.(x2)
x3, σ3, stat = questao3(μ, Σ, α, r, γ)
value.(x3)

