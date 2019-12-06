using Distributions, Plots
#
# Questao 1)
#
function simulate(w, nsim)
    d = Normal(w)
    returns = rand( nsim)
    return returns
end

function plota_wealth(func, inv_func, name)
    nsim = 1000
    premios = []
    wealths = []
    certainty_eqv = []
    expect_retv = []
    ret = simulate(1, nsim) *25
    sort!(ret)
    for w in 1:100
        uti = [func(w, ret[i]) for i in 1:length(ret)]
        expect_ret = sum(ret)/nsim
        expect_uti = sum(uti)/nsim
        certainty_eq = inv_func(expect_uti, w)
        premio = expect_ret - certainty_eq
        push!(premios, premio)
        push!(wealths, w)
        push!(certainty_eqv, certainty_eq)
        push!(expect_retv, expect_ret)
    end
    plot(wealths, premios, label=[name])
    xlabel!("Riqueza inicial")
    ylabel!("Premio")
    title!("Comparação")
end

function util_cara(w, r)
    e = MathConstants.e
    return -e^(-0.01 * (w+r))
end
function inv_cara(u, w)
    return log(-u) / (-0.01) - w
end

plota_wealth(util_cara, inv_cara, "CARA")


#
# Questao 2)
#
function util_crra(w, r)
    return 2 * (w+r)^(1/2)
end
function inv_crra(u, w)
    return (u)^2 / 4 - w
end

plota_wealth(util_crra, inv_crra, "CRRA")




#
# Questao 3)
#
npoints = 10
w = 100
ln = log(w)
gamma = [10^(-Float64(i)) for i in collect(1:npoints)]
val = []
for i in 1:npoints
    v =  ( w ^ gamma[i] - 1 ) / gamma[i]
    push!(val, v)
end
# val = sort(val)
# gamma = sort(gamma)
plot(val, label = "lim ln()")
plot!([ln for i in 1:npoints], label = "Ln(w)")
xlabel!("Iterações")
title!("Comparação")

# Questao 4)
using JuMP, Ipopt, Distributions
function max_utility(func, Rf, R, w)
    ncen = length(R)
    m =JuMP.model(with_optimizer(Ipopt.Optimizer))
    @variable(m, 0 <= x <= w) 
    @objective(m, sum(func(r*x+(w-x)*Rf,0) for r in R)/ncen)
    optimize!(m)
    return value.(x), termination_status(m)
end
function utility(func, Rf, R, w, x)
    ncen = length(R)
    u= 0
    for r in R
        if r*x+(w-x)*Rf > 0
        u += func(r*x+(w-x)*Rf,0)
        end
    end
    u = u /ncen
    return u
end
W = 100
Rf = 1.01
nsim = 100
opt_cara = []
opt_cara_x = []
opt_crra = []
opt_crra_x = []
for Rm in collect(1:0.001:10)
    # simulating
    d = Normal(Rm, 1)
    R = rand(d, nsim)
    cara_lst = [] #util_crra
    crra_lst = [] # util_crra
    for x in 0:1:w
        cara = utility(util_cara, Rf, R, w, x)
        push!(cara_lst, cara)
        crra = utility(util_crra, Rf, R, w, x)
        push!(crra_lst, crra)
    end

    val1, i = findmax(cara_lst)
    push!(opt_cara, val1)
    push!(opt_cara_x, i)
    val2, i = findmax(crra_lst)
    push!(opt_crra, val2)
    push!(opt_crra_x, i)
end
Rm = collect(1:0.001:10)
plot(Rm, opt_cara, label = "CARA")
xlabel!("R")
ylabel!("Valor ótimo")
plot(Rm, opt_crra, label = "CRRA")
xlabel!("R")
ylabel!("Valor ótimo")