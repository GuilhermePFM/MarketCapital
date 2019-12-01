#
# 2
#

# i)
function calc_lucro(P, ry)
    L = 0
    for (i,p) in enumerate(P)
        L += p/(1+ry)^i
    end
    return L
end

y=0.15
ry = 1+y
Pa = [100, 100, 1100]
Pb = [50, 50, 1050]
Pc = [0, 0, 1000]
Pd = [1000, 0, 0]

La = calc_lucro(Pa, ry)
Lb = calc_lucro(Pb, ry)
Lc = calc_lucro(Pc, ry)
Ld = calc_lucro(Pd, ry)

# ii)
function duration(P, ry)
    L = calc_lucro(P, ry)
    T = length(P)
    return sum(t* P[t]/(1+ry)^t for t in 1:T) / L
end

Da = duration(Pa, ry)
Db = duration(Pb, ry)
Dc = duration(Pc, ry)
Dd = duration(Pd, ry)
Dl = duration([0,2000,0], ry)

#
function convexity(P, ry, m=1)
    L = calc_lucro(P, ry)
    C = 1/L * 1/(1 + ry/m)^2 * sum( (k*(k+1) / m^2) *(P[k] / (1+ry/m)^k)  for k in 1:length(P))
    return C
end

Ca = convexity(Pa, ry)
Cb = convexity(Pb, ry)
Cc = convexity(Pc, ry)
Cd = convexity(Pd, ry)

# iv) Imunização:
P = [La, Lb, Lc, Ld]
L = calc_lucro([0,2000,0], ry)
Cl = convexity([0,2000,0], ry)
D = [Da, Db, Dc, Dd]
C = [Ca, Cb, Cc, Cd]
function imunizacao(P, D, C, L, Dl,Cl)
    nvar = size(P,1)
    m = JuMP.Model(with_optimizer(Ipopt.Optimizer))
    @variable(m, x[1:nvar] >=0)
    @constraint(m, x'P >= L)
    @constraint(m, x'D == Dl)
    @constraint(m, x'C >= Cl)
    @objective(m, Min, x'P)
    optimize!(m)

    return x
end
x =imunizacao(P, D, C, L, Dl,Cl)
value.(x)