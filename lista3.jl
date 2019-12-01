function Util(x, wealth)
    e = MathConstants.e
    return -e^(-0.01 * (x + wealth))
end
function Inv_Util(u, wealth)
    return log(-u) * (-100) - wealth
end
returns = rand(num_samples) * 25 .+ 90;
# defining variables
num_samples = 1000;
wealth = 11
# utility function results 
result = Util.(sort(returns));
# right equivalent
aver_results = sum(result) / num_samples;
right_eq = Inv_Util(aver_results, wealth);
# risk premium
premium = aver_results - right_eq

#
# 2
#
function Util(x, wealth)
    return 2 * (x + wealth)^(1/2)
end
function Inv_Util(u, wealth)
    return 4 / (u + wealth)^2
end
returns = rand(num_samples) * 25 .+ 90;
# defining variables
num_samples = 1000;
wealth = 12
# utility function results 
result = Util.(sort(returns));
# right equivalent
aver_results = sum(result) / num_samples;
right_eq = Inv_Util(aver_results, wealth);
# risk premium
premium = aver_results - right_eq

#
# 3
#
using Distributions
function Util1(x)
    e = MathConstants.e
    return -e^(-0.01 * (x))
end
function Util2(x)
    return 2 * (x)^(1/2)
end
Aver_R = collect(-1:0.001:1)
final_results1 = zeros(size(Aver_R)[1],2)
final_results2 = zeros(size(Aver_R)[1],2)
for j in 1:size(Aver_R)[1]
    # risky asset
    y = Normal(Aver_R[j],1)
    num_samples = 10000
    y_sample = rand(y,num_samples)
    r = MathConstants.e.^(y_sample)
    sum(r)/num_samples
    # risk free asset
    Aver_Rfree = 1.01
    # other assumptions
    wealth = 100
    num_disc = 1000
    x = collect(0:(wealth/num_disc):wealth)
    result1 = zeros(num_disc+1)
    result2 = zeros(num_disc+1)
    for i in 1:(num_disc+1)
        result1[i] = sum( Util1.( x[i] .* r[:] .+ (wealth - x[i]) * Aver_Rfree ) ) / num_samples
        result2[i] = sum( Util2.( x[i] .* r[:] .+ (wealth - x[i]) * Aver_Rfree ) ) / num_samples
    end
    final_results1[j,1] = findmax(result1)[1]
    final_results1[j,2] = findmax(result1)[2]
    final_results2[j,1] = findmax(result2)[1]
    final_results2[j,2] = findmax(result2)[2]
end