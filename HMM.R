library('depmixS4')
library('quantmod')
library(magrittr)
set.seed(1)
setwd("D:/Repositories/PUC/Mercado de Capitais/Code")

#
# Obtain S&P500 data from 2004 onwards and
# create the returns stream from this
# getSymbols( "^GSPC", src = "yahoo", from="2010-01-01" )
# getSymbols("AAPL", src = "yahoo", from="2010-01-01")
getSymbols(c("^GSPC", "AAPL", "MSFT", "GOOG", "PRB", "VALE", "FB", "AMZN", "IBM", "MCD", "GE"), src = "yahoo",from="2013-01-01")

# plot(AAPL[, "AAPL.Close"], main = "AAPL")

stocks <- as.xts(data.frame(AAPL = AAPL[, "AAPL.Close"], MSFT = MSFT[, "MSFT.Close"], GOOG = GOOG[, "GOOG.Close"], MCD[, "MCD.Close"], GE[, "GE.Close"], 
                            GSPC = GSPC[, "GSPC.Close"],  VALE = VALE[,"VALE.Close"], FB = FB[,"FB.Close"], AMZN=AMZN[,"AMZN.Close"], IBM=IBM[,"IBM.Close"]))

stock_return = apply(stocks, 1, function(x) {x / stocks[1,]}) %>% 
  t %>% as.xts

write.csv(stock_return,"stock_returns.csv", row.names = FALSE)
# dim(MCD[, "MCD.Close"])
# dim(AAPL[, "AAPL.Close"])
# dim(MSFT[, "MSFT.Close"])
# dim(GOOG[, "GOOG.Close"])
# dim(GSPC[, "GSPC.Close"])
# dim(FB[, "FB.Close"])
# dim(PRB[, "PRB.Close"])
# dim(TSLA[, "TSLA.Close"])
# dim(VALE[, "VALE.Close"])
# dim(AMZN[, "AMZN.Close"])
# dim(IBM[, "IBM.Close"])

# plot(as.zoo(stock_return), screens = 1, lty = 1:3, xlab = "Date", ylab = "Return")
# legend("topleft", c("AAPL", "MSFT", "GOOG", "PRB", "MGLU3"), lty = 1:3, cex = 0.5)

# Fit a Hidden Markov Model with three states 
# to the S&P500 returns stream
stock_return.matrix = coredata(stock_return)
hmm <- depmix(response=list(AAPL.Close~1, MSFT.Close ~1, GOOG.Close~1, MCD.Close~1, GE.Close~1, GSPC.Close~1,  VALE.Close ~1, FB.Close ~1, AMZN.Close~1, IBM.Close~1), 
              family = list(gaussian(),gaussian(),gaussian(),gaussian(),gaussian(),gaussian(),gaussian(),gaussian(),gaussian(),gaussian()), 
              nstates = 3, data=data.frame(stock_return.matrix))
hmmfit <- fit(hmm, verbose = FALSE)

#
post_probs <- posterior(hmmfit)
tail(post_probs)
write.csv(post_probs,"post_probs.csv")
#
transition.matrix = hmmfit@trDens
write.csv(transition.matrix,"transition_matrix.csv")

# SIMULATES
nsim = 1000
alpha= setNames(data.frame(matrix(ncol = 3, nrow = nsim)), c("1", "2", "3"))
beta= setNames(data.frame(matrix(ncol = 3, nrow = nsim)), c("1", "2", "3"))
scenario = setNames(data.frame(matrix(ncol = 3, nrow = nsim)), c("scenario"))
for(i in 1:2){
  mod <- simulate(hmmfit, nsim=1)
  hmmfit.forback = forwardbackward(mod)
  
  alpha[i,] = tail(hmmfit.forback$alpha, n=1)
  beta[i,] = tail(hmmfit.forback$beta, n=1)
  # gamma[i,] = tail(hmmfit.forback$gamma, n=1)
  # print(tail(mod@response,1))
  print(mod@response[[1]][[1]])
}
mvrnorm(n = 1, mu, Sigma, tol = 1e-6, empirical = FALSE, EISPACK = FALSE)
write.csv(alpha,"alpha.csv")
write.csv(beta,"beta.csv")

