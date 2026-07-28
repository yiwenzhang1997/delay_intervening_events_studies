

#This simulates the mediator issue in Costa and Boakes, 2011
#In all simulations, the slopes for A and B on C should be close to 1
#I used rnorm(n, mean=1, sd=1) for all exogenous noise. If a mean of 0 is used, sometimes these problems go away, but that is just an edge case / knife-edge. 

n<-50000

A_trace_interval                     <- rnorm(n, mean=1, sd=1)
B_number_of_distractors_during_trace <- rnorm(n, mean=1, sd=1) + A_trace_interval
C_performance                        <- rnorm(n, mean=1, sd=1) + 1*A_trace_interval + 1*B_number_of_distractors_during_trace
lm(C_performance ~ A_trace_interval + B_number_of_distractors_during_trace)

# Coefficients:
#   (Intercept)                      A_trace_interval  B_number_of_distractors_during_trace  
# 0.00556                               0.98561                               1.00441  


#Now added U_number_of_distractors as a cause of both B and C, and U and A multiplicatively cause B
#The multiplication comes from the fact that in Costa and Boakes (2011), by manipulating the trace delay and the number of distractors
#while also keeping the rate of each individual distractor constant, the number of distractors during the trace delay
#is determined by the number of distractors x time. Technically this is also x rate, but rate is constant so can be ignored.
#estimates for A and B are quite wrong
A_trace_interval                     <- rnorm(n, mean=1, sd=1)
U_number_of_distractors              <- rnorm(n, mean=1, sd=1)
B_number_of_distractors_during_trace <- rnorm(n, mean=1, sd=1) + A_trace_interval * U_number_of_distractors #multiplicative
C_performance                        <- rnorm(n, mean=1, sd=1) + 1*A_trace_interval + 1*B_number_of_distractors_during_trace + 1*U_number_of_distractors
lm(C_performance ~ A_trace_interval + B_number_of_distractors_during_trace)
# (Intercept)                      A_trace_interval  B_number_of_distractors_during_trace  
# 1.012                                 0.671                                 1.329  

#now testing when U has a bigger impact on C
A_trace_interval                     <- rnorm(n, mean=1, sd=1)
U_number_of_distractors              <- rnorm(n, mean=1, sd=1)
B_number_of_distractors_during_trace <- rnorm(n, mean=1, sd=1) + A_trace_interval * U_number_of_distractors #multiplicative
C_performance                        <- rnorm(n, mean=1, sd=1) + 1*A_trace_interval + 1*B_number_of_distractors_during_trace + 4*U_number_of_distractors #see 4
lm(C_performance ~ A_trace_interval + B_number_of_distractors_during_trace)
# (Intercept)                      A_trace_interval  B_number_of_distractors_during_trace  
# 3.7019                               -0.3392                                2.3263  




#The additive case, not just multiplicative is an issue
A_trace_interval                     <- rnorm(n, mean=1, sd=1)
U_number_of_distractors              <- rnorm(n, mean=1, sd=1)
B_number_of_distractors_during_trace <- rnorm(n, mean=1, sd=1) + A_trace_interval + U_number_of_distractors #additive
C_performance                        <- rnorm(n, mean=1, sd=1) + 1*A_trace_interval + 1*B_number_of_distractors_during_trace + 1*U_number_of_distractors
lm(C_performance ~ A_trace_interval + B_number_of_distractors_during_trace)

# A is underestimated, B is overestimated
(Intercept)                      A_trace_interval  B_number_of_distractors_during_trace  
1.0039                                0.4951                                1.4997  


#if U has a bigger impact on C than A and B
A_trace_interval                     <- rnorm(n, mean=1, sd=1)
U_number_of_distractors              <- rnorm(n, mean=1, sd=1)
B_number_of_distractors_during_trace <- rnorm(n, mean=1, sd=1) + A_trace_interval + U_number_of_distractors #additive
C_performance                        <- rnorm(n, mean=1, sd=1) + 1*A_trace_interval + 1*B_number_of_distractors_during_trace + 4*U_number_of_distractors #see 4
lm(C_performance ~ A_trace_interval + B_number_of_distractors_during_trace)
#A is now negative, B is really overestimated
# (Intercept)                      A_trace_interval  B_number_of_distractors_during_trace  
# 0.9282                               -1.0144                                3.0250  



