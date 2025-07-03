# Load necessary packages
library(broom)
library(tidyverse)
library(brms)

e<-rnorm(24, 0, 0.1)
d_83 <- read.csv("three_pos_10_2_0_12.csv")
d_67 <- read.csv("three_pos_8_4_0_12.csv")
d_50 <- read.csv("three_pos_9_3_3_9.csv")

# 0.83
model_brms_a_83 <- brm(e_a ~ c_a + c_b + c_c, 
                       data = d_83, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_a_83)

model_brms_b_83 <- brm(e_b ~ c_a + c_b + c_c, 
                       data = d_83, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_b_83)


model_brms_c_83 <- brm(e_c ~ c_a + c_b + c_c, 
                       data = d_83, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_c_83)

# 0.67

model_brms_a_67 <- brm(e_a ~ c_a + c_b + c_c, 
                       data = d_67, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_a_67)

model_brms_b_67 <- brm(e_b ~ c_a + c_b + c_c, 
                       data = d_67, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_b_67)


model_brms_c_67 <- brm(e_c ~ c_a + c_b + c_c, 
                       data = d_67, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_c_67)

# 0.50
model_brms_a_50 <- brm(e_a ~ c_a + c_b + c_c, 
                       data = d_50, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_a_50)

model_brms_b_50 <- brm(e_b ~ c_a + c_b + c_c, 
                       data = d_50, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_b_50)


model_brms_c_50 <- brm(e_c ~ c_a + c_b + c_c, 
                       data = d_50, 
                       family = bernoulli(link = "logit"),
                       chains = 4, 
                       iter = 2000, 
                       seed = 123)
summary(model_brms_c_50)


get_model_summary <- function(model, model_name) {
  summary_df <- as.data.frame(posterior_summary(model))
  summary_df <- summary_df %>%
    mutate(Model = model_name,
           Parameter = rownames(summary_df)) %>%
    select(Model, Parameter, Estimate, Q2.5, Q97.5)
  return(summary_df)
}

# Extract summaries for all models
summary_a_83 <- get_model_summary(model_brms_a_83, "Model A (0.83)")
summary_b_83 <- get_model_summary(model_brms_b_83, "Model B (0.83)")
summary_c_83 <- get_model_summary(model_brms_c_83, "Model C (0.83)")

summary_a_67 <- get_model_summary(model_brms_a_67, "Model A (0.67)")
summary_b_67 <- get_model_summary(model_brms_b_67, "Model B (0.67)")
summary_c_67 <- get_model_summary(model_brms_c_67, "Model C (0.67)")

summary_a_50 <- get_model_summary(model_brms_a_50, "Model A (0.50)")
summary_b_50 <- get_model_summary(model_brms_b_50, "Model B (0.50)")
summary_c_50 <- get_model_summary(model_brms_c_50, "Model C (0.50)")

# Combine all summaries into one data frame
all_summaries <- bind_rows(
  summary_a_83, summary_b_83, summary_c_83,
  summary_a_67, summary_b_67, summary_c_67,
  summary_a_50, summary_b_50, summary_c_50
)

# Display the combined summary table
rownames(all_summaries) <-c()
all_summaries %>%
  filter(Parameter %in% c("b_c_a", "b_c_b", "b_c_c")) %>%
  mutate(mean_95 = paste0(round(Estimate,3), " [", round(Q2.5,3), ", ", round(Q97.5,3), "]"))%>%
  select(Model, Parameter, mean_95) %>%
  pivot_wider(names_from = Parameter, values_from = mean_95)  %>% 
  write.csv("exp1a_three_datasets_brms_results.csv")
