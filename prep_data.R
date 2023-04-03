library(arrow)
library(tidyverse)
library(glue)

inv_logit <- function(x) 1 / (1 + exp(-x))

n_obs <- rpois(30, lambda = 1000)


for (i in seq_len(30) - 1) {
    
    n <- n_obs[i + 1]
    d <- tibble(
        x1 = rnorm(n),
        x2 = rnorm(n),
        x3 = rnorm(n),
        x4 = rnorm(n),
        x5 = rnorm(n)
    ) |> 
        mutate(
            ml = -2 + x1 - x2 + x3 + x4 - x5,
            ml = rbinom(n, size = 1, p = inv_logit(ml))
        )
    
    train_idx <- sample(nrow(d), size = n / 2)
    
    d |> 
        slice(train_idx) |> 
        write_parquet(
            glue("data_{i}.parquet")
        )

    
    d |> 
        slice(-train_idx) |> 
        write_parquet(
            glue("test_data_{i}.parquet")
        )
    
}
