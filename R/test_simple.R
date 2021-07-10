source("R/create_components.R", encoding = "UTF-8")
source("R/compute_variables.R", encoding = "UTF-8")
source("R/solve_simple.R", encoding = "UTF-8")

# Load data --------------------------------------------------------------------
load("data/simdata.RData", verbose = TRUE)

# Create the model compenents --------------------------------------------------
components <- create_components(data)

# Experiment -------------------------------------------------------------------
# 10% reduction in afac("Labor", "srv")
# Variables are in exact change. 10% reduction -> 0.9
components$variables$afac[2, 5] <- 0.9
solve_simple(components)
