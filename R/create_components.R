#' Create the components of the simple model
#' 
#' Given an input data, it creates the sets, coefficients and variables to
#' run the simulations.
#' 
#' @param data a list of named arrays. The data object should have this list of
#' arrays: 
#' c("COM", "IND", "SEC", "S2", "USR", "SRC", "LUSR", "LFIN", "USE", "FACT", "SIG") 

create_components <- function(data) {
  # Sets
  sets <- data[c("COM", "IND")]
  sets[["FIN"]] <- c("Hou", "Gov", "Inv", "Exp")
  sets[["LOCFIN"]] <- c("Hou", "Gov", "Inv")
  sets[["USR"]] <- union(sets[["IND"]], sets[["FIN"]])
  sets[["LOCUSR"]] <- union(sets[["IND"]], sets[["LOCFIN"]])
  sets[["SRC"]] <- c("dom", "imp")
  sets[["FAC"]] <- c("Capital", "Labor")
  sets[["SEC"]] <- intersect(sets[["COM"]], sets[["IND"]])

  COM <- sets$COM
  IND <- sets$IND
  SRC <- sets$SRC
  FAC <- sets$FAC
  USR <- sets$USR
  LOCUSR <- sets$LOCUSR
  LOCFIN <- sets$LOCFIN

  # Coefficientes

  USE <- data$USE + 1e-8
  FACTOR <- data$FACT
  SIGMA <- data$SIG # ifelse(data$SIG == 1, 1 + 1e-2, data$SIG)
  USE_S <- apply(USE, c(1, 3), sum)
  VALADD <- apply(FACTOR, 2, sum)
  COSTS <- apply(USE_S, 2, sum)
  COSTS[sets$IND] <- COSTS[sets$IND] + VALADD
  SALES <- apply(USE, c(1, 2), sum)
  DIFF <- COSTS[sets$IND] - SALES[sets$IND, "dom"]
  VGDP <- sum(VALADD)

  coefficients <- list(
    USE = USE,
    FACTOR = FACTOR,
    SIGMA = SIGMA,
    USE_S = USE_S,
    VALADD = VALADD,
    COSTS = COSTS,
    SALES = SALES,
    VGDP = VGDP
  )

  # Variables
  n_com <- length(sets$COM)
  n_ind <- length(sets$IND)
  n_fac <- length(sets$FAC)
  n_locfin <- length(sets$LOCFIN)
  n_locusr <- length(sets$LOCUSR)
  n_src <- length(sets$SRC)

  variables <- list(
    delB = 0,
    xgdp = 1,
    wgdpinc = 1,
    phi = 1,
    xexp = rep(1, n_com),
    fqexp = rep(1, n_com),
    z = rep(1, n_ind),
    pfimp = rep(1, n_com),
    wtot = rep(1, n_locfin),
    ftot = rep(1, n_locfin),
    xtot = rep(1, n_locfin),
    ptot = rep(1, n_locfin),
    xfac_i = rep(1, n_fac),
    ffac_i = rep(1, n_fac),
    pfac_i = rep(1, n_fac),
    p = array(1, dim = c(n_com, n_src)),
    xdem = array(1, dim = c(n_com, n_src)),
    pfac = array(1, dim = c(n_fac, n_ind)),
    pfac_f = rep(1, n_ind),
    xfac = array(1, dim = c(n_fac, n_ind)),
    afac = array(1, dim = c(n_fac, n_ind)),
    ffac = array(1, dim = c(n_fac, n_ind)),
    pcomp = array(1, dim = c(n_com, n_locusr)),
    xcomp = array(1, dim = c(n_com, n_locusr)),
    x = array(1, dim = c(n_com, n_src, n_locusr))
  )

  list(
    sets = sets,
    coefficients = coefficients,
    variables = variables
  )
}