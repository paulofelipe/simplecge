#' Compute the defined variables
#'
#' @param components a list with the variables, coefficients and sets

compute_variables <- function(components) {
  # TODO: convert and use named arrays

  variables <- components$variables
  coefficients <- components$coefficients
  sets <- components$sets

  res <- with(c(variables, coefficients, sets), {
    n_com <- length(COM)

    imp <- which(SRC %in% "imp")
    p[, imp] <- pfimp[] * phi[]

    SIGMA <- as.vector(SIGMA)

    for (u in seq_along(LOCUSR)) {
      # CES
      ces_pcomp <-
        (rowSums(USE[, , u] / USE_S[, u] * p[]^(1 - SIGMA[])))^(1 / (1 - SIGMA))

      # Cobb-Douglas
      cd_pcomp <- exp(rowSums(USE[, , u] / USE_S[, u] * log(p[])))

      pcomp[, u] <- ifelse(SIGMA == 1, cd_pcomp, ces_pcomp)
    }

    # solve ptot
    for (l in seq_along(LOCFIN)) {
      ll <- which(LOCUSR %in% LOCFIN[l])
      ptot[l] <- prod(pcomp[, ll]^(USE_S[, ll] / COSTS[ll]))
    }

    # 1 = Capital
    # 2 = Labor
    # solve pfac
    hou <- which(LOCFIN %in% "Hou")
    for (f in seq_along(FAC)) {
      pfac[f, ] <- ptot[hou] * ffac[f, ] * ffac_i[f]
    }

    # solve pfac_f
    for (i in seq_along(IND)) {
      pfac_f[i] <-
        sum(
          (FACTOR[, i] / VALADD[i]) * (pfac[, i] * afac[, i])^(1 - 0.5)
        )^(1 / (1 - 0.5))
    }

    # solve xfac
    for (f in seq_along(FAC)) {
      xfac[f, ] <- z[] * afac[f, ] * (pfac[f, ] * afac[f, ] / pfac_f[])^(-0.5)
    }

    # solve wtot
    wtot[] <- ftot[] * wgdpinc[]

    # solve xtot
    for (l in seq_along(LOCFIN)) {
      xtot[l] <- wtot[l] / ptot[l]
    }

    # solve xgdp
    # Laspeyres
    xgdp_l <- sum(FACTOR[] * xfac[] / afac[]) / sum(FACTOR[])
    # Paasche
    xgdp_p <- sum(FACTOR[] * pfac[] * xfac[]) / sum(FACTOR * pfac[] * afac[])
    # Fisher
    xgdp <- sqrt(xgdp_l * xgdp_p)

    # solve xcomp
    for (u in seq_along(LOCUSR)) {
      if (LOCUSR[u] %in% IND) {
        i <- which(LOCUSR[u] %in% IND)
        xcomp[, u] <- z[u]
      } else {
        l <- which(LOCUSR[u] %in% LOCFIN)
        xcomp[, u] <- wtot[l] / pcomp[, u]
      }
    }

    # solve x
    for (c in seq_along(COM)) {
      for (s in seq_along(SRC)) {
        x[c, s, ] <- xcomp[c, ] * (p[c, s] / pcomp[c, ])^(-SIGMA[c])
      }
    }

    # solve xexp
    dom <- which(SRC %in% "dom")
    xexp[] <- fqexp[] * (p[, dom] / phi)^(-5)

    # solve xdem
    expusr <- which(USR == "Exp")
    locusr <- which(USR %in% LOCUSR)
    for (s in seq_along(SRC)) {
      xdem[, s] <- 1 / SALES[, s] * (
        rowSums(USE[, s, locusr] * x[, s, locusr]) +
          USE[, s, expusr] * xexp[]
      )
    }

    # solve wgdexp
    locfin <- which(USR %in% LOCFIN)
    expusr <- which(USR == "Exp")
    dom <- which(SRC == "dom")
    imp <- which(SRC == "imp")

    wgdpexp <- 1 / VGDP[] * (
      # Demanda final local
      sum(sapply(locfin, function(l) USE_S[, l] * wtot[l - n_com])) +
        # exportações
        sum(USE[, dom, expusr] * p[, dom] * xexp[]) -
        # importações
        sum(SALES[, imp] * p[, imp] * xdem[, imp])
    )

    # solve delB
    delB <-
      1 / (VGDP[] * wgdpinc[]) * (
        sum(USE[, dom, expusr] * p[, dom] * xexp[]) -
          sum(SALES[, imp] * p[, imp] * xdem[, imp])
      ) -
      1 / VGDP[] * (sum(USE[, dom, expusr]) - sum(SALES[, imp]))

    list(
      delB = delB, xgdp = xgdp, wgdpinc = wgdpinc, wgdpexp = wgdpexp, phi = phi,
      xexp = xexp, fqexp = fqexp, z = z, pfimp = pfimp, wtot = wtot, ftot = ftot,
      xtot = xtot, ptot = ptot, xfac_i = xfac_i, ffac_i = ffac_i, p = p,
      xdem = xdem, pfac = pfac, pfac_f = pfac_f, xfac = xfac, afac = afac,
      ffac = ffac, pcomp = pcomp, xcomp = xcomp, x = x
    )
  })

  res
}
