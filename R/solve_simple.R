solve_simple <- function(components) {
  variables <- components$variables
  coefficients <- components$coefficients
  sets <- components$sets

  # System function
  fn <- function(start, variables, coefficients, sets) {
    n_com <- length(sets$COM)
    variables$ffac_i[2] <- start[1]
    variables$ffac[1, ] <- start[2:(1 + n_com)]
    start <- start[-(1:(n_com + 1))]
    variables$p[, 1] <- start[1:n_com]
    variables$z[] <- start[(n_com + 1):(2 * n_com)]
    variables$wgdpinc[] <- start[length(start)]

    components$variables <- variables
    variables <- compute_variables(components)

    res <- with(c(variables, coefficients, sets), {
      dom <- which(SRC == "dom")
      sec <- which(USR %in% SEC)
      res <- c(
        xfac_i[2] - sum(FACTOR[2, ] * xfac[2, ]) / sum(FACTOR[2, ]),
        1 - xfac[1, ],
        z[] - xdem[, dom],
        p[sec, dom] - 1 / COSTS[sec] * (
          colSums(USE_S[, sec] * pcomp[, sec]) + VALADD[sec] * pfac_f[sec]
        ),
        wgdpinc - 1 / VGDP[] * sum(FACTOR[] * pfac[] * xfac[])
      )
      res
    })

    # list(res = res, variables = variables)
    res
  }

  # Start values
  x0 <- c(
    variables$ffac_i[2],
    variables$ffac[1, ],
    variables$p[, 1],
    variables$z[],
    variables$wgdpinc[]
  )

  # Nonlinear system solution
  sol <- nleqslv::nleqslv(
    x = x0,
    fn = fn,
    variables = variables,
    coefficients = coefficients,
    sets = sets,
    control = list(trace = 1)
  )

  x0 <- sol$x

  # Update variables
  n_com <- length(sets$COM)
  variables$ffac_i[2] <- x0[1]
  variables$ffac[1, ] <- x0[2:(1 + n_com)]
  x0 <- x0[-(1:(n_com + 1))]
  variables$p[, 1] <- x0[1:n_com]
  variables$z[] <- x0[(n_com + 1):(2 * n_com)]
  variables$wgdpinc[] <- x0[length(x0)]

  components$variables <- variables
  variables <- compute_variables(components)
  variables
}
