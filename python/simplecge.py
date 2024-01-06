import rdata
import numpy as np
import pandas as pd
from itertools import product
import jax.numpy as jnp
import jax
from functools import partial
import time

jax.config.update("jax_enable_x64", True)


def create_components(file):
    data = rdata.parser.parse_file(file)
    data = rdata.conversion.convert(data)
    data = data["data"]

    ### Sets ###
    sets = {}
    sets["COM"] = data["COM"]
    sets["IND"] = data["IND"]
    sets["FIN"] = ["Hou", "Gov", "Inv", "Exp"]
    sets["LOCFIN"] = ["Hou", "Gov", "Inv"]
    sets["USR"] = sets["IND"] + sets["FIN"]
    sets["LOCUSR"] = sets["IND"] + sets["LOCFIN"]
    sets["SRC"] = ["dom", "imp"]
    sets["FAC"] = ["Capital", "Labor"]
    sets["SEC"] = sets["COM"]

    ### Sets sizes ###
    n_com = len(sets["COM"])
    n_ind = len(sets["IND"])
    n_fac = len(sets["FAC"])
    n_locusr = len(sets["LOCUSR"])
    n_locfin = len(sets["LOCFIN"])

    sets_sizes = {
        "n_com": n_com,
        "n_ind": n_ind,
        "n_fac": n_fac,
        "n_locusr": n_locusr,
        "n_locfin": n_locfin,
    }

    ### Coeficients ####
    USE = data["USE"].to_numpy() + 1e-8
    FACTOR = data["FACT"].to_numpy()
    SIGMA = data["SIG"].to_numpy()
    # Convert to jax array
    USE = jnp.array(USE)
    FACTOR = jnp.array(FACTOR)
    SIGMA = jnp.array(SIGMA)
    # Create derived coefficients
    USE_S = USE.sum(axis=1)
    VALADD = FACTOR.sum(axis=0)
    COSTS = USE_S.sum(axis=0)
    IND_COSTS = COSTS[:n_ind] + VALADD
    COSTS = COSTS.at[:n_ind].set(IND_COSTS)
    SALES = USE.sum(axis=2)
    DIFF = COSTS[:n_ind] - SALES[:n_ind, 0]
    VGDP = VALADD.sum()

    SHARES = USE / jnp.expand_dims(USE_S, axis=1)
    SHARES = jnp.nan_to_num(SHARES, nan=0.5)

    # 1 if SIGMA = 1 (Cobb-Douglas) otherwise 0

    coefs = {
        "USE": USE,
        "FACTOR": FACTOR,
        "SIGMA": SIGMA,
        "USE_S": USE_S,
        "VALADD": VALADD,
        "COSTS": COSTS,
        "SALES": SALES,
        "DIFF": DIFF,
        "VGDP": VGDP,
        "SHARES": SHARES,
    }

    ### Variables
    z = jnp.ones(n_ind)  # Change in supply
    pdom = jnp.ones(n_ind)  # Change in domestic price
    pfimp = jnp.ones(n_ind)  # Change in foreign price
    phi = jnp.ones(1)  # Change in exchange rate (numeraire)
    ffac = jnp.ones((n_fac, n_ind))  # Factor wage shift by industry
    ffac_i = jnp.ones(n_fac)  # Factor wage shift
    afac = jnp.ones((n_fac, n_ind))  # Factor-using technical change
    p = jnp.ones((n_com, 2))  # Commodity prices
    pcomp = jnp.ones((n_com, n_locusr))  # Price dom/imp composites
    ptot = jnp.ones(n_locfin)  # Price indices for local final users
    pfac = jnp.ones((n_fac, n_ind))  # Factor prices
    pfac_f = jnp.ones(n_ind)  # Factor composite
    xfac = jnp.ones((n_fac, n_ind))  # Factor use by industry
    xfac_i = jnp.ones(n_fac)  # Factor use
    wtot = jnp.ones(n_locfin)  # Nominal expenditure by local final users
    wgdpinc = jnp.array(1.0)  # Nominal GDP from income side
    xtot = jnp.ones(n_locusr)  # Real expenditure by local final users
    xcomp = jnp.ones((n_com, n_locusr))  # Quantity dom/imp composites
    x = jnp.ones((n_com, 2, n_locusr))  # # Commodity demands
    xexp = jnp.ones((n_com))  # Export quantities
    fqexp = jnp.ones((n_com))  # Right shift in export demand
    xgdp = jnp.ones(1)  # Real GDP
    xdem = jnp.ones((n_com, 2))  # Total demand for goods
    wgdpexp = jnp.ones(1)  # Nominal GDP from expenditure side
    delB = jnp.zeros(1)  # (Nominal balance of trade)/{nominal GDP}

    vars = {
        "z": z,
        "pdom": pdom,
        "pfimp": pfimp,
        "phi": phi,
        "ffac": ffac,
        "ffac_i": ffac_i,
        "afac": afac,
        "p": p,
        "pcomp": pcomp,
        "ptot": ptot,
        "pfac": pfac,
        "pfac_f": pfac_f,
        "xfac": xfac,
        "xfac_i": xfac_i,
        "wtot": wtot,
        "wgdpinc": wgdpinc,
        "xtot": xtot,
        "xcomp": xcomp,
        "x": x,
        "xexp": xexp,
        "fqexp": fqexp,
        "xgdp": xgdp,
        "xdem": xdem,
        "wgdpexp": wgdpexp,
        "delB": delB,
    }

    return sets, sets_sizes, coefs, vars


@partial(jax.jit, static_argnames=["sizes"])
def update_vars(sizes, coefs, vars):
    USE = coefs["USE"]
    FACTOR = coefs["FACTOR"]
    SIGMA = coefs["SIGMA"]
    USE_S = coefs["USE_S"]
    VALADD = coefs["VALADD"]
    COSTS = coefs["COSTS"]
    SALES = coefs["SALES"]
    DIFF = coefs["DIFF"]
    VGDP = coefs["VGDP"]
    SHARES = coefs["SHARES"]

    z = vars["z"]
    pdom = vars["pdom"]
    pfimp = vars["pfimp"]
    phi = vars["phi"]
    ffac = vars["ffac"]
    ffac_i = vars["ffac_i"]
    afac = vars["afac"]
    p = vars["p"]
    pcomp = vars["pcomp"]
    ptot = vars["ptot"]
    pfac = vars["pfac"]
    pfac_f = vars["pfac_f"]
    xfac = vars["xfac"]
    xfac_i = vars["xfac_i"]
    wtot = vars["wtot"]
    wgdpinc = vars["wgdpinc"]
    xtot = vars["xtot"]
    xcomp = vars["xcomp"]
    x = vars["x"]
    xexp = vars["xexp"]
    fqexp = vars["fqexp"]
    xgdp = vars["xgdp"]
    xdem = vars["xdem"]
    wgdpexp = vars["wgdpexp"]
    delB = vars["delB"]

    n_com = sizes[0]
    n_ind = sizes[1]
    n_fac = sizes[2]
    n_locusr = sizes[3]
    n_locfin = sizes[4]

    #################################
    ### Compute defined variables ###
    #################################
    # Commodity prices (E_p)
    pimp = pfimp * phi  # Change in imported price
    p = p.at[:, 0].set(pdom)
    p = p.at[:, 1].set(pimp)

    # Price dom/imp composites (pcomp)
    # Obs.: For some commodities, we compute the CES price index and for others we
    # compute the Cobb-Douglas price index
    ces_pcomp = jnp.sum(
        SHARES[:, :, :n_locusr]
        * p[:, :, jnp.newaxis] ** (1 - SIGMA[:, jnp.newaxis, jnp.newaxis]),
        axis=1,
    ) ** (1 / (1 - SIGMA[:, jnp.newaxis]))

    cd_pcomp = jnp.exp(
        (SHARES[:, :, :n_locusr] * jnp.log(p[:, :, jnp.newaxis])).sum(axis=1)
    )

    pcomp = jnp.where(SIGMA[:, jnp.newaxis] == 1, cd_pcomp, ces_pcomp)
    # Price indices for local final users(ptot)
    slf = n_ind  # start local final users index
    elf = n_ind + n_locfin  # end local final users index
    ptot = (pcomp[:, slf:elf] ** (USE_S[:, slf:elf] / COSTS[slf:elf])).prod(axis=0)

    # Factor prices (pfac)
    idx_hou = sets["LOCFIN"].index("Hou")
    pfac = ptot[idx_hou] * ffac * ffac_i[:, jnp.newaxis]

    # Factor composite prices (pfac_f)
    pfac_f = jnp.sum(FACTOR / VALADD * (pfac * afac) ** (1 - 0.5), axis=0) ** (
        1 / (1 - 0.5)
    )

    # Factor use (xfac_f)
    xfac = z[jnp.newaxis, :] * afac * (pfac * afac / pfac_f[jnp.newaxis, :]) ** (-0.5)

    # Nominal expenditure by local final users (wtot)
    wtot = jnp.ones(n_locfin) * wgdpinc

    # Real expenditure by local final users (xtot)
    xtot = wtot / ptot

    # Real GDP (xgdp)
    xgdp_l = (FACTOR * xfac / afac).sum() / FACTOR.sum()
    xgdp_p = (FACTOR * pfac * xfac).sum() / jnp.sum(FACTOR * pfac * afac)
    xgdp = jnp.sqrt(xgdp_l * xgdp_p)

    # Quantity dom/imp composites (xcomp)
    xcomp1 = jnp.ones((n_com, n_ind)) * z
    xcomp2 = wtot / pcomp[:, n_ind:]
    xcomp = jnp.concatenate((xcomp1, xcomp2), axis=1)

    # Commodity demands (x)
    x = xcomp[:, jnp.newaxis, :] * (
        p[:, :, jnp.newaxis] / pcomp[:, jnp.newaxis, :]
    ) ** (-SIGMA[:, jnp.newaxis, jnp.newaxis])

    # Export quantities (xexp)
    xexp = fqexp * (p[:, 0] / phi) ** (-5)

    # Total demand for goods (xdem)
    xexp2 = xexp.reshape(n_com, 1)
    # Exports expanded to include "imports" demand. Imports change = 1
    xexp2 = jnp.concatenate((xexp2, jnp.ones((n_com, 1))), axis=1)
    xexp2 = jnp.expand_dims(xexp2, axis=2)

    SHARES_SALES = USE / SALES[:, :, jnp.newaxis]
    SHARES_SALES = jnp.nan_to_num(SHARES_SALES, nan=1 / (n_locusr + 1))

    xdem = (SHARES_SALES * jnp.concatenate((x, xexp2), axis=2)).sum(axis=2)

    # Nominal GDP from expenditure side (wgdpexp)
    wgdpexp = (
        1
        / VGDP
        * (
            (
                (USE_S[:, slf:elf] * wtot[jnp.newaxis, :]).sum()
                + (USE[:, 0, n_locusr] * p[:, 0] * xexp).sum()
                - (SALES[:, 1] * p[:, 1] * xdem[:, 1]).sum()
            )
        )
    )

    # Nominal balance of trade / nominal GDP (delB)
    delB = 1 / (VGDP * wgdpinc) * (
        (USE[:, 0, n_locusr] * p[:, 0] * xexp).sum()
        - (SALES[:, 1] * p[:, 1] * xdem[:, 1]).sum()
    ) - 1 / VGDP * ((USE[:, 0, n_locusr]).sum() - (SALES[:, 1]).sum())

    ###################
    ### Update vars ###
    ###################
    for key in vars:
        vars[key] = locals()[key]

    ##########################################
    ### Compute market clearing conditions ###
    ##########################################

    # Labor market clearing
    res0 = xfac_i[1] - (FACTOR[1, :] * xfac[1, :]).sum() / sum(FACTOR[1, :]).sum()
    res0 = res0[jnp.newaxis]

    # Capital market clearing for each industry
    res1 = 1 - xfac[0, :]

    # Commodities market clearing
    res2 = z - xdem[:, 0]

    # Zero-profit condition
    res3 = p[:, 0] - 1 / COSTS[:n_ind] * (
        (USE_S[:, :n_ind] * pcomp[:, :n_ind]).sum(axis=0) + VALADD * pfac_f
    )

    # wgdp condition
    res4 = wgdpinc - 1 / VGDP * (FACTOR * pfac * xfac).sum()
    res4 = res4[jnp.newaxis]

    vars["res"] = jnp.concatenate((res0, res1, res2, res3, res4), axis=0)

    return vars


def simple(x0, sizes, coefs, vars):
    n_ind = sizes[0]
    vars = vars.copy()

    vars["ffac_i"] = vars["ffac_i"].at[1].set(x0[0])
    vars["ffac"] = vars["ffac"].at[0, :].set(x0[1 : (n_ind + 1)])
    vars["pdom"] = vars["pdom"].at[:].set(x0[(n_ind + 1) : (2 * n_ind + 1)])
    vars["z"] = vars["z"].at[:].set(x0[(2 * n_ind + 1) : (3 * n_ind + 1)])
    vars["wgdpinc"] = x0[3 * n_ind + 1]

    vars = update_vars(sizes, coefs, vars)

    return vars["res"]


def solve(x0, sizes, coefs, vars, sparse=True):
    sizes_tup = tuple(sizes.values())

    if sparse:
        fn = lambda x: simple(x, sizes_tup, coefs, vars)
        for i in range(20):
            f = fn(x0)
            if jnp.linalg.norm(f) < 1e-8:
                break

            dp = jax.scipy.sparse.linalg.bicgstab(
                lambda x: jax.jvp(fn, (x0,), (x,))[1], -f
            )[0]
            x0 = x0 + dp

    if not sparse:
        jac_simple = jax.jit(jax.jacfwd(simple, argnums=0), static_argnums=1)
        for i in range(20):
            jac = jac_simple(x0, sizes_tup, coefs, vars)
            f = simple(x0, sizes_tup, coefs, vars)
            if jnp.linalg.norm(f) < 1e-8:
                break
            dp = jnp.linalg.solve(jac, -f)
            x0 = x0 + dp

    vars["ffac_i"] = vars["ffac_i"].at[1].set(x0[0])
    vars["ffac"] = vars["ffac"].at[0, :].set(x0[1 : (n_ind + 1)])
    vars["pdom"] = vars["pdom"].at[:].set(x0[(n_ind + 1) : (2 * n_ind + 1)])
    vars["z"] = vars["z"].at[:].set(x0[(2 * n_ind + 1) : (3 * n_ind + 1)])
    vars["wgdpinc"] = x0[3 * n_ind + 1]

    vars = update_vars(sizes_tup, coefs, vars)

    del vars["res"]

    return vars


def format_vars(vars, sets):
    vars_dim = {
        "z": ["IND"],
        "pdom": ["IND"],
        "pfimp": ["IND"],
        "phi": [],
        "ffac": ["FAC", "IND"],
        "ffac_i": ["FAC"],
        "afac": ["FAC", "IND"],
        "p": ["COM", "SRC"],
        "pcomp": ["COM", "LOCUSR"],
        "ptot": ["LOCFIN"],
        "pfac": ["FAC", "IND"],
        "pfac_f": ["IND"],
        "xfac": ["FAC", "IND"],
        "xfac_i": ["FAC"],
        "wtot": ["LOCFIN"],
        "wgdpinc": [],
        "xtot": ["LOCFIN"],
        "xcomp": ["COM", "LOCUSR"],
        "x": ["COM", "SRC", "LOCUSR"],
        "xexp": ["COM"],
        "fqexp": ["COM"],
        "xgdp": [],
        "xdem": ["COM", "SRC"],
        "wgdpexp": [],
        "delB": [],
    }

    dfs = {}
    for var in vars_dim:
        print(var)
        if len(vars_dim[var]) > 0:
            tmp = {}
            for dim in vars_dim[var]:
                tmp[dim] = sets[dim]

            df = pd.DataFrame(
                [row for row in product(*tmp.values())], columns=tmp.keys()
            )
            df["value"] = vars[var].flatten()
            dfs[var] = df

        if len(vars_dim[var]) == 0:
            df = pd.DataFrame({"value": vars[var].reshape(1)})
            dfs[var] = df

    return dfs


sets, sizes, coefs, vars = create_components("../data/simdata.RData")
# sets, sizes, coefs, vars = create_components("../data/hugeD500.RData")
sizes_tup = tuple(sizes.values())
n_ind = sizes_tup[0]
x0 = jnp.ones(3 * n_ind + 2)
# vars["phi"] = vars["phi"].at[:].set(1.1)

# vars["afac"] = vars["afac"].at[1, :].set(0.98)
vars["afac"] = vars["afac"].at[1, 4].set(0.9)

vars = solve(x0, sizes, coefs, vars, sparse=True)

results = format_vars(vars, sets)

results["z"]
print(np.round((results["xgdp"] - 1) * 100, 3))
np.round((results["xfac"]["value"] - 1) * 100, 3)
