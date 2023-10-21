import rdata
import numpy as np
import jax.numpy as jnp

data = rdata.parser.parse_file("../data/simdata.RData")
# data = rdata.parser.parse_file("../data/hugeD500.RData")

data = rdata.conversion.convert(data)
data = data["data"]

# Sets
sets = {}
sets["COM"] = data["COM"]
sets["IND"] = data["IND"]
sets["FIN"] = ["Hou", "Gov", "Inv", "Exp"]
sets["LOCFIN"] = ["Hou", "Gov", "Inv"]
sets["USR"] = sets["IND"] + sets["FIN"]
sets["LOCUSR"] = sets["IND"] + sets["LOCFIN"]
sets["SRC"] = ["dom", "imp"]
sets["FAC"] = ["Capital", "Labor"]
sets["SEC"] = list(set(set(sets["COM"]) & set(sets["IND"])))

# Which sets["FIN"] is equal to "Gov"
sets["FIN"].index("Gov")

dom_idx = sets["SRC"].index("dom")
imp_idx = sets["SRC"].index("imp")
ind_in_usr = [i for i, x in enumerate(sets["USR"]) if x in sets["IND"]]
fin_in_locusr = [i for i, x in enumerate(sets["LOCUSR"]) if x in sets["FIN"]]

n_com = len(sets["COM"])
n_ind = len(sets["IND"])
n_fac = len(sets["FAC"])
n_locusr = len(sets["LOCUSR"])
n_locfin = len(sets["LOCFIN"])

# Coeficients ------------------------------------------------------------------
# Load coefficients from base data
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

# 1 if SIGMA = 1 (Cobb-Douglas) otherwise 0
CD = np.array(SIGMA == 1, dtype=np.int8)

# Variables --------------------------------------------------------------------
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
xfac = jnp.ones((n_fac, n_ind))  # Factor use
wtot = jnp.ones(n_locfin)  # Nominal expenditure by local final users
wgdpinc = jnp.array(1.0)  # Nominal GDP from income side
xtot = jnp.ones(n_locusr)  # Real expenditure by local final users
xcomp = jnp.ones((n_com, n_locusr))  # Quantity dom/imp composites
x = jnp.ones((n_com, 2, n_locusr))  # # Commodity demands
xexp = jnp.ones((n_com))  # Export quantities

# Equations --------------------------------------------------------------------
# Commodity prices (E_p)
pimp = pfimp * phi  # Change in imported price
p = p.at[:, 0].set(pdom)
p = p.at[:, 1].set(pimp)


# Price dom/imp composites (pcomp)
# Obs.: For some commodities, we compute the CES price index and for others we
# compute the Cobb-Douglas price index
ces_pcomp = jnp.sum(
    SHARES[:, :, :n_locusr]
    * p.reshape(n_com, 2, 1) ** (1 - SIGMA.reshape(n_com, 1, 1)),
    axis=1,
) ** (1 / (1 - SIGMA.reshape(n_com, 1)))

cd_pcomp = jnp.exp(
    (SHARES[:, :, :n_locusr] * jnp.log(p.reshape(n_com, 2, 1))).sum(axis=1)
)

pcomp = ces_pcomp * (1 - CD.reshape(n_com, 1)) + cd_pcomp * CD.reshape(n_com, 1)

# Price indices for local final users(ptot)
slf = n_ind  # start local final users index
elf = n_ind + n_locfin  # end local final users index
ptot = (pcomp[:, slf:elf] ** (USE_S[:, slf:elf] / COSTS[slf:elf])).prod(axis=0)

# Factor prices (pfac)
idx_hou = sets["LOCFIN"].index("Hou")
pfac = ptot[idx_hou] * ffac * ffac_i.reshape(n_fac, 1)

# Factor composite prices (pfac_f)
pfac_f = jnp.sum(FACTOR / VALADD * (pfac * afac) ** (1 - 0.5), axis=0) ** (
    1 / (1 - 0.5)
)

# Factor use (xfac_f)
xfac = z.reshape(1, n_ind) * afac * (pfac * afac / pfac_f.reshape(1, n_ind)) ** (-0.5)

# Nominal expenditure by local final users (wtot)
wtot = jnp.ones(n_locfin) * wgdpinc

# Real expenditure by local final users (xtot)
xtot = wtot / ptot

# Quantity dom/imp composites (xcomp)
xcomp1 = jnp.ones((n_com, n_ind)) * z
xcomp2 = wtot / pcomp[:, n_ind:]
xcomp = jnp.concatenate((xcomp1, xcomp2), axis=1)

# Commodity demands (x)
x = xcomp.reshape(n_com, 1, n_locusr) * (
    p.reshape((n_com, 2, 1)) / pcomp.reshape(n_com, 1, n_locusr)
) ** (-SIGMA.reshape(n_com, 1, 1))
