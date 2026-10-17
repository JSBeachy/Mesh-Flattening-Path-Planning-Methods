"""Mesh flattening methods: the four compared by Mahrukh et al., Robotics 2025, 14, 160,
and six more added by this extension.

Each method maps a disk-topology triangle mesh (V, F) to UV coordinates (n, 2).
"""
from . import abf, arap, bff, cetm, dem, floater, lscm, mds, scp, slim

# The paper's four free-boundary methods.
PAPER_METHODS = {
    "LSCM": lscm.flatten,
    "BFF": bff.flatten,
    "CETM": cetm.flatten,
    "ARAP": arap.flatten,
}

# Added by the extension.
EXTENSION_METHODS = {
    "SCP": scp.flatten,          # spectral conformal (Mullen et al. 2008)
    "ABF++": abf.flatten,        # angle based flattening (Sheffer et al. 2005)
    "SLIM": slim.flatten,        # symmetric Dirichlet, locally injective (Rabinovich et al. 2017)
    "MDS": mds.flatten,          # geodesic distance preserving (Zigelman et al. 2002)
    "DEM": dem.flatten,          # area preserving, density-equalizing (Choi & Rycroft 2018)
    "Floater": floater.flatten,  # fixed circular boundary, mean value weights (Floater 2003)
}

METHODS = {**PAPER_METHODS, **EXTENSION_METHODS}

# What each method tries to preserve.
FAMILY = {"LSCM": "conformal", "BFF": "conformal", "CETM": "conformal", "SCP": "conformal",
          "ABF++": "conformal", "ARAP": "isometric", "SLIM": "isometric", "MDS": "distances",
          "DEM": "areas", "Floater": "fixed boundary"}
