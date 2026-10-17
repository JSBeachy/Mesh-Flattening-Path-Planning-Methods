"""The four free-boundary flattening methods compared by Mahrukh et al., Robotics 2025, 14, 160.

Each method maps a disk-topology triangle mesh (V, F) to UV coordinates (n, 2).
"""
from . import arap, bff, cetm, lscm

METHODS = {
    "LSCM": lscm.flatten,
    "BFF": bff.flatten,
    "CETM": cetm.flatten,
    "ARAP": arap.flatten,
}
