"""3D TDGL (Model A) for ferroelectric capacitors with layered electrostatics.

`tdgl3d.solver` holds the physics: material and grid descriptions, defect
sites, and `StackModel` (ETD2 time stepping of scalar Pz with self-consistent
electrostatics through dielectric/ferroelectric/dielectric stacks).
`tdgl3d.animations` turns saved runs into cross-section viewers and MP4s.
"""
from .solver import Material, Grid, NucleationSite, StackModel, triangular_voltage

__all__ = ['Material', 'Grid', 'NucleationSite', 'StackModel', 'triangular_voltage']
__version__ = '0.1.0'
