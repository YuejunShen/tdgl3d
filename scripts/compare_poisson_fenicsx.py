"""Check the tdgl3d electrostatics against an independent FEniCSx Poisson solve.

For a prescribed polarization P(x,y,z) in BTO and terminal voltage V, both codes
solve  div(eps grad phi) = d(Theta_f P)/dz  in the dielectric/BTO/dielectric
stack with phi = V at the bottom electrode and phi = 0 at the top.

tdgl3d: lateral Fourier modes + finite-volume z grid (StackModel.electrostatics).
FEniCSx: 3D Q2 finite elements on the weak form
    int eps grad(phi).grad(v) dx = int p dv/dz dx,
in which potential and normal-D continuity at interfaces are natural. The test
polarization uses only cos(2 pi m x/Lx) cos(2 pi n y/Ly) modes, so the
periodic solution is mirror-symmetric about x = 0, Lx/2 and y = 0, Ly/2; the FE
domain is that quarter cell with zero-flux side walls.

Run in a separate environment that provides dolfinx, e.g.
    conda run -n fenicsx python scripts/compare_poisson_fenicsx.py
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tdgl3d import Material, Grid, StackModel  # noqa: E402

NM = 1e-9
LX = LY = 20e-9
NXY = 16
FV_LEVELS = [(16, 1), (32, 2), (64, 4), (128, 8)]  # (BTO cells, cells per dielectric)
FE_LEVELS = {'coarse': (1.0, 0.5), 'fine': (0.5, 0.25)}  # (lateral, z) element size in nm
CASES = {'P only, V=0': (1.0, 0.0), 'V=1 V only, P=0': (0.0, 1.0), 'P and V=1 V': (1.0, 1.0)}


def polarization(lib, x, y, z, m):
    """Mean + two lateral cosine modes, each with its own z profile (C/m^2)."""
    kx, ky, s = 2*np.pi/LX, 2*np.pi/LY, z/m.h_f
    return float(m.Ps)*(0.5 + 0.2*s
                        + 0.3*lib.cos(kx*x)*(1 + 0.5*lib.cos(np.pi*s))
                        + 0.2*lib.cos(kx*x)*lib.cos(2*ky*y)*lib.sin(np.pi*s + 0.3))


def fenicsx_solve(m, P_scale, voltage, h_lat, h_z, degree=2):
    """Return callables phi(points), E_z(points); points in metres, shape (n, 3)."""
    from mpi4py import MPI
    from petsc4py import PETSc
    import ufl
    from dolfinx import fem, geometry, mesh
    from dolfinx.fem.petsc import LinearProblem

    hd, hf = m.h_d/NM, m.h_f/NM
    if not (np.isclose(hd/h_z, round(hd/h_z)) and np.isclose(hf/h_z, round(hf/h_z))):
        raise ValueError("z element size must put nodes on both BTO interfaces.")
    z_lo, z_hi = -hd, hf+hd
    n_cells = [round(LX/2/NM/h_lat), round(LY/2/NM/h_lat), round((z_hi-z_lo)/h_z)]
    msh = mesh.create_box(MPI.COMM_SELF,
                          [np.array([0.0, 0.0, z_lo]), np.array([LX/2/NM, LY/2/NM, z_hi])],
                          n_cells, cell_type=mesh.CellType.hexahedron)
    space = fem.functionspace(msh, ("Lagrange", degree))

    # Lengths in nm: div_nm(eps_r grad_nm phi) = d_nm(p NM/eps0), with p NM/eps0 in volts.
    x = ufl.SpatialCoordinate(msh)
    in_fe = ufl.And(ufl.gt(x[2], 0.0), ufl.lt(x[2], hf))
    eps_r = ufl.conditional(in_fe, m.eps_f, m.eps_d)
    # A Constant keeps UFL from simplifying the P = 0 case to an empty form.
    scale = fem.Constant(msh, PETSc.ScalarType(P_scale))
    p = ufl.conditional(in_fe, scale*polarization(ufl, x[0]*NM, x[1]*NM, x[2]*NM, m), 0.0)
    p_volts = p*NM/m.eps0

    u, v = ufl.TrialFunction(space), ufl.TestFunction(space)
    a = eps_r*ufl.inner(ufl.grad(u), ufl.grad(v))*ufl.dx
    L = p_volts*v.dx(2)*ufl.dx(metadata={'quadrature_degree': 8})
    bottom = fem.locate_dofs_geometrical(space, lambda X: np.isclose(X[2], z_lo))
    top = fem.locate_dofs_geometrical(space, lambda X: np.isclose(X[2], z_hi))
    bcs = [fem.dirichletbc(PETSc.ScalarType(voltage), bottom, space),
           fem.dirichletbc(PETSc.ScalarType(0.0), top, space)]
    options = {'ksp_type': 'cg', 'pc_type': 'hypre', 'pc_hypre_type': 'boomeramg',
               'ksp_rtol': 1e-13, 'ksp_atol': 1e-16, 'ksp_max_it': 2000}
    try:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=options,
                                petsc_options_prefix=f"poisson_{id(msh)}_")
    except TypeError:
        problem = LinearProblem(a, L, bcs=bcs, petsc_options=options)
    phi = problem.solve()
    phi = phi[0] if isinstance(phi, tuple) else phi
    if problem.solver.getConvergedReason() <= 0:
        raise RuntimeError("FEniCSx linear solve did not converge.")

    # -d phi/dz of a Q2 hexahedral field lies exactly in discontinuous Q2.
    dg = fem.functionspace(msh, ("DG", degree))
    E_z = fem.Function(dg)
    points_ref = dg.element.interpolation_points
    points_ref = points_ref() if callable(points_ref) else points_ref
    E_z.interpolate(fem.Expression(-phi.dx(2)/NM, points_ref))

    tree = geometry.bb_tree(msh, msh.topology.dim)

    def evaluator(function):
        def evaluate(points):
            points = np.ascontiguousarray(points/NM, dtype=np.float64)
            colliding = geometry.compute_colliding_cells(
                msh, geometry.compute_collisions_points(tree, points), points)
            cells = np.array([colliding.links(i)[0] for i in range(len(points))], dtype=np.int32)
            return function.eval(points, cells)[:, 0]
        return evaluate

    return evaluator(phi), evaluator(E_z), space.dofmap.index_map.size_global


def quadrant_points(model, z):
    """Model cell centers with x < Lx/2, y < Ly/2, at heights z; and the index mask."""
    ix, iy = model.x < LX/2, model.y < LY/2
    X, Y, Z = np.meshgrid(model.x[ix], model.y[iy], z, indexing='ij')
    return np.c_[X.ravel(), Y.ravel(), Z.ravel()], ix, iy


def exact_mean_field(m, P_scale, voltage, z):
    """q = 0 mode: D_z is uniform; integrate (D - p)/eps across the stack."""
    mean_P = lambda s: P_scale*float(m.Ps)*(0.5 + 0.2*s)
    integral_p = P_scale*float(m.Ps)*(0.5 + 0.1)*m.h_f/(m.eps0*m.eps_f)
    resistance = 2*m.h_d/(m.eps0*m.eps_d) + m.h_f/(m.eps0*m.eps_f)
    D = (voltage + integral_p)/resistance
    return (D - mean_P(z/m.h_f))/(m.eps0*m.eps_f)


def relative_error(a, b):
    return float(np.max(np.abs(a-b))/max(np.max(np.abs(b)), 1e-300))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out', type=Path, default=ROOT/'docs'/'verification'/'verification_poisson_fenicsx.json')
    args = parser.parse_args()
    m = Material()
    results = {'description': __doc__.split('\n\n')[0], 'Lx_m': LX, 'Ly_m': LY,
               'nx': NXY, 'ny': NXY, 'fe_degree': 2, 'cases': {}}

    for case, (P_scale, voltage) in CASES.items():
        print(f"\n=== {case} ===")
        fe = {}
        for level, (h_lat, h_z) in FE_LEVELS.items():
            start = time.perf_counter()
            fe[level] = fenicsx_solve(m, P_scale, voltage, h_lat, h_z)
            print(f"FEniCSx {level:6s}: h_lat={h_lat} nm, h_z={h_z} nm, "
                  f"{fe[level][2]} dofs, {time.perf_counter()-start:.1f} s")

        rows = []
        for nz, nd in FV_LEVELS:
            model = StackModel(m, Grid(nx=NXY, ny=NXY, nz=nz, nd=nd, Lx=LX, Ly=LY))
            X, Y, Z = np.meshgrid(model.x, model.y, model.z, indexing='ij')
            P = P_scale*polarization(np, X, Y, Z, m)
            out = model.electrostatics(P, voltage)

            pts_phi, ix, iy = quadrant_points(model, model.z_stack)
            pts_E, _, _ = quadrant_points(model, model.z)
            phi_fv = out['phi'][np.ix_(ix, iy)].ravel()
            E_fv = out['E_z'][np.ix_(ix, iy)].ravel()
            row = {'nz_BTO': nz, 'nd_dielectric': nd, 'dz_BTO_nm': m.h_f/nz/NM,
                   'gauss_relative_residual': out['gauss_relative_residual']}
            for level in FE_LEVELS:
                phi_fe, E_fe, _ = fe[level]
                row[f'phi_err_vs_{level}'] = relative_error(phi_fv, phi_fe(pts_phi))
                row[f'Ez_err_vs_{level}'] = relative_error(E_fv, E_fe(pts_E))
            phi_fe, E_fe, _ = fe['fine']
            row['Ez_max_err_z_nm'] = float(pts_E[np.argmax(np.abs(E_fv-E_fe(pts_E))), 2]/NM)
            row['fe_coarse_vs_fine_phi'] = relative_error(fe['coarse'][0](pts_phi), phi_fe(pts_phi))
            row['fe_coarse_vs_fine_Ez'] = relative_error(fe['coarse'][1](pts_E), E_fe(pts_E))
            row['mean_Ez_err_vs_exact'] = relative_error(
                out['E_z'].mean(axis=(0, 1)), exact_mean_field(m, P_scale, voltage, model.z))
            rows.append(row)

        for coarse, fine in zip(rows[:-1], rows[1:]):
            for key in ['phi_err_vs_fine', 'Ez_err_vs_fine', 'mean_Ez_err_vs_exact']:
                if min(coarse[key], fine[key]) > 1e-12:  # orders of round-off are meaningless
                    fine[f'order_{key}'] = float(np.log2(coarse[key]/fine[key]))

        print(f"{'nz':>4} {'dz[nm]':>7} {'phi err':>10} {'Ez err':>10} {'order':>6} "
              f"{'max at z[nm]':>12} {'mean Ez vs exact':>17} {'FE c-vs-f Ez':>13}")
        for row in rows:
            print(f"{row['nz_BTO']:4d} {row['dz_BTO_nm']:7.3f} {row['phi_err_vs_fine']:10.2e} "
                  f"{row['Ez_err_vs_fine']:10.2e} {row.get('order_Ez_err_vs_fine', np.nan):6.2f} "
                  f"{row['Ez_max_err_z_nm']:12.2f} {row['mean_Ez_err_vs_exact']:17.2e} "
                  f"{row['fe_coarse_vs_fine_Ez']:13.2e}")
        results['cases'][case] = {'P_scale': P_scale, 'voltage_V': voltage, 'levels': rows}

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == '__main__':
    main()
