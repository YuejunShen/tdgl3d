"""Independent physical checks; run with python -m unittest discover -s tests -v."""
from dataclasses import replace
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import solve
from tdgl3d import Material, Grid, NucleationSite, StackModel
from reference_waveforms import PiecewiseVoltage, PeriodicVoltage, pund_waveform, pund_currents


class PhysicsTests(unittest.TestCase):
    def small_model(self, sites=(), hd=1e-9):
        return StackModel(Material(h_d=hd), Grid(nx=4, ny=4, nz=8), sites=sites)

    def test_uniform_electric_field_and_charge(self):
        for hd in [0, 1e-9, 1.4e-9]:
            model = self.small_model(hd=hd)
            m = model.m
            P = np.full((4, 4, 8), -.2)
            e = model.electrostatics(P, .4)
            np.testing.assert_allclose(e['E_z'], m.theta*.4/m.h_f-m.k*(-.2), atol=1e-6)
            np.testing.assert_allclose(e['D_faces'], m.C_area*.4+m.theta*(-.2), atol=1e-13)
            self.assertLess(e['gauss_relative_residual'], 1e-11)

    def test_nonuniform_gauss_law_and_energy_gradient_with_sites(self):
        site = NucleationSite((2e-9, 3e-9, 20e-9), (1e-9, 2e-9, 8e-9),
                              bias=2e5, delta_a=4e6)
        model = self.small_model(sites=[site])
        P = model.initial_polarization(.1)
        field = model.electrostatics(P, .4)
        self.assertLess(field['gauss_relative_residual'], 1e-11)
        direction = np.random.default_rng(41).normal(size=P.shape)*model.m.Ps
        eps = 1e-5
        numerical = (model.energy_per_area(P+eps*direction, .4)
                     -model.energy_per_area(P-eps*direction, .4))/(2*eps)
        analytical = model.m.h_f*np.mean(-model.drift(P, .4)/model.m.gamma*direction)
        np.testing.assert_allclose(numerical, analytical, rtol=3e-7, atol=1e-11)

    def test_electrode_potentials_and_zero_bias_response(self):
        # Recover electrode potentials from adjacent cell potential and D.
        model = self.small_model()
        field = model.electrostatics(model.initial_polarization(.1), .35)
        phi, D = field['phi'], field['D_faces']
        bottom = phi[..., 0]+D[..., 0]*model.widths[0]/(2*model.eps[0])
        top = phi[..., -1]-D[..., -1]*model.widths[-1]/(2*model.eps[-1])
        np.testing.assert_allclose(bottom, .35, atol=1e-12)
        np.testing.assert_allclose(top, 0., atol=1e-12)

    def test_uniform_dynamics_against_independent_radau_and_time_refinement(self):
        model = self.small_model()
        m = model.m
        end = 2*m.t0
        voltage = PiecewiseVoltage((0, .5*end, end), (0, .8, .8))
        times = np.linspace(0, end, 81)
        P0 = np.full((4, 4, 8), -m.Ps)
        coarse = model.simulate(voltage, times, P0, max_step_over_t0=.04)
        fine = model.simulate(voltage, times, P0, max_step_over_t0=.0125)
        def rhs(t, P):
            return -m.gamma*((m.alpha_1+m.k)*P+m.alpha_11*P**3
                            +m.alpha_111*P**5-m.theta*voltage(t)/m.h_f)
        reference = solve_ivp(rhs, (0, end), [-m.Ps], t_eval=times,
                              method='Radau', rtol=1e-10, atol=1e-12,
                              max_step=m.t0/100)
        self.assertTrue(reference.success)
        e1 = np.max(np.abs(coarse['P']-reference.y[0]))
        e2 = np.max(np.abs(fine['P']-reference.y[0]))
        self.assertLess(e2, e1/3)
        self.assertLess(e2/m.Ps, .002)
        self.assertLess(np.max(fine['P_std'])/m.Ps, 1e-10)

    def test_fixed_voltage_energy_decay_including_defects(self):
        site = NucleationSite((3e-9, 3e-9, 25e-9), (2e-9, 2e-9, 8e-9),
                              bias=3e5, delta_a=4e6)
        model = self.small_model(sites=[site])
        times = np.linspace(0, .4*model.m.t0, 61)
        voltage = PiecewiseVoltage((0, times[-1]), (.1, .1))
        r = model.simulate(voltage, times, model.initial_polarization(.1),
                           max_step_over_t0=.002)
        self.assertLess(np.max(np.diff(r['G_area'])), 1e-11)

    def test_current_charge_identity_and_integrated_current(self):
        model = self.small_model()
        times = np.linspace(0, model.m.t0, 61)
        r = model.simulate(PeriodicVoltage(.5, model.m.t0), times,
                           model.initial_polarization(.05), temperature=300, random_seed=7)
        np.testing.assert_allclose(r['D'], model.m.C_area*r['V']+model.m.theta*r['P'], atol=1e-14)
        np.testing.assert_allclose(r['J_total'], r['J_background']+r['J_polarization'], atol=1e-9)
        np.testing.assert_allclose(np.sum(r['J_total']*np.diff(times)), r['D'][-1]-r['D'][0], atol=1e-14)

    def test_noise_reproducibility_and_zero_noise_independence(self):
        model = self.small_model()
        times = np.linspace(0, .02*model.m.t0, 5)
        voltage = PiecewiseVoltage((0, times[-1]), (0, 0))
        a = model.simulate(voltage, times, temperature=300, random_seed=41)
        b = model.simulate(voltage, times, temperature=300, random_seed=41)
        c = model.simulate(voltage, times, temperature=300, random_seed=42)
        np.testing.assert_array_equal(a['P_final'], b['P_final'])
        self.assertGreater(np.max(np.abs(a['P_final']-c['P_final'])), 0)
        a = model.simulate(voltage, times, random_seed=41)
        b = model.simulate(voltage, times, random_seed=42)
        np.testing.assert_array_equal(a['P_final'], b['P_final'])

    def test_noise_mean_variance_matches_fluctuation_dissipation(self):
        model = self.small_model()
        m = model.m
        h = .001
        T = 300.
        count = 3000
        draws = np.random.default_rng(43).normal(size=(4, 4, 8, count))
        sigma = np.sqrt(2*1.380649e-23*T/(abs(m.alpha_1)*model.cell_volume*m.Ps**2))
        increments = m.Ps*model.ifft(sigma*(model._linear_noise_matrix(h)@model.fft(draws)))
        means = increments.mean(axis=(0, 1, 2))
        lam = -(m.alpha_1+m.k)/abs(m.alpha_1)
        expected = (m.Ps*sigma)**2*np.expm1(2*lam*h)/(2*lam)/(4*4*8)
        self.assertLess(abs(np.var(means, ddof=1)/expected-1), .08)
        self.assertLess(abs(means.mean()), 4*np.sqrt(expected/count))

    def test_simple_noise_amplitude_and_reproducibility(self):
        model = self.small_model(hd=0)
        m = model.m
        h = 1e-4
        times = np.array([0, h*m.t0])
        voltage = PiecewiseVoltage((0, times[-1]), (0, 0))
        stds = [model.simulate(voltage, times, noise_over_Ps=.01, random_seed=s,
                               max_step_over_t0=h)['P_std'][-1] for s in range(20)]
        # One step from the uniform state: each cell gets rms .01*Ps*sqrt(h).
        self.assertLess(abs(np.sqrt(np.mean(np.square(stds)))/(.01*m.Ps*np.sqrt(h))-1), .05)
        a = model.simulate(voltage, times, noise_over_Ps=.01, random_seed=3)
        b = model.simulate(voltage, times, noise_over_Ps=.01, random_seed=3)
        np.testing.assert_array_equal(a['P_final'], b['P_final'])
        with self.assertRaises(ValueError):
            model.simulate(voltage, times, noise_over_Ps=-.01)
        with self.assertRaises(ValueError):
            model.simulate(voltage, times, noise_over_Ps=.01, temperature=300)

    def test_hermitian_projection_and_long_run_stability(self):
        for ny in [8, 7]:
            model = StackModel(Material(), Grid(nx=6, ny=ny, nz=4))
            rng = np.random.default_rng(5)
            a = rng.normal(size=(model.nmode, 4, 1))+1j*rng.normal(size=(model.nmode, 4, 1))
            np.testing.assert_allclose(model.hermitian(a.copy()), model.fft(model.ifft(a)),
                                       atol=1e-14)
        # Without the projection, rounding in the redundant ky=0 column grows
        # like exp(0.33 t/t0) and overflows this run after ~110 t0.
        m = Material()
        model = StackModel(m, Grid(nx=24, ny=24, nz=4, Lx=12e-9, Ly=12e-9))
        P0 = .01*m.Ps*np.random.default_rng(0).normal(size=(24, 24, 4))
        r = model.simulate(lambda t: 0., np.linspace(0, 300*m.t0, 4), P0=P0,
                           max_step_over_t0=.05, noise_over_Ps=.01)
        self.assertLess(np.abs(r['P_final']).max(), 1.1*m.Ps)

    def test_noiseless_defect_free_uniform_state_stays_uniform(self):
        model = self.small_model()
        times = np.linspace(0, .5*model.m.t0, 21)
        r = model.simulate(PiecewiseVoltage((0, times[-1]), (.3, .3)), times)
        self.assertLess(r['P_std'].max(), 1e-13)

    def test_pund_preserves_history_and_matches_manual_subtraction(self):
        model = self.small_model()
        m = model.m
        v, block = pund_waveform(.6, .1*m.t0, .4*m.t0, .1*m.t0, .2*m.t0)
        n = 32
        r = model.simulate(v, np.linspace(0, 4*block, 4*n+1), snapshot_stride=n)
        sub = pund_currents(r, n)
        np.testing.assert_allclose(sub['J_PU'], r['J_total'][:n]-r['J_total'][n:2*n])
        # Equivalent two-stage continuation of the same material state.
        first = model.simulate(v, r['t'][:n+1])
        second = model.simulate(v, r['t'][n:2*n+1], P0=first['P_final'])
        np.testing.assert_allclose(second['P'], r['P'][n:2*n+1], atol=2e-12)
        bg = r['J_background'].reshape(4, n)
        np.testing.assert_allclose(bg[0], bg[1], atol=1e-8)

    def test_spatial_refinement_for_smooth_force(self):
        m = Material()
        coarse = StackModel(m, Grid(nx=8, ny=8, nz=16))
        fine = StackModel(m, Grid(nx=16, ny=16, nz=32, nd=4))
        end = .2*m.t0
        voltage = PiecewiseVoltage((0, end), (.15, .15))
        times = np.linspace(0, end, 21)
        def initial(model):
            x = 2*np.pi*model.x[:, None, None]/model.g.Lx
            z = np.pi*model.z[None, None, :]/m.h_f
            return np.broadcast_to(-m.Ps+.02*m.Ps*np.cos(x)*np.cos(z),
                                   (model.g.nx, model.g.ny, model.g.nz)).copy()
        a = coarse.simulate(voltage, times, initial(coarse), max_step_over_t0=.002)
        b = fine.simulate(voltage, times, initial(fine), max_step_over_t0=.002)
        self.assertLess(np.max(np.abs(a['P']-b['P']))/m.Ps, 5e-4)

    def test_robin_linear_boundary_value_problem_against_continuum(self):
        # Independent continuum solution of -P'' + P/ell² = 1/ell².
        # Symmetric Robin faces give P=1-cosh((z-H/2)/ell)/denominator.
        m = Material(lambda_bottom=5e-9, lambda_top=5e-9)
        ell = 10e-9
        errors = []
        for nz in (16, 32, 64):
            model = StackModel(m, Grid(nx=2, ny=2, nz=nz))
            Z = model.gradient_operator[0]/m.G11
            numerical = solve(np.eye(nz)/ell**2-Z, np.ones(nz)/ell**2,
                              assume_a='pos')
            denominator = (np.cosh(m.h_f/(2*ell))
                           +m.lambda_bottom/ell*np.sinh(m.h_f/(2*ell)))
            exact = 1-np.cosh((model.z-m.h_f/2)/ell)/denominator
            errors.append(np.max(np.abs(numerical-exact)))
            np.testing.assert_allclose(numerical, numerical[::-1], atol=1e-13)
        self.assertLess(errors[1], errors[0]/3.5)
        self.assertLess(errors[2], errors[1]/3.5)
        self.assertLess(errors[-1], 5e-4)

    def test_robin_surface_energy_force_and_mean_drift(self):
        # Unequal lengths and a nonuniform field exercise both endpoint signs.
        model = StackModel(Material(lambda_bottom=2e-9, lambda_top=7e-9),
                           Grid(nx=4, ny=4, nz=12))
        m = model.m
        P = model.initial_polarization(.2)
        direction = np.random.default_rng(83).normal(size=P.shape)*m.Ps
        eps = 1e-5
        numerical = (model.energy_per_area(P+eps*direction, .2)
                     -model.energy_per_area(P-eps*direction, .2))/(2*eps)
        analytical = m.h_f*np.mean(-model.drift(P, .2)/m.gamma*direction)
        np.testing.assert_allclose(numerical, analytical, rtol=3e-7, atol=1e-11)
        neumann = StackModel(replace(m, lambda_bottom=None, lambda_top=None), model.g)
        np.testing.assert_allclose(model.energy_per_area(P, .2)
                                   -neumann.energy_per_area(P, .2),
                                   model.boundary_energy_per_area(P), rtol=1e-12)
        faces = model.surface_polarization(P)
        for i, length in enumerate((m.lambda_bottom, m.lambda_top)):
            cell = P[..., 0 if i == 0 else -1]
            np.testing.assert_allclose((cell-faces[..., i])/(model.dz/2),
                                       faces[..., i]/length, rtol=1e-13)
        r = model.simulate(lambda t: .2, np.linspace(0, .1*m.t0, 11),
                           P0=P, max_step_over_t0=.001, snapshot_stride=1)
        for i, snapshot in enumerate(r['P_snapshots']):
            np.testing.assert_allclose(r['dPdt_drift'][i],
                                       np.mean(model.drift(snapshot, .2)), rtol=2e-5)
        self.assertLess(np.max(np.diff(r['G_area'])), 1e-11)
        np.testing.assert_allclose(r['J_total'], r['J_polarization'], atol=1e-9)

    def test_robin_neumann_and_zero_length_limits(self):
        m = Material()
        model = self.small_model()
        almost = StackModel(replace(m, lambda_bottom=1e6, lambda_top=1e6), model.g)
        np.testing.assert_allclose(almost.gradient_operator, model.gradient_operator,
                                   rtol=1e-12, atol=1e-8)
        suppressed = StackModel(replace(m, lambda_bottom=0., lambda_top=0.), model.g)
        np.testing.assert_array_equal(suppressed.surface_polarization(
            np.ones((4, 4, 8))), 0.)
        for invalid in (-1e-9, np.inf, np.nan):
            with self.assertRaises(ValueError):
                Material(lambda_bottom=invalid)

    def test_robin_transient_against_independent_radau(self):
        model = StackModel(Material(lambda_bottom=5e-9, lambda_top=5e-9),
                           Grid(nx=4, ny=4, nz=8))
        m = model.m
        P0 = model.initial_polarization(.15)
        times = np.linspace(0, .8*m.t0, 9)
        reference = solve_ivp(lambda t, p: model.drift(p.reshape(P0.shape), .15).ravel(),
                              (0, times[-1]), P0.ravel(), method='Radau',
                              rtol=1e-10, atol=1e-12)
        self.assertTrue(reference.success)
        errors = []
        for step in (.04, .01):
            r = model.simulate(lambda t: .15, times, P0=P0, max_step_over_t0=step)
            errors.append(np.max(abs(r['P_final'].ravel()-reference.y[:, -1]))/m.Ps)
        self.assertLess(errors[1], errors[0]/8)
        self.assertLess(errors[1], 2e-4)

    def test_invalid_parameters_and_protocols(self):
        with self.assertRaises(ValueError):
            NucleationSite((0, 0, 0), (0, 1e-9, 1e-9))
        with self.assertRaises(ValueError):
            PiecewiseVoltage((0, 0), (0, 1))
        with self.assertRaises(ValueError):
            self.small_model().simulate(lambda t: 0, [0, 0])
        with self.assertRaises(ValueError):
            self.small_model().simulate(lambda t: 0, [0, 1e-6], temperature=-1)


if __name__ == '__main__':
    unittest.main()
