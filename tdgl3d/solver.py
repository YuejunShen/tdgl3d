"""Scalar Pz(x,y,z) TDGL with two explicit dielectric layers.

Periodic Fourier discretization in x,y; conservative, cell-centered electrostatics
across the entire layered stack in z. Polarization exists only in the FE. The
linear Poisson response is cached, then updated exactly for every polarization
state. Exponential Heun (ETD2) integrates the linear gradient/electrostatic terms
exactly and the nonlinear Landau terms to second order. Fixed Gaussian
nucleation sites modify the local free energy. Optional fluctuation-dissipation
noise uses exact linear stochastic convolution at the chosen cell cutoff.
Adapted from the existing ../tdgl3d.py in this research workspace.
"""

from dataclasses import dataclass
import numpy as np
from scipy.linalg import eigh, solve_banded


@dataclass(frozen=True)
class Material:
    alpha_1: float = -1.8558e7
    alpha_11: float = -2.097e8
    alpha_111: float = 1.0352e10
    eps0: float = 8.8541878128e-12
    eps_f: float = 600.0
    eps_d: float = 300.0
    h_f: float = 50e-9
    h_d: float = 1e-9  # EACH dielectric layer
    gamma: float = 1e-3
    G11: float = 2e-10  # V m^3 / C; z gradient
    G44: float = 1e-11  # V m^3 / C; x and y gradients
    # Outward-normal Robin BC: dP/dn = -P/lambda, independently at each BTO face.
    # None gives Neumann; zero is the fully suppressed (Dirichlet) limit.
    lambda_bottom: float | None = None
    lambda_top: float | None = None

    def __post_init__(self):
        lengths = ('lambda_bottom', 'lambda_top')
        if not all(np.isfinite(v) for key, v in vars(self).items() if key not in lengths):
            raise ValueError("Material parameters must be finite.")
        for key in lengths:
            value = getattr(self, key)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError("Extrapolation lengths must be None or finite and nonnegative.")
        if min(self.eps0, self.eps_f, self.eps_d, self.h_f, self.gamma) <= 0:
            raise ValueError("Permittivities, film thickness, and mobility must be positive.")
        if min(self.h_d, self.G11, self.G44) < 0:
            raise ValueError("Dielectric thickness and gradient coefficients must be nonnegative.")
        if self.alpha_1 >= 0 or self.alpha_111 <= 0:
            raise ValueError("The bare-film scaling requires alpha_1 < 0 and alpha_111 > 0.")

    @property
    def Ps(self):
        return np.sqrt((-self.alpha_11 + np.sqrt(self.alpha_11**2
                        - 4*self.alpha_1*self.alpha_111))/(2*self.alpha_111))

    @property
    def t0(self):
        return 1/(self.gamma*abs(self.alpha_1))

    @property
    def theta(self):
        return (self.h_f/self.eps_f)/(self.h_f/self.eps_f + 2*self.h_d/self.eps_d)

    @property
    def k(self):
        return 2*self.h_d/(self.eps0*(self.eps_d*self.h_f + 2*self.eps_f*self.h_d))

    @property
    def C_area(self):
        return self.eps0/(self.h_f/self.eps_f + 2*self.h_d/self.eps_d)


@dataclass(frozen=True)
class Grid:
    nx: int = 16
    ny: int = 16
    nz: int = 32  # FE cells; each dielectric has nd cells
    nd: int = 2
    Lx: float = 8e-9
    Ly: float = 8e-9

    def __post_init__(self):
        if any(not isinstance(n, (int, np.integer)) or n < 2
               for n in [self.nx, self.ny, self.nz]):
            raise ValueError("nx, ny, nz must be integers >= 2.")
        if not isinstance(self.nd, (int, np.integer)) or self.nd < 1:
            raise ValueError("nd must be an integer >= 1.")
        if not np.isfinite(self.Lx+self.Ly) or min(self.Lx, self.Ly) <= 0:
            raise ValueError("Lateral lengths must be finite and positive.")


def triangular_voltage(t, amplitude, period, offset=0.0):
    """Starts at offset, initially rising; exact piecewise-linear waveform."""
    phase = np.mod(np.asarray(t)/period + 0.25, 1.0)
    return offset + amplitude*(1.0 - 4.0*np.abs(phase-0.5))


@dataclass(frozen=True)
class NucleationSite:
    """Fixed Gaussian defect: center/radius in m, bias in V/m, delta_a in V m/C.

    Positive delta_a softens a negative quadratic coefficient toward zero.
    Positive bias favors positive P; it is a material defect, not extra voltage.
    The Gaussian uses periodic lateral distances and ordinary z distance.
    """
    center: tuple
    radius: tuple
    bias: float = 0.0
    delta_a: float = 0.0

    def __post_init__(self):
        if len(self.center) != 3 or len(self.radius) != 3:
            raise ValueError("Site center and radius must have three components.")
        if not np.all(np.isfinite([*self.center, *self.radius, self.bias, self.delta_a])):
            raise ValueError("Site parameters must be finite.")
        if min(self.radius) <= 0:
            raise ValueError("Site radii must be positive.")


class StackModel:
    def __init__(self, material=None, grid=None, sites=()):
        self.m = material or Material()
        self.g = grid or Grid()
        m, g = self.m, self.g
        self.nxy = g.nx*g.ny
        self.nky = g.ny//2+1
        self.nmode = g.nx*self.nky
        self.x = (np.arange(g.nx)+0.5)*g.Lx/g.nx
        self.y = (np.arange(g.ny)+0.5)*g.Ly/g.ny
        self.dz = m.h_f/g.nz
        self.z = (np.arange(g.nz)+0.5)*self.dz
        nd = g.nd if m.h_d > 0 else 0
        self.fe = slice(nd, nd+g.nz)
        self.widths = np.r_[np.full(nd, m.h_d/max(nd, 1)),
                            np.full(g.nz, self.dz),
                            np.full(nd, m.h_d/max(nd, 1))]
        self.eps = m.eps0*np.r_[np.full(nd, m.eps_d),
                                 np.full(g.nz, m.eps_f), np.full(nd, m.eps_d)]
        self.z_faces = np.r_[0, np.cumsum(self.widths)]-m.h_d
        self.z_stack = (self.z_faces[:-1]+self.z_faces[1:])/2
        kx = 2*np.pi*np.fft.fftfreq(g.nx, g.Lx/g.nx)
        ky = 2*np.pi*np.fft.rfftfreq(g.ny, g.Ly/g.ny)
        self.k2 = (kx[:, None]**2 + ky[None, :]**2).reshape(-1)
        self._build_electrostatics()
        self._build_dynamics()
        self._step_cache = {}
        self._noise_cache = {}
        self.sites = tuple(sites)
        self._build_sites()

    def _build_sites(self):
        """Time-independent defect fields; no cells are pinned or overwritten."""
        self.delta_alpha = np.zeros((self.g.nx, self.g.ny, self.g.nz))
        self.seed_field = np.zeros_like(self.delta_alpha)
        X, Y, Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')
        for site in self.sites:
            cx, cy, cz = site.center
            if not (0 <= cx < self.g.Lx and 0 <= cy < self.g.Ly
                    and 0 <= cz <= self.m.h_f):
                raise ValueError("Nucleation-site centers must lie inside the FE box.")
            dx = (X-cx+self.g.Lx/2) % self.g.Lx-self.g.Lx/2
            dy = (Y-cy+self.g.Ly/2) % self.g.Ly-self.g.Ly/2
            dz = Z-cz
            r = site.radius
            profile = np.exp(-0.5*((dx/r[0])**2+(dy/r[1])**2+(dz/r[2])**2))
            self.delta_alpha += site.delta_a*profile
            self.seed_field += site.bias*profile

    @property
    def cell_volume(self):
        return self.g.Lx/self.g.nx*self.g.Ly/self.g.ny*self.dz

    def uniform_initial(self, sign=-1):
        """Uniform Neumann-stack minimum; a preparation state, not a Robin equilibrium."""
        m = self.m
        discriminant = m.alpha_11**2-4*(m.alpha_1+m.k)*m.alpha_111
        if discriminant <= 0:
            magnitude = 0.0
        else:
            squared = (-m.alpha_11+np.sqrt(discriminant))/(2*m.alpha_111)
            magnitude = np.sqrt(max(0.0, squared))
        if sign not in [-1, 1]:
            raise ValueError("sign must be -1 or +1.")
        return np.full((self.g.nx, self.g.ny, self.g.nz), sign*magnitude)

    def fft(self, a):
        """Physical array (nx,ny,nz,nrun) -> (mode,nz,nrun)."""
        return np.fft.rfft2(a, axes=(0, 1)).reshape(self.nmode, self.g.nz, -1)

    def ifft(self, a):
        return np.fft.irfft2(a.reshape(self.g.nx, self.nky, self.g.nz, -1),
                            s=(self.g.nx, self.g.ny), axes=(0, 1))

    def hermitian(self, a):
        """Restore a(-kx) = conj(a(kx)) in the ky=0 (and even-ny Nyquist) columns.

        irfft2 discards the anti-Hermitian part of these columns, so the
        nonlinear terms never limit it. Rounding errors in it otherwise grow at
        the largest linear rate (about 0.7/t0) and leak into P after ~100 t0.
        """
        g = self.g
        b = a.reshape(g.nx, self.nky, g.nz, -1)
        flip = (-np.arange(g.nx)) % g.nx
        for column in {0, g.ny//2} if g.ny % 2 == 0 else {0}:
            b[:, column] = (b[:, column]+np.conj(b[flip, column]))/2
        return b.reshape(a.shape)

    def _build_electrostatics(self):
        """Face fluxes enforce potential and normal-D continuity at interfaces."""
        m, g = self.m, self.g
        n = self.widths.size
        resistance = self.widths/(2*self.eps)
        conductance = np.r_[1/resistance[0],
                            1/(resistance[:-1]+resistance[1:]),
                            1/resistance[-1]]
        # D_face = T phi + S P + boundary_flux*V.
        T = np.zeros((n+1, n))
        T[0, 0] = -conductance[0]
        T[-1, -1] = conductance[-1]
        S_stack = np.zeros((n+1, n))
        S_stack[0, 0] = S_stack[-1, -1] = 1
        for j in range(1, n):
            T[j, j-1:j+1] = [conductance[j], -conductance[j]]
            S_stack[j, j-1:j+1] = [
                resistance[j-1]/(resistance[j-1]+resistance[j]),
                resistance[j]/(resistance[j-1]+resistance[j])]
        S = S_stack[:, self.fe]
        B = S[:-1]-S[1:]
        boundary = np.zeros(n)
        boundary[0] = conductance[0]
        boundary_flux = np.zeros(n+1)
        boundary_flux[0] = conductance[0]
        self.T, self.S, self.B = T, S, B
        self.boundary_flux = boundary_flux
        self.base_diagonal = conductance[:-1]+conductance[1:]
        self.off_diagonal = -conductance[1:-1]
        self.phi_response = np.empty((self.nmode, n, g.nz))
        self.electric_response = np.empty((self.nmode, g.nz, g.nz))
        self.charge_response = np.empty((self.nmode, g.nz))
        self.phi_voltage = None
        for mode, k2 in enumerate(self.k2):
            ab = np.zeros((3, n))
            ab[0, 1:] = self.off_diagonal
            ab[1] = self.base_diagonal + self.widths*self.eps*k2
            ab[2, :-1] = self.off_diagonal
            phiP = solve_banded((1, 1), ab, B)
            self.phi_response[mode] = phiP
            fluxP = T@phiP + S
            fieldP = (fluxP[self.fe.start:self.fe.stop]
                      + fluxP[self.fe.start+1:self.fe.stop+1])/(2*m.eps0*m.eps_f)
            fieldP -= np.eye(g.nz)/(m.eps0*m.eps_f)
            self.electric_response[mode] = fieldP
            self.charge_response[mode] = fluxP[0]
            if mode == 0:
                self.phi_voltage = solve_banded((1, 1), ab, boundary)
                fluxV = T@self.phi_voltage + boundary_flux
                self.field_voltage = (fluxV[self.fe.start:self.fe.stop]
                                      + fluxV[self.fe.start+1:self.fe.stop+1])/(2*m.eps0*m.eps_f)
                self.charge_voltage = fluxV[0]
        asymmetry = np.max(np.abs(self.electric_response
                                  - self.electric_response.transpose(0, 2, 1)))
        if asymmetry > 1e-7*np.max(np.abs(self.electric_response)):
            raise ArithmeticError("Electrostatic response lost variational symmetry.")

    def _build_dynamics(self):
        m, g = self.m, self.g
        zlap = np.diag(np.r_[-1.0, np.full(g.nz-2, -2.0), -1.0])
        zlap += np.diag(np.ones(g.nz-1), 1)+np.diag(np.ones(g.nz-1), -1)
        zlap /= self.dz**2
        # Eliminate each interface value at its half-cell offset. Positive lambda
        # suppresses P at BOTH faces: P'(bottom)=+P_b/lambda_b,
        # P'(top)=-P_t/lambda_t. The symmetric matrix includes the resulting
        # boundary energy and remains compatible with eigh and thermal noise.
        self.boundary_inverse_lengths = np.array([
            0.0 if length is None else 1/(length+self.dz/2)
            for length in (m.lambda_bottom, m.lambda_top)])
        zlap[0, 0] -= self.boundary_inverse_lengths[0]/self.dz
        zlap[-1, -1] -= self.boundary_inverse_lengths[1]/self.dz
        eye = np.eye(g.nz)
        self.gradient_operator = (m.G11*zlap[None]
                                  - m.G44*self.k2[:, None, None]*eye)
        self.L = ((self.gradient_operator+self.electric_response)
                  - m.alpha_1*eye)/abs(m.alpha_1)
        self.eigenvalues = np.empty((self.nmode, g.nz))
        self.eigenvectors = np.empty_like(self.L)
        for mode in range(self.nmode):
            self.eigenvalues[mode], self.eigenvectors[mode] = eigh(self.L[mode])
        self.c3 = m.alpha_11*m.Ps**2/abs(m.alpha_1)
        self.c5 = m.alpha_111*m.Ps**4/abs(m.alpha_1)
        self.voltage_force = self.nxy*self.field_voltage/(abs(m.alpha_1)*m.Ps)

    def initial_polarization(self, perturbation=0.02):
        """Smooth deterministic seed in x,y,z; volume mean exactly -Ps."""
        x = 2*np.pi*self.x[:, None, None]/self.g.Lx
        y = 2*np.pi*self.y[None, :, None]/self.g.Ly
        z = np.pi*self.z[None, None, :]/self.m.h_f
        seed = ((np.cos(x)+0.7*np.cos(y)+0.5*np.cos(x+y))*np.cos(z)
                + 0.25*np.cos(x-2*y)*(1+0.3*np.cos(2*z)))
        seed -= seed.mean()
        seed /= np.sqrt(np.mean(seed**2))
        return self.m.Ps*(-1+perturbation*seed)

    def electrostatics(self, P, voltage):
        """Full-stack phi, FE Ez and face Dz, for a single physical P field."""
        m, g = self.m, self.g
        P = np.asarray(P, dtype=float)
        if P.shape != (g.nx, g.ny, g.nz):
            raise ValueError("P must have shape (nx,ny,nz).")
        Ph = self.fft(P[..., None])[..., 0]
        phih = np.einsum('mij,mj->mi', self.phi_response, Ph)
        phih[0] += self.nxy*self.phi_voltage*voltage
        fluxh = np.einsum('ij,mj->mi', self.T, phih)+np.einsum('ij,mj->mi', self.S, Ph)
        fluxh[0] += self.nxy*self.boundary_flux*voltage
        Ezh = np.einsum('mij,mj->mi', self.electric_response, Ph)
        Ezh[0] += self.nxy*self.field_voltage*voltage

        def inverse(a):
            return np.fft.irfft2(a.reshape(g.nx, self.nky, -1),
                                s=(g.nx, g.ny), axes=(0, 1))

        # Conservative Gauss residual, including lateral Fourier flux divergence.
        residual = fluxh[:, 1:]-fluxh[:, :-1] + self.widths*self.eps*self.k2[:, None]*phih
        scale = max(1e-30, np.max(np.abs(fluxh)),
                    np.max(np.abs(self.widths*self.eps*self.k2[:, None]*phih)))
        return dict(phi=inverse(phih), E_z=inverse(Ezh), D_faces=inverse(fluxh),
                    gauss_relative_residual=float(np.max(np.abs(residual))/scale))

    def _etd_matrices(self, h):
        key = round(float(h), 12)
        if key not in self._step_cache:
            q = self.eigenvectors
            z = h*self.eigenvalues
            small = np.abs(z) < 1e-5
            phi1 = np.empty_like(z)
            phi2 = np.empty_like(z)
            phi1[small] = 1+z[small]/2+z[small]**2/6+z[small]**3/24
            phi2[small] = 0.5+z[small]/6+z[small]**2/24+z[small]**3/120
            phi1[~small] = np.expm1(z[~small])/z[~small]
            phi2[~small] = (np.expm1(z[~small])-z[~small])/z[~small]**2

            def matrix(values):
                return (q*values[:, None, :])@q.transpose(0, 2, 1)

            self._step_cache[key] = (matrix(np.exp(z)), matrix(h*phi1), matrix(h*phi2))
        return self._step_cache[key]

    def energy_per_area(self, P, voltage=0.0):
        """Reduced fixed-voltage electric enthalpy, J/m² of electrode area.

        The voltage-independent self-field contributes -P*E_dep/2; the applied
        field contributes -P*E_app. The final term includes the voltage source's
        work on the linear background capacitance. With Robin BCs the gradient
        quadratic also includes the eliminated interface/half-cell energy;
        boundary_energy_per_area reports that contribution. Do not add it again.
        """
        m = self.m
        P = np.asarray(P, dtype=float)
        depolarization_field = self.electrostatics(P, 0.0)['E_z']
        gradient_field = self.ifft(self.gradient_operator@self.fft(P[..., None]))[..., 0]
        landau = m.alpha_1*P**2/2 + m.alpha_11*P**4/4 + m.alpha_111*P**6/6
        landau += 0.5*self.delta_alpha*P**2-self.seed_field*P
        gradient = -0.5*np.mean(P*gradient_field)
        electric = -0.5*np.mean(P*depolarization_field)-m.theta*voltage*np.mean(P)/m.h_f
        return m.h_f*(np.mean(landau)+gradient+electric)-0.5*m.C_area*voltage**2

    def surface_polarization(self, P):
        """Reconstructed BTO interface P, ordered (bottom, top); not endpoint cells."""
        P = np.asarray(P)
        factors = 1-self.dz/2*self.boundary_inverse_lengths
        return np.stack((P[..., 0]*factors[0], P[..., -1]*factors[1]), axis=-1)

    def boundary_energy_per_area(self, P):
        """J/m² from the surface energy plus adjacent half-cell gradients.

        Eliminating the face P from G11*(P_cell-P_face)^2/dz +
        G11*P_face^2/(2*lambda) gives G11*P_cell^2/[2*(lambda+dz/2)].
        This contribution is already included in energy_per_area.
        """
        P = np.asarray(P)
        return self.m.G11/2*(self.boundary_inverse_lengths[0]*np.mean(P[..., 0]**2)
                            +self.boundary_inverse_lengths[1]*np.mean(P[..., -1]**2))

    def drift(self, P, voltage):
        """Deterministic dP/dt in SI units; excludes stochastic increments."""
        m = self.m
        gradient = self.ifft(self.gradient_operator@self.fft(P[..., None]))[..., 0]
        E = self.electrostatics(P, voltage)['E_z']
        return -m.gamma*((m.alpha_1+self.delta_alpha)*P+m.alpha_11*P**3
                         +m.alpha_111*P**5-gradient-self.seed_field-E)

    def _nonlinear(self, state, voltage):
        """Dimensionless force outside the cached linear operator."""
        u = self.ifft(state)
        with np.errstate(over='raise', invalid='raise'):
            force = (-self.c3*u**3-self.c5*u**5
                     -self.delta_alpha[..., None]/abs(self.m.alpha_1)*u
                     +self.seed_field[..., None]/(abs(self.m.alpha_1)*self.m.Ps))
        result = self.fft(force)
        result[0] += self.voltage_force[:, None]*float(voltage)
        return result

    def _linear_noise_matrix(self, h):
        """sqrt(integral_0^h exp(2 L s) ds), in dimensionless time."""
        key = round(float(h), 12)
        if key not in self._noise_cache:
            z = 2*h*self.eigenvalues
            ratio = np.ones_like(z)
            nonzero = np.abs(z) > 1e-8
            ratio[nonzero] = np.expm1(z[nonzero])/z[nonzero]
            ratio[~nonzero] = 1+z[~nonzero]/2+z[~nonzero]**2/6
            q = self.eigenvectors
            self._noise_cache[key] = ((q*np.sqrt(h*ratio)[:, None, :])
                                      @q.transpose(0, 2, 1))
        return self._noise_cache[key]

    def simulate(self, voltage, times, P0=None, max_step_over_t0=0.01,
                 temperature=0.0, random_seed=0, snapshot_stride=10,
                 progress=None, noise_over_Ps=0.0):
        """Integrate one waveform, preserving the state between its segments.

        voltage(t) returns a scalar voltage in V. A waveform can provide
        breakpoints(start,end) so integration lands on every corner. Input times
        are physical seconds. Temperature is the noise bath temperature in K;
        it does not silently change the fixed Landau coefficients. Zero disables
        noise. Cell noise obeys <xi_i(t)xi_j(t')>=2 gamma kBT/dV delta_ij delta(t-t').
        noise_over_Ps is a simple alternative, not tied to temperature: after every
        step each cell gets an independent kick of rms noise_over_Ps*Ps*sqrt(dt/t0),
        so uncorrelated kicks add up to rms noise_over_Ps*Ps per t0.

        Deterministic stepping is ETD2. Noise uses the exact linear stochastic
        convolution with independent real-cell Gaussian draws; with nonlinear
        forces the stochastic method is first-order weak. Current is reported
        as charge increments over finite output intervals, never a derivative
        of a white-noise trajectory. dPdt_drift is a separate endpoint quantity.
        """
        m, g = self.m, self.g
        times = np.asarray(times, dtype=float)
        if (times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times))
                or np.any(np.diff(times) <= 0)):
            raise ValueError("times must be finite and strictly increasing.")
        if not np.isfinite(max_step_over_t0) or max_step_over_t0 <= 0:
            raise ValueError("max_step_over_t0 must be positive and finite.")
        if not np.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        if not np.isfinite(noise_over_Ps) or noise_over_Ps < 0:
            raise ValueError("noise_over_Ps must be finite and nonnegative.")
        if temperature and noise_over_Ps:
            raise ValueError("Use either thermal noise or noise_over_Ps, not both.")
        if not isinstance(snapshot_stride, (int, np.integer)) or snapshot_stride < 1:
            raise ValueError("snapshot_stride must be a positive integer.")
        if P0 is None:
            P0 = self.uniform_initial()
        P0 = np.asarray(P0, dtype=float)
        if P0.shape != (g.nx, g.ny, g.nz) or not np.all(np.isfinite(P0)):
            raise ValueError("P0 must be finite with shape (nx,ny,nz).")
        volts = np.array([float(voltage(t)) for t in times])
        if not np.all(np.isfinite(volts)):
            raise ValueError("The waveform returned nonfinite voltages.")
        nt = times.size
        result = {key: np.empty(nt) for key in
                  ['P', 'P_std', 'positive_fraction', 'E_f', 'D', 'dPdt_drift', 'G_area']}
        indices = np.unique(np.r_[np.arange(0, nt, snapshot_stride), nt-1])
        snapshots = np.empty((indices.size, g.nx, g.ny, g.nz), dtype=np.float32)
        U = self.fft(P0[..., None]/m.Ps)
        rng = np.random.default_rng(random_seed)
        kB = 1.380649e-23
        sigma = np.sqrt(2*kB*temperature/(abs(m.alpha_1)*self.cell_volume*m.Ps**2))
        snapshot_index = 0
        steps = 0

        def record(i):
            nonlocal snapshot_index
            P = self.ifft(U)[..., 0]*m.Ps
            Pmean = P.mean()
            # The zero lateral mode contains the exact volume-mean electric field.
            E0 = m.Ps*(self.electric_response[0]@U[0, :, 0]).real/self.nxy
            Emean = np.mean(E0+self.field_voltage*volts[i])
            # The averaged gradient vanishes only in the Neumann case.
            gradient_mean = np.mean(m.Ps*(self.gradient_operator[0]@U[0, :, 0]).real/self.nxy)
            result['P'][i] = Pmean
            result['P_std'][i] = P.std()
            result['positive_fraction'][i] = np.mean(P > 0)
            result['E_f'][i] = Emean
            result['D'][i] = m.C_area*volts[i]+m.theta*Pmean
            result['dPdt_drift'][i] = -m.gamma*(m.alpha_1*Pmean
                + m.alpha_11*np.mean(P**3)+m.alpha_111*np.mean(P**5)
                +np.mean(self.delta_alpha*P-self.seed_field)-gradient_mean-Emean)
            result['G_area'][i] = self.energy_per_area(P, volts[i])
            if snapshot_index < indices.size and i == indices[snapshot_index]:
                snapshots[snapshot_index] = P
                snapshot_index += 1

        record(0)
        for i in range(nt-1):
            corners = (voltage.breakpoints(times[i], times[i+1])
                       if hasattr(voltage, 'breakpoints') else [])
            edges = np.unique(np.r_[times[i], corners, times[i+1]])
            for left, right in zip(edges[:-1], edges[1:]):
                nsub = max(1, int(np.ceil((right-left)/m.t0/max_step_over_t0-1e-10)))
                dt = (right-left)/nsub
                h = dt/m.t0
                expL, hphi1, hphi2 = self._etd_matrices(h)
                noise_matrix = self._linear_noise_matrix(h) if temperature else None
                for j in range(nsub):
                    t = left+j*dt
                    N0 = self._nonlinear(U, voltage(t))
                    predictor = expL@U+hphi1@N0
                    U = predictor+hphi2@(self._nonlinear(predictor, voltage(t+dt))-N0)
                    if temperature:
                        gaussian = rng.normal(size=(*P0.shape, 1))
                        U += sigma*(noise_matrix@self.fft(gaussian))
                    elif noise_over_Ps:
                        gaussian = rng.normal(size=(*P0.shape, 1))
                        U += noise_over_Ps*np.sqrt(h)*self.fft(gaussian)
                    U = self.hermitian(U)
                    steps += 1
            record(i+1)
            if not np.isfinite(result['P'][i+1]):
                raise FloatingPointError("Nonfinite state; reduce max_step_over_t0.")
            if progress is not None and ((i+1) % max(1, (nt-1)//10) == 0):
                progress(i+1, nt-1)
        Pfinal = self.ifft(U)[..., 0]*m.Ps
        dt_out = np.diff(times)
        Jpol = np.diff(result['P'])/dt_out
        Jbg = m.C_area*np.diff(volts)/dt_out
        result.update(t=times.copy(), V=volts, t_current=(times[:-1]+times[1:])/2,
                      V_current=(volts[:-1]+volts[1:])/2, dPdt=Jpol,
                      J_polarization=m.theta*Jpol, J_background=Jbg,
                      J_total=np.diff(result['D'])/dt_out,
                      snapshot_indices=indices, P_snapshots=snapshots, P_final=Pfinal,
                      integration_steps=np.array(steps), temperature=np.array(temperature),
                      noise_over_Ps=np.array(noise_over_Ps),
                      random_seed=np.array(random_seed))
        if not all(np.all(np.isfinite(a)) for a in result.values()):
            raise FloatingPointError("Nonfinite result; reduce max_step_over_t0.")
        return result

