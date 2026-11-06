"""Reference voltage functions for independent solver tests; the notebook owns the simulation presets."""
from dataclasses import dataclass
import numpy as np
from tdgl3d import triangular_voltage


@dataclass(frozen=True)
class PiecewiseVoltage:
    times: tuple
    values: tuple
    name: str = 'custom'

    def __post_init__(self):
        t, v = np.asarray(self.times), np.asarray(self.values)
        if (t.ndim != 1 or t.size < 2 or t.shape != v.shape
                or not np.all(np.isfinite(t)) or not np.all(np.isfinite(v))
                or np.any(np.diff(t) <= 0)):
            raise ValueError('Provide matching finite knots with increasing times.')

    def __call__(self, t):
        return np.interp(t, self.times, self.values)

    def breakpoints(self, start, end):
        knots = np.asarray(self.times)
        return knots[(knots > start) & (knots < end)]


@dataclass(frozen=True)
class PeriodicVoltage:
    amplitude: float
    period: float
    kind: str = 'triangle'
    offset: float = 0.0

    def __post_init__(self):
        if (not np.isfinite(self.amplitude+self.period+self.offset)
                or self.amplitude < 0 or self.period <= 0
                or self.kind not in ['triangle', 'sine']):
            raise ValueError('Use finite amplitude >= 0, period > 0, triangle or sine.')

    def __call__(self, t):
        if self.kind == 'triangle':
            return triangular_voltage(t, self.amplitude, self.period, self.offset)
        return self.offset+self.amplitude*np.sin(2*np.pi*np.asarray(t)/self.period)

    def breakpoints(self, start, end):
        if self.kind == 'sine':
            return np.array([])
        n = np.arange(int(np.floor(2*start/self.period))-2,
                      int(np.ceil(2*end/self.period))+2)
        corners = self.period*(0.25+0.5*n)
        return corners[(corners > start) & (corners < end)]


def example_waveforms(amplitude, duration):
    """All begin at 0 V; pulse protocols end with a zero-voltage relaxation."""
    t, a = duration, amplitude
    return {
        'triangle': PeriodicVoltage(a, t, 'triangle'),
        'sine': PeriodicVoltage(a, t, 'sine'),
        'trapezoid': PiecewiseVoltage(tuple(t*np.array([0, .08, .65, .73, 1])),
                                     (0, a, a, 0, 0), 'trapezoid'),
        'two_step': PiecewiseVoltage(tuple(t*np.array([0, .08, .3, .38, .65, .73, 1])),
                                    (0, .45*a, .45*a, a, a, 0, 0), 'two_step'),
    }


def pund_waveform(amplitude, rise, hold, fall, gap):
    """P,U,N,D with repeated pulse shape and continuous material state.

    Begin from a negatively prepared state. No state reset occurs between P/U
    or N/D. gap includes relaxation after each pulse. The caller decides whether
    the chosen amplitude/hold really saturates the sample.
    """
    if (not np.all(np.isfinite([amplitude, rise, hold, fall, gap]))
            or amplitude <= 0 or min(rise, hold, fall, gap) <= 0):
        raise ValueError('PUND amplitude and durations must be finite and positive.')
    block = rise+hold+fall+gap
    times, values = [0.0], [0.0]
    for i, sign in enumerate([1, 1, -1, -1]):
        start = i*block
        times.extend([start+rise, start+rise+hold,
                      start+rise+hold+fall, (i+1)*block])
        values.extend([sign*amplitude, sign*amplitude, 0.0, 0.0])
    return PiecewiseVoltage(tuple(times), tuple(values), 'PUND'), block


def pund_currents(result, points_per_block):
    """Subtract matched finite-bin terminal currents, exactly as P-U and N-D."""
    n = points_per_block
    if len(result['J_total']) != 4*n:
        raise ValueError('PUND output needs four equal blocks of current bins.')
    J = result['J_total'].reshape(4, n)
    dt = np.diff(result['t'])[:n]
    relative_t = result['t_current'][:n]-result['t'][0]
    pu, nd = J[0]-J[1], J[2]-J[3]
    return dict(t=relative_t, J_PU=pu, J_ND=nd,
                delta_D_PU=np.cumsum(pu*dt), delta_D_ND=np.cumsum(nd*dt))
