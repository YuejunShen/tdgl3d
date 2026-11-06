"""Run the notebook's parameter-cell case as two matched variants.

--compare gradient   material gradient terms vs G11 = G44 = 0. The Robin surface
                     term is proportional to G11, so it also disappears.
--compare pinning    no defects vs the pinning sites in the notebook's `defects`
                     (or --defects JSON). Sites: local alpha_1 -> factor*alpha_1.

Everything else is identical: initial state, waveform, noise seed and time
step. --waveform, --simulation and --defects take JSON that overrides the
notebook settings without editing it, e.g. a positive pulse and relaxation:

    python scripts/compare_cases.py --compare pinning \
        --waveform '{"kind": "hold_release", "hold_t0": 10, "relaxation_t0": 100}' \
        --simulation '{"snapshot_dt_t0": 0.5}' \
        --defects '{"pinning": {"count": 64}}'

Each case is cached in its own runs/<kind>_<hash>/ folder with the
cross-section MP4 and orthogonal_fields.npz. runs/<compare>_comparison_<kind>_<hash>/
holds mean-P traces (P-V loops for periodic waveforms; post-release decay and
an exponential-tail fit for hold_release), a side-by-side MP4 and summary.json.
For hold_release, displayed time zero is voltage removal.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json

import numpy as np
from compare_robin_domains import ROOT, notebook_driver, save_figure
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tdgl3d.animations import (build_orthogonal_fields, write_field_movie,
                              _site_outlines, _centers)


def loop_metrics(result, model, period_t0):
    """Remanent P and coercive voltages from the last full cycle."""
    t = result['t']/model.m.t0
    last = t >= t[-1]-period_t0-1e-9
    # Phase boundaries repeat their timestamp; keep one copy of each sample.
    last &= np.r_[True, np.diff(t) > 0]
    V, P = result['V'][last], result['P'][last]
    crossings = []
    for i in range(len(P)-1):
        if P[i] == 0 or P[i]*P[i+1] < 0:
            f = 0. if P[i] == P[i+1] else P[i]/(P[i]-P[i+1])
            crossings.append(float(V[i]+f*(V[i+1]-V[i])))
    remanent = []
    for i in range(len(V)-1):
        if V[i] == 0 or V[i]*V[i+1] < 0:
            f = 0. if V[i] == V[i+1] else V[i]/(V[i]-V[i+1])
            remanent.append(float(P[i]+f*(P[i+1]-P[i])))
    return dict(final_cycle_P_zero_crossing_V=crossings,
                final_cycle_P_at_V0_C_m2=remanent,
                final_cycle_max_mean_P_C_m2=float(P.max()),
                final_cycle_min_mean_P_C_m2=float(P.min()),
                final_positive_fraction=float(result['positive_fraction'][-1]),
                final_spatial_std_P_C_m2=float(result['P_std'][-1]))


def release_metrics(result, model, event_t0):
    """Mean P at removal and at fixed delays after it (hold_release)."""
    m = model.m
    t = result['t']/m.t0-event_t0
    keep = np.r_[True, np.diff(result['t']) > 0]
    t, P = t[keep], result['P'][keep]
    after = t >= -1e-9
    at = lambda delay: float(np.interp(delay, t[after], P[after]))
    P_release = at(0.)
    below = np.flatnonzero(after & (abs(P) <= abs(P_release)/2))
    return dict(mean_P_at_release_C_m2=P_release,
                mean_P_after_release_C_m2={f'{d:g}_t0': at(d) for d in (1, 2, 5, 10, 20, 50, 100)
                                           if d <= t[-1]+1e-9},
                time_to_half_mean_P_t0=None if not len(below) else float(t[below[0]]),
                final_mean_P_C_m2=float(P[-1]),
                final_positive_fraction=float(result['positive_fraction'][-1]),
                final_spatial_std_P_C_m2=float(result['P_std'][-1]))


def plot_traces(cases, out, title, waveform_kind, origin_t0=0.):
    colours = ['#2f6aa0', '#d0603a']
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), layout='constrained')
    model = cases[0]['model']
    xlabel = 'Time since voltage removal / t₀' if waveform_kind == 'hold_release' else 'Time / t₀'
    t = cases[0]['result']['t']/model.m.t0-origin_t0
    axes[0, 0].plot(t, cases[0]['result']['V'], color='#555')
    axes[0, 0].set(ylabel='V (V)', xlabel=xlabel, title='Applied voltage')
    periodic = waveform_kind in ('triangle', 'sine')
    for case, colour in zip(cases, colours):
        r, m = case['result'], case['model'].m
        tau = r['t']/m.t0-origin_t0
        axes[0, 1].plot(tau, r['P'], color=colour, label=case['name'])
        if periodic:
            axes[1, 0].plot(r['V'], r['P'], color=colour, label=case['name'])
        else:
            after = tau > 0
            axes[1, 0].semilogx(tau[after], r['P'][after], color=colour, label=case['name'])
        axes[1, 1].plot(tau, r['P_std'], color=colour, label=case['name'])
    Ps = model.m.Ps
    for ax in (axes[0, 1], axes[1, 0]):
        for level in (-Ps, Ps):
            ax.axhline(level, color='#999', ls=':', lw=.8)
    axes[0, 1].set(ylabel='Mean P (C/m²)', xlabel=xlabel, title='Volume-averaged P (dotted: ±Ps)')
    if periodic:
        axes[1, 0].set(ylabel='Mean P (C/m²)', xlabel='V (V)', title='P–V loop')
    else:
        axes[1, 0].set(ylabel='Mean P (C/m²)', xlabel=xlabel+' (log)',
                       title='After the voltage is removed')
    axes[1, 1].set(ylabel='Spatial std of P (C/m²)', xlabel=xlabel,
                   title='Spatial inhomogeneity')
    for ax in axes.flat[1:]:
        ax.legend(fontsize=9)
    fig.suptitle(title)
    save_figure(fig, out, 'mean_P_comparison')


def relaxation_fit(tau, P, tail_start):
    """Plateau P_inf, noise floor, and exponential tail time constant (t0)."""
    after = tau > 0
    tail = tau >= tail_start
    P_inf = float(P[tail].mean())
    floor = float(P[tail].std())
    dP = P-P_inf
    window = after & (dP > 50*floor) & (dP < 2e-3)
    if window.sum() < 5:
        return P_inf, floor, None, None
    # The window must be one continuous, monotonically decaying stretch; an
    # overshoot or a decay that crosses P_inf (domain formation) is not an
    # exponential tail and gets no time constant.
    idx = np.flatnonzero(window)
    if np.any(np.diff(idx) != 1) or np.any(np.diff(dP[idx]) > 0):
        return P_inf, floor, None, None
    slope, intercept = np.polyfit(tau[window], np.log(dP[window]), 1)
    residual = np.log(dP[window])-(slope*tau[window]+intercept)
    if slope >= 0 or np.max(abs(residual)) > .15:
        return P_inf, floor, None, None
    return P_inf, floor, -1/slope, (float(tau[window][0]), float(tau[window][-1]), intercept)


def plot_relaxation(cases, out, title, origin_t0, colours=None, settled_floor=1e-4):
    """Mean P after removal on linear and log time, and P - P_inf on a log axis.

    A case counts as settled when the spread of P over the second half of the
    relaxation is below settled_floor (C/m²). Only settled cases get a plateau,
    an exponential-tail fit and a P - P_inf curve. Others are marked as still
    evolving, because their final value is not a plateau.
    """
    colours = colours or ['#2f6aa0', '#d0603a', '#3a9d5d', '#8e5bb5', '#c79a2c', '#555555'][:len(cases)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), layout='constrained')
    fits = {}
    floors = []
    for case, colour in zip(cases, colours):
        name = case.get('name', case.get('label'))
        r, m = case['result'], case['model'].m
        keep = np.r_[True, np.diff(r['t']) > 0]
        tau, P = r['t'][keep]/m.t0-origin_t0, r['P'][keep]
        P_inf, floor, time_constant, window = relaxation_fit(tau, P, tau[-1]/2)
        settled = floor < settled_floor
        fits[name] = dict(settled=bool(settled), final_mean_P_C_m2=float(P[-1]),
                          P_inf_C_m2=P_inf if settled else None,
                          second_half_std_C_m2=floor,
                          tail_time_constant_t0=time_constant if settled else None,
                          fit_window_t0=None if (window is None or not settled) else window[:2])
        case['fit'] = fits[name]
        tag = (f'P∞ = {P_inf:.4f}' if settled else f'still evolving; P(end) = {P[-1]:.4f}')
        axes[0].plot(tau, P, color=colour, lw=2, label=f'{name}  ({tag})')
        after = tau > 0
        axes[1].semilogx(tau[after], P[after], color=colour, lw=2, label=name)
        if settled:
            floors.append(floor)
            for ax in axes[:2]:
                ax.axhline(P_inf, color=colour, ls=':', lw=1)
            dP = np.where(P-P_inf > 0, P-P_inf, np.nan)
            extra = '' if time_constant is None else f'  (τ = {time_constant:.2f} t₀)'
            axes[2].semilogy(tau[after], dP[after], color=colour, lw=2, label=name+extra)
            if window is not None:
                line_t = np.linspace(0, window[1]+1.5, 50)
                axes[2].semilogy(line_t, np.exp(window[2]-line_t/time_constant),
                                 color=colour, ls='--', lw=1.1)
    axes[0].axvline(0, color='k', ls='--', lw=.9)
    axes[0].set(xlabel='Time since voltage removal / t₀', ylabel='Mean P (C/m²)',
                title='Mean P, full run (dotted: plateau of settled cases)')
    axes[1].set(xlabel='Time since voltage removal / t₀ (log)', ylabel='Mean P (C/m²)',
                title='After removal, log time')
    if floors:
        noise = 3*max(floors)
        axes[2].axhspan(1e-12, noise, color='#bbbbbb', alpha=.35, lw=0)
        axes[2].set_ylim(noise/3, .3)
    axes[2].set(xlabel='Time since voltage removal / t₀', ylabel='Mean P − P∞ (C/m²)',
                title='Approach to plateau, settled cases only\n(gray: 3σ noise; dashed: exponential fit)',
                xlim=(0, 15))
    for ax in axes:
        ax.grid(True, alpha=.3)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=9, loc='best')
    fig.suptitle(title)
    save_figure(fig, out, 'relaxation_comparison')
    return fits


def comparison_movie(cases, out, fps, title):
    """Side-by-side Pz (x-y mid-film, x-z mid-y) with the mean-P trace."""
    fields = [c['fields'] for c in cases]
    n = min(len(f['t']) for f in fields)
    for f in fields[1:]:
        np.testing.assert_allclose(f['t'][:n], fields[0]['t'][:n])
    Ps = cases[0]['model'].m.Ps
    plt.rcParams.update({'font.size': 12, 'axes.grid': False})
    fig = plt.figure(figsize=(12, 13), layout='constrained')
    rows = fig.subfigures(3, 1, height_ratios=[1, 1, .7])
    meshes = []
    for row, (case, f, sub) in enumerate(zip(cases, fields, rows)):
        sub.supylabel(case['name'], fontsize=17, fontweight='bold')
        axes = sub.subplots(1, 2)
        for ax, (p, h, v) in zip(axes, [('xy', 'x', 'y'), ('xz', 'x', 'z')]):
            ax.set_facecolor('#c8c8c8')
            mesh = ax.pcolormesh(f[f'{h}_faces_m']*1e9, f[f'{v}_faces_m']*1e9, f[f'P_{p}'][0],
                                 cmap='RdBu_r', vmin=-1.1*Ps, vmax=1.1*Ps, shading='flat')
            meshes.append((mesh, f, f'P_{p}'))
            if p == 'xz':
                for z in (0, float(f['fe_thickness_m'])*1e9):
                    ax.axhline(z, color='#222', ls='--', lw=.7)
            zc = (f['z_faces_m'][:-1]+f['z_faces_m'][1:])/2
            outlines, level = _site_outlines(f, case['model'], (zc >= 0) & (zc <= float(f['fe_thickness_m'])))
            if outlines is not None and (outlines[p] < level).any():
                ax.contour(_centers(f[f'{h}_faces_m'])*1e9, _centers(f[f'{v}_faces_m'])*1e9,
                           outlines[p], levels=[level], colors='k', linewidths=.9, linestyles='solid')
            ax.set(xlabel=h+' (nm)', ylabel=v+' (nm)', aspect='equal')
            if row == 0:
                cut = (f'z = {float(f["xy_z_m"])*1e9:.1f} nm' if p == 'xy'
                       else f'y = {float(f["xz_y_m"])*1e9:.1f} nm')
                ax.set_title(f'P$_z$,  {p[0]}–{p[1]} plane  ({cut})', fontsize=15, pad=10)
        sub.colorbar(mesh, ax=axes.tolist(), label='P$_z$ (C/m²)', shrink=.85)
    ax = rows[2].subplots()
    markers = []
    colours = ['#2f6aa0', '#d0603a']
    for case, colour in zip(cases, colours):
        r, m = case['result'], case['model'].m
        ax.plot(r['t']/m.t0, r['P'], color=colour, lw=2, label=case['name'])
        markers.append(ax.plot([], [], 'o', color=colour, ms=8)[0])
    ax2 = ax.twinx()
    r0, m0 = cases[0]['result'], cases[0]['model'].m
    ax2.plot(r0['t']/m0.t0, r0['V'], color='#888', lw=1, ls='--')
    ax2.set_ylabel('V (V), dashed')
    ax.set(xlabel='Time / t₀', ylabel='Mean P (C/m²)')
    ax.grid(True, alpha=.25)
    ax.legend(loc='upper right', fontsize=11)
    cursor = ax.axvline(0, color='k', lw=.8)
    heading = fig.suptitle('', fontsize=16)
    t0 = float(fields[0]['t0_s'])
    origin = float(fields[0]['origin_s'])/t0
    for line in ax.lines+ax2.lines:
        line.set_xdata(np.asarray(line.get_xdata())-origin)
    ax.relim()
    ax.autoscale_view()
    if origin:
        ax.set_xlabel('Time since voltage removal / t₀')
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=-1,
                          extra_args=['-pix_fmt', 'yuv420p', '-crf', '18',
                                      '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
    path = out/'comparison.mp4'
    with writer.saving(fig, str(path), 100):
        for i in range(n):
            tau = float(fields[0]['t'][i])/t0-origin
            for mesh, f, key in meshes:
                mesh.set_array(f[key][i].ravel())
            for case, marker in zip(cases, markers):
                r = case['result']
                j = int(np.argmin(abs(r['t']/case['model'].m.t0-origin-tau)))
                marker.set_data([r['t'][j]/case['model'].m.t0-origin], [r['P'][j]])
            cursor.set_xdata([tau, tau])
            heading.set_text(f'{title}\nt / t₀ = {tau:6.2f}      V = {float(fields[0]["V"][i]):+.3f} V')
            writer.grab_frame()
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fps', type=int, default=None, help="Default: the notebook's view['fps'].")
    parser.add_argument('--max-frames', type=int, default=None,
                        help="Default: the notebook's view['max_frames'].")
    parser.add_argument('--plots-only', action='store_true',
                        help='Use cached runs; redraw figures but skip all MP4s.')
    parser.add_argument('--waveform', type=json.loads, default={},
                        help="JSON merged into the notebook's waveform dict.")
    parser.add_argument('--simulation', type=json.loads, default={},
                        help="JSON merged into the notebook's simulation dict.")
    parser.add_argument('--compare', choices=('gradient', 'pinning'), default='gradient')
    parser.add_argument('--defects', type=json.loads, default={},
                        help="JSON merged into the notebook's defects['pinning'] (pinning only).")
    args = parser.parse_args()
    ns = notebook_driver()
    notebook = json.loads((ROOT/'3D_TDGL.ipynb').read_text())
    for cell in notebook['cells']:
        if 'plot-definitions' in cell.get('metadata', {}).get('tags', []):
            exec(compile(''.join(cell['source']), '<notebook plots>', 'exec'), ns)
    view = dict(ns['view'])
    if args.fps is not None:
        view['fps'] = args.fps
    if args.max_frames is not None:
        view['max_frames'] = args.max_frames
    waveform = dict(ns['waveform'], **args.waveform)
    simulation = dict(ns['simulation'], **args.simulation)
    initial = ns['initial']
    kind = waveform['kind']
    if kind in ('triangle', 'sine'):
        shape = (f"{kind}, {waveform['amplitude_V']:g} V, period {waveform['period_t0']:g} t₀"
                 f" × {waveform['cycles']}")
    elif kind == 'hold_release':
        shape = (f"hold {waveform['amplitude_V']:+g} V for {waveform['hold_t0']:g} t₀, "
                 f"then 0 V for {waveform['relaxation_t0']:g} t₀")
    else:
        shape = f"{kind}, {waveform['amplitude_V']:g} V"
    label = f"{shape} · noise {simulation['noise_over_Ps']:g}"
    base = deepcopy(ns['material_settings'])
    if args.compare == 'gradient':
        variants = [('with gradient', base, None),
                    ('no gradient (G11 = G44 = 0)', dict(base, G11=0., G44=0.), None)]
        heading = 'With vs without gradient energy'
    else:
        defects = deepcopy(ns.get('defects') or dict(pinning=dict(count=0, alpha1_factor=10.,
                                                                   radius_nm=1., seed=11)))
        defects['pinning'].update(args.defects.get('pinning', {}))
        pin = defects['pinning']
        if not pin['count']:
            raise SystemExit('Set a nonzero pinning count in the notebook or with --defects.')
        def describe(value, unit=''):
            return (f'{value:g}{unit}' if np.ndim(value) == 0
                    else f'{value[0]:g}–{value[1]:g}{unit}')
        variants = [('no pinning sites', base, None),
                    (f"{pin['count']} pinning sites (α₁ × {describe(pin['alpha1_factor'])}, "
                     f"r = {describe(pin['radius_nm'], ' nm')})", base, defects)]
        heading = 'With vs without pinning sites'
    cases, manifests = [], []
    P0 = None
    for name, settings, defects in variants:
        model, pieces, event, config, out = ns['prepare_run'](
            settings, ns['geometry'], initial, waveform, simulation, ROOT/'runs', defects)
        field = ns['make_initial_state'](model, initial)
        if P0 is None:
            P0 = field
        else:
            np.testing.assert_array_equal(P0, field)
        print(f'{name}: {out.relative_to(ROOT)}', flush=True)
        ns['save_sites'](model, out)
        phases, result = ns['run_protocol'](model, pieces, config, out, initial, simulation)
        ns['check_result'](model, phases, result)
        fields = None
        if not args.plots_only:
            ids, origin = ns['select_animation_frames'](result, model, event, view)
            fields = build_orthogonal_fields(result, model, view, ids, origin)
            movie_dir = out/'cross_sections'
            movie_dir.mkdir(exist_ok=True)
            np.savez_compressed(movie_dir/'orthogonal_fields.npz', **fields)
            movie = write_field_movie(fields, model, movie_dir/'cross_sections.mp4',
                                      fps=view['fps'], title=f'{name} · {label}',
                                      traces=result)
            print('  saved', movie.relative_to(ROOT), flush=True)
        if kind in ('triangle', 'sine'):
            metrics = loop_metrics(result, model, float(waveform['period_t0']))
        elif event is not None:
            metrics = release_metrics(result, model, event)
        else:
            metrics = dict(final_mean_P_C_m2=float(result['P'][-1]),
                           final_spatial_std_P_C_m2=float(result['P_std'][-1]))
        cases.append(dict(name=name, model=model, result=result, fields=fields))
        manifests.append(dict(name=name, run_folder=str(out.relative_to(ROOT)),
                              config=config, metrics=metrics))
        print(json.dumps(metrics, indent=2), flush=True)
    signature = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT/'runs'/f'{args.compare}_comparison_{kind}_{signature}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(cases=manifests,
        comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        indent=2)+'\n')
    plot_traces(cases, out, f'{heading} · {label}', kind,
                0. if event is None else float(event))
    if kind == 'hold_release':
        fits = plot_relaxation(cases, out,
            f'Mean-P relaxation after removal · {heading.lower()} · {label}', float(event))
        (out/'relaxation_fit.json').write_text(json.dumps(fits, indent=2)+'\n')
        print(json.dumps(fits, indent=2), flush=True)
    if not args.plots_only:
        comparison_movie(cases, out, view['fps'], f'{heading} · {label}')
    print('Comparison saved:', out.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
