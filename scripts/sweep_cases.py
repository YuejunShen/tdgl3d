"""Run the notebook's parameter-cell case for a list of variants and compare them.

Variants are given either as JSON or, for the common case of a dielectric
thickness sweep, with --dielectric-nm. That option keeps the dielectric cell size
fixed (--dielectric-cell-nm, default 0.5 nm). A thickness of 0 means ideal
electrodes. Example, a pulse-and-release protocol for several thicknesses:

    python scripts/sweep_cases.py --dielectric-nm 0 0.5 1 2 3 4 \\
        --waveform '{"kind": "hold_release", "hold_t0": 10, "relaxation_t0": 100}' \\
        --simulation '{"snapshot_dt_t0": 0.5}'

General form: --cases '[{"label": "...", "material": {...}, "geometry": {...},
"defects": {...}}, ...]', where each entry overrides the notebook settings.

Each case is cached in its own runs/<kind>_<hash>/ folder with the
cross-section MP4. runs/sweep_<name>_<kind>_<hash>/ holds the relaxation
comparison, plateau and time-constant summary, final domain snapshots,
a side-by-side MP4 and summary.json.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json

import numpy as np
from compare_robin_domains import ROOT, notebook_driver, save_figure
from compare_cases import release_metrics, loop_metrics, plot_relaxation
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tdgl3d.animations import build_orthogonal_fields, write_field_movie


def palette(n):
    return [plt.cm.viridis(x) for x in np.linspace(0, .88, n)]


def unique_traces(result, model, origin_t0):
    keep = np.r_[True, np.diff(result['t']) > 0]
    return result['t'][keep]/model.m.t0-origin_t0, result['P'][keep]


def plot_summary(cases, out, title, xlabel, values):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3), layout='constrained')
    P_inf = [c['fit']['final_mean_P_C_m2'] for c in cases]
    taus = [c['fit']['tail_time_constant_t0'] for c in cases]
    pos = [c['metrics']['final_positive_fraction'] for c in cases]
    axes[0].plot(values, P_inf, 'o-', color='#2f6aa0', lw=2, ms=8)
    for v, P, c in zip(values, P_inf, cases):
        if not c['fit']['settled']:
            axes[0].plot(v, P, 'o', mfc='white', mec='#2f6aa0', ms=11, mew=2)
    axes[0].set(xlabel=xlabel, ylabel='Mean P at the end (C/m²)',
                title='Final mean P (open: still evolving)')
    finite = [(v, t) for v, t in zip(values, taus) if t is not None]
    if finite:
        axes[1].plot(*zip(*finite), 'o-', color='#d0603a', lw=2, ms=8)
    axes[1].set(xlabel=xlabel, ylabel='τ (t₀)', title='Exponential-tail time constant\n'
                '(missing: decay not a clean exponential)')
    axes[2].plot(values, pos, 'o-', color='#555', lw=2, ms=8)
    axes[2].set(xlabel=xlabel, ylabel='Fraction of cells with P > 0',
                title='Reversed-domain content at the end', ylim=(-.05, 1.05))
    for ax in axes:
        ax.grid(True, alpha=.3)
    fig.suptitle(title)
    save_figure(fig, out, 'plateau_summary')


def plot_final_domains(cases, out, title):
    Ps = cases[0]['model'].m.Ps
    fig, axes = plt.subplots(2, len(cases), figsize=(3.3*len(cases), 7.5),
                             layout='constrained', squeeze=False)
    for col, case in enumerate(cases):
        model, P = case['model'], case['result']['P_final']
        g = model.g
        xf = np.linspace(0, g.Lx, g.nx+1)*1e9
        yf = np.linspace(0, g.Ly, g.ny+1)*1e9
        im = axes[0, col].pcolormesh(xf, yf, P[:, :, g.nz//2].T, cmap='RdBu_r',
                                     vmin=-Ps, vmax=Ps, shading='flat')
        full = np.full((g.nx, len(model.widths)), np.nan)
        full[:, model.fe] = P[:, g.ny//2, :]
        axes[1, col].set_facecolor('#c8c8c8')
        axes[1, col].pcolormesh(xf, model.z_faces*1e9, full.T, cmap='RdBu_r',
                                vmin=-Ps, vmax=Ps, shading='flat')
        for z in (0, model.m.h_f*1e9):
            axes[1, col].axhline(z, color='#222', ls='--', lw=.6)
        axes[0, col].set(title=case['label'], xlabel='x (nm)', aspect='equal')
        axes[1, col].set(xlabel='x (nm)', aspect='equal')
        for ax in axes[:, col]:
            ax.grid(False)
    axes[0, 0].set_ylabel('mid-film x–y\ny (nm)')
    axes[1, 0].set_ylabel('mid-y x–z\nz (nm)')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='P$_z$ (C/m²)', shrink=.7)
    fig.suptitle(title)
    save_figure(fig, out, 'final_domains')


def sweep_movie(cases, out, fps, title):
    fields = [c['fields'] for c in cases]
    n = min(len(f['t']) for f in fields)
    for f in fields[1:]:
        np.testing.assert_allclose(f['t'][:n], fields[0]['t'][:n])
    Ps = cases[0]['model'].m.Ps
    t0 = float(fields[0]['t0_s'])
    origin = float(fields[0]['origin_s'])/t0
    plt.rcParams.update({'font.size': 12, 'axes.grid': False})
    fig = plt.figure(figsize=(3.4*len(cases)+1.5, 12), layout='constrained')
    rows = fig.subfigures(3, 1, height_ratios=[1, 1.35, .8])
    rows[0].supylabel('P$_z$, mid-film x–y', fontsize=15, fontweight='bold')
    rows[1].supylabel('P$_z$, mid-y x–z', fontsize=15, fontweight='bold')
    top = rows[0].subplots(1, len(cases))
    bottom = rows[1].subplots(1, len(cases))
    meshes = []
    for case, f, a0, a1 in zip(cases, fields, top, bottom):
        for ax, p, h, v in ((a0, 'xy', 'x', 'y'), (a1, 'xz', 'x', 'z')):
            ax.set_facecolor('#c8c8c8')
            mesh = ax.pcolormesh(f[f'{h}_faces_m']*1e9, f[f'{v}_faces_m']*1e9, f[f'P_{p}'][0],
                                 cmap='RdBu_r', vmin=-Ps, vmax=Ps, shading='flat')
            meshes.append((mesh, f, f'P_{p}'))
            ax.set(aspect='equal', xlabel=h+' (nm)')
            if p == 'xz':
                for z in (0, float(f['fe_thickness_m'])*1e9):
                    ax.axhline(z, color='#222', ls='--', lw=.6)
        a0.set_title(case['label'], fontsize=14)
    rows[1].colorbar(mesh, ax=bottom.tolist(), label='P$_z$ (C/m²)', shrink=.8)
    ax = rows[2].subplots()
    markers = []
    for case, colour in zip(cases, palette(len(cases))):
        tau, P = unique_traces(case['result'], case['model'], origin)
        ax.plot(tau, P, color=colour, lw=2, label=case['label'])
        markers.append((ax.plot([], [], 'o', color=colour, ms=8)[0], tau, P))
    ax.set(xlabel='Time since voltage removal / t₀' if origin else 'Time / t₀',
           ylabel='Mean P (C/m²)')
    ax.grid(True, alpha=.25)
    ax.legend(fontsize=10, ncol=min(len(cases), 6), loc='upper right')
    cursor = ax.axvline(0, color='#c0392b', lw=1.5)
    heading = fig.suptitle('', fontsize=16)
    writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=-1,
                          extra_args=['-pix_fmt', 'yuv420p', '-crf', '18',
                                      '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
    path = out/'comparison.mp4'
    with writer.saving(fig, str(path), 100):
        for i in range(n):
            tau_i = float(fields[0]['t'][i])/t0-origin
            for mesh, f, key in meshes:
                mesh.set_array(f[key][i].ravel())
            for marker, tau, P in markers:
                j = int(np.argmin(abs(tau-tau_i)))
                marker.set_data([tau[j]], [P[j]])
            cursor.set_xdata([tau_i, tau_i])
            heading.set_text(f'{title}\nt / t₀ = {tau_i:6.2f}      V = {float(fields[0]["V"][i]):+.3f} V')
            writer.grab_frame()
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dielectric-nm', type=float, nargs='+',
                        help='Dielectric thickness of EACH layer, in nm (0: ideal electrodes).')
    parser.add_argument('--dielectric-cell-nm', type=float, default=.5)
    parser.add_argument('--cases', type=json.loads, help='JSON list of case overrides.')
    parser.add_argument('--name', default=None, help='Short sweep name for the output folder.')
    parser.add_argument('--waveform', type=json.loads, default={})
    parser.add_argument('--simulation', type=json.loads, default={})
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--plots-only', action='store_true',
                        help='Use cached runs; redraw figures but skip all MP4s.')
    args = parser.parse_args()
    if (args.dielectric_nm is None) == (args.cases is None):
        parser.error('Give exactly one of --dielectric-nm or --cases.')
    ns = notebook_driver()
    notebook = json.loads((ROOT/'3D_TDGL.ipynb').read_text())
    for cell in notebook['cells']:
        if 'plot-definitions' in cell.get('metadata', {}).get('tags', []):
            exec(compile(''.join(cell['source']), '<notebook plots>', 'exec'), ns)
    view = dict(ns['view'])
    if args.fps is not None:
        view['fps'] = args.fps
    waveform = dict(ns['waveform'], **args.waveform)
    simulation = dict(ns['simulation'], **args.simulation)
    initial = ns['initial']
    kind = waveform['kind']
    if args.dielectric_nm is not None:
        if min(args.dielectric_nm) < 0 or args.dielectric_cell_nm <= 0:
            parser.error('Thicknesses must be nonnegative and the cell size positive.')
        specs = [dict(label=f'h_d = {h:g} nm', value=h, material=dict(h_d=h*1e-9),
                      geometry=dict(nd=max(1, int(round(h/args.dielectric_cell_nm)))))
                 for h in args.dielectric_nm]
        name, xlabel = args.name or 'dielectric', 'Dielectric thickness per layer h_d (nm)'
    else:
        specs = [dict(c, value=c.get('value', i)) for i, c in enumerate(args.cases)]
        name, xlabel = args.name or 'cases', 'case'
    if kind == 'hold_release':
        shape = (f"hold {waveform['amplitude_V']:+g} V for {waveform['hold_t0']:g} t₀, "
                 f"then 0 V for {waveform['relaxation_t0']:g} t₀")
    else:
        shape = f"{kind}, {waveform['amplitude_V']:g} V"
    label = f"{shape} · noise {simulation['noise_over_Ps']:g}"
    cases, manifests = [], []
    for spec in specs:
        settings = dict(ns['material_settings'], **spec.get('material', {}))
        geometry = dict(ns['geometry'], **spec.get('geometry', {}))
        defects = spec.get('defects')
        model, pieces, event, config, out = ns['prepare_run'](
            settings, geometry, initial, waveform, simulation, ROOT/'runs', defects)
        print(f"{spec['label']}: {out.relative_to(ROOT)}", flush=True)
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
                                      fps=view['fps'], title=f"{spec['label']} · {label}",
                                      traces=result)
            print('  saved', movie.relative_to(ROOT), flush=True)
        if kind in ('triangle', 'sine'):
            metrics = loop_metrics(result, model, float(waveform['period_t0']))
        elif event is not None:
            metrics = release_metrics(result, model, event)
        else:
            metrics = dict(final_mean_P_C_m2=float(result['P'][-1]),
                           final_positive_fraction=float(result['positive_fraction'][-1]),
                           final_spatial_std_P_C_m2=float(result['P_std'][-1]))
        cases.append(dict(label=spec['label'], value=spec['value'], model=model,
                          result=result, fields=fields, metrics=metrics, event=event))
        manifests.append(dict(label=spec['label'], value=spec['value'],
                              run_folder=str(out.relative_to(ROOT)), config=config, metrics=metrics))
        print(json.dumps(metrics, indent=2), flush=True)
    signature = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT/'runs'/f'sweep_{name}_{kind}_{signature}'
    out.mkdir(parents=True, exist_ok=True)
    event = cases[0]['event']
    origin = 0. if event is None else float(event)
    title = f'{label}'
    fits = plot_relaxation(cases, out, f'Mean-P relaxation · {title}', origin,
                           colours=palette(len(cases)))
    plot_summary(cases, out, f'Relaxation summary · {title}', xlabel, [c['value'] for c in cases])
    plot_final_domains(cases, out, f'Final domain state (t/t₀ = {cases[0]["result"]["t"][-1]/cases[0]["model"].m.t0-origin:g} after removal) · {title}')
    for m, c in zip(manifests, cases):
        m['relaxation_fit'] = c['fit']
    (out/'summary.json').write_text(json.dumps(dict(cases=manifests,
        sweep_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),
        indent=2)+'\n')
    print(json.dumps(fits, indent=2), flush=True)
    if not args.plots_only:
        sweep_movie(cases, out, view['fps'], f'Sweep · {title}')
    print('Sweep saved:', out.relative_to(ROOT), flush=True)


if __name__ == '__main__':
    main()
