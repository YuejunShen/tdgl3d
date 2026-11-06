"""Run matched, zero-voltage Neumann/Robin domain-formation experiments.

Uses the notebook's parameter cell, initialization, cache, and protocol runner.
Run with the same Python environment as 3D_TDGL.ipynb, from any directory.
"""
from pathlib import Path
from copy import deepcopy
import argparse
import hashlib
import json
import os
import sys
import tempfile
import types

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'tdgl3d_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def notebook_driver():
    module = types.ModuleType('robin_comparison_notebook')
    sys.modules[module.__name__] = module
    notebook = json.loads((ROOT/'3D_TDGL.ipynb').read_text())
    for cell in notebook['cells']:
        if set(cell.get('metadata', {}).get('tags', [])) & {'imports', 'parameters', 'definitions'}:
            exec(compile(''.join(cell['source']), str(ROOT/'3D_TDGL.ipynb'), 'exec'), module.__dict__)
    return module.__dict__


def save_figure(fig, out, stem):
    fig.savefig(out/f'{stem}.png', dpi=170)
    fig.savefig(out/f'{stem}.pdf')
    plt.close(fig)


def final_metrics(model, result):
    P = result['P_final']
    m = model.m
    residual = m.t0/m.Ps*np.sqrt(np.mean(model.drift(P, 0.)**2))
    rms = np.sqrt(np.mean(result['P_snapshots'].astype(float)**2, axis=(1, 2, 3)))
    above = np.flatnonzero(rms >= .1*m.Ps)
    return dict(mean_P_C_m2=float(P.mean()), rms_P_C_m2=float(np.sqrt(np.mean(P**2))),
                mean_abs_P_C_m2=float(np.mean(np.abs(P))),
                spatial_std_P_C_m2=float(P.std()), positive_fraction=float(np.mean(P > 0)),
                surface_rms_P_C_m2=np.sqrt(np.mean(model.surface_polarization(P)**2,
                                                  axis=(0, 1))).tolist(),
                endpoint_cell_rms_P_C_m2=np.sqrt(np.mean(P[..., [0, -1]]**2,
                                                        axis=(0, 1))).tolist(),
                middle_plane_rms_P_C_m2=float(np.sqrt(np.mean(P[..., model.g.nz//2]**2))),
                energy_J_m2=float(result['G_area'][-1]),
                boundary_energy_J_m2=float(model.boundary_energy_per_area(P)),
                dimensionless_rms_drift=float(residual),
                max_recorded_energy_increase_J_m2=float(np.max(np.diff(result['G_area']))),
                time_to_rms_0p1_Ps_t0=None if not len(above) else
                    float(result['snapshot_t'][above[0]]/m.t0),
                gauss_relative_residual=float(model.electrostatics(P, 0.)['gauss_relative_residual']))


def plot_comparison(cases, out):
    colours = ['#467aa9', '#d27954']
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    for (name, model, r, _), colour in zip(cases, colours):
        tau = r['t']/model.m.t0
        axes[0, 0].plot(tau, r['P'], color=colour, label=name)
        rms = np.sqrt(np.mean(r['P_snapshots'].astype(float)**2, axis=(1, 2, 3)))
        axes[0, 1].plot(r['snapshot_t']/model.m.t0, rms, color=colour, label=name)
        axes[1, 0].plot(tau, r['G_area']*1e3, color=colour)
        profile = np.sqrt(np.mean(r['P_final']**2, axis=(0, 1)))
        axes[1, 1].plot(model.z*1e9, profile, color=colour)
        faces = np.sqrt(np.mean(model.surface_polarization(r['P_final'])**2, axis=(0, 1)))
        axes[1, 1].scatter([0, model.m.h_f*1e9], faces, color=colour, marker='o', s=25)
    axes[0, 0].set(ylabel='Mean P (C/m²)', xlabel='Time / t₀')
    axes[0, 0].legend(fontsize=9)
    axes[0, 1].set(ylabel='RMS P (C/m²)', xlabel='Time / t₀')
    axes[1, 0].set(ylabel='Total enthalpy / area (mJ/m²)', xlabel='Time / t₀')
    axes[1, 1].set(ylabel='Final lateral RMS P (C/m²)', xlabel='z inside BTO (nm)')
    axes[1, 1].set_title('Dots: reconstructed interface values', fontsize=10)
    fig.suptitle('Spontaneous domain formation at V = 0; identical initial perturbation; no thermal noise')
    save_figure(fig, out, 'comparison_diagnostics')

    duration = cases[0][2]['t'][-1]/cases[0][1].m.t0
    times = np.unique(np.minimum([0, 2, 5, 10, 30, duration], duration))
    fig, axes = plt.subplots(2, len(times), figsize=(2.5*len(times), 5.5),
                             squeeze=False, layout='constrained')
    for row, (name, model, r, _) in enumerate(cases):
        for col, time in enumerate(times):
            idx = np.argmin(abs(r['snapshot_t']/model.m.t0-time))
            im = axes[row, col].imshow(r['P_snapshots'][idx, :, :, model.g.nz//2].T,
                origin='lower', extent=[0, model.g.Lx*1e9, 0, model.g.Ly*1e9],
                cmap='RdBu_r', vmin=-model.m.Ps, vmax=model.m.Ps, interpolation='nearest')
            axes[row, col].set_title(f't/t₀ = {r["snapshot_t"][idx]/model.m.t0:g}')
            axes[row, col].set_xlabel('x (nm)')
            axes[row, col].grid(False)
        axes[row, 0].set_ylabel(name+'\ny (nm)')
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Mid-film Pz (C/m²)', shrink=.8)
    save_figure(fig, out, 'domain_evolution')

    fig, axes = plt.subplots(2, 2, figsize=(9, 9), layout='constrained')
    for col, (name, model, r, _) in enumerate(cases):
        P = r['P_final']
        axes[0, col].imshow(P[:, :, model.g.nz//2].T, origin='lower',
            extent=[0, model.g.Lx*1e9, 0, model.g.Ly*1e9], cmap='RdBu_r',
            vmin=-model.m.Ps, vmax=model.m.Ps, interpolation='nearest')
        axes[0, col].set(title=name, xlabel='x (nm)', ylabel='y (nm)')
        xfaces = np.linspace(0, model.g.Lx*1e9, model.g.nx+1)
        full = np.full((model.g.nx, len(model.widths)), np.nan)
        full[:, model.fe] = P[:, model.g.ny//2, :]
        axes[1, col].set_facecolor('#e4e4e4')
        im = axes[1, col].pcolormesh(xfaces, model.z_faces*1e9, full.T,
            cmap='RdBu_r', vmin=-model.m.Ps, vmax=model.m.Ps, shading='flat')
        for z in (0., model.m.h_f*1e9):
            axes[1, col].axhline(z, color='#444444', linewidth=.7)
        axes[1, col].set(xlabel='x (nm)', ylabel='z (nm)', aspect='equal')
        axes[1, col].set_title('Mid-y section; gray = dielectric', fontsize=10)
    for ax in axes.flat:
        ax.grid(False)
    fig.colorbar(im, ax=axes.ravel().tolist(), label='Final Pz (C/m²)', shrink=.7)
    fig.suptitle(f'Final domain structure at t/t₀ = {duration:g}')
    save_figure(fig, out, 'final_domains')


def animate_comparison(cases, out):
    # Two mid-film views on one shared physical polarization scale.
    indices = np.unique(np.linspace(0, len(cases[0][2]['snapshot_t'])-1, 80).astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), layout='constrained')
    images = []
    for ax, (name, model, r, _) in zip(axes, cases):
        images.append(ax.imshow(r['P_snapshots'][0, :, :, model.g.nz//2].T,
            origin='lower', extent=[0, model.g.Lx*1e9, 0, model.g.Ly*1e9],
            cmap='RdBu_r', vmin=-model.m.Ps, vmax=model.m.Ps, interpolation='nearest'))
        ax.set(title=name, xlabel='x (nm)', ylabel='y (nm)')
        ax.grid(False)
    fig.colorbar(images[0], ax=axes.tolist(), label='Pz (C/m²)', shrink=.8)
    title = fig.suptitle('')
    def update(index):
        for im, (_, model, r, _) in zip(images, cases):
            im.set_data(r['P_snapshots'][index, :, :, model.g.nz//2].T)
        title.set_text(f'V = 0; t/t₀ = {cases[0][2]["snapshot_t"][index]/cases[0][1].m.t0:g}')
        return images+[title]
    anim = FuncAnimation(fig, update, frames=indices, interval=125, blit=False)
    (out/'domain_formation.html').write_text(
        '<!doctype html><meta charset="utf-8"><title>Robin vs Neumann domain formation</title>'
        +anim.to_jshtml())
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--duration-t0', type=float, default=100.)
    parser.add_argument('--lambda-nm', type=float, default=5.)
    parser.add_argument('--step-t0', type=float, default=.01)
    args = parser.parse_args()
    if min(args.duration_t0, args.lambda_nm, args.step_t0) <= 0:
        parser.error('Duration, extrapolation length, and time step must be positive.')
    ns = notebook_driver()
    initial = deepcopy(ns['initial']); initial['kind'] = 'near_zero'
    waveform = deepcopy(ns['waveform'])
    waveform.update(kind='zero', relaxation_t0=args.duration_t0)
    simulation = deepcopy(ns['simulation'])
    simulation.update(noise_over_Ps=0., max_step_t0=args.step_t0)
    cases = []
    manifests = []
    P0 = None
    for name, length in [('Neumann', None), (f'Robin λ = {args.lambda_nm:g} nm', args.lambda_nm*1e-9)]:
        settings = deepcopy(ns['material_settings'])
        settings.update(lambda_bottom=length, lambda_top=length)
        model, pieces, _, config, out = ns['prepare_run'](
            settings, ns['geometry'], initial, waveform, simulation, ROOT/'runs')
        field = ns['make_initial_state'](model, initial)
        if P0 is None:
            P0 = field
        else:
            np.testing.assert_array_equal(P0, field)
        print(name, out, flush=True)
        phases, result = ns['run_protocol'](model, pieces, config, out, initial, simulation)
        ns['check_result'](model, phases, result)
        assert np.max(np.diff(result['G_area'])) < 1e-10, 'Deterministic energy increased.'
        metrics = final_metrics(model, result)
        assert metrics['gauss_relative_residual'] < 1e-9
        cases.append((name, model, result, out))
        manifests.append(dict(name=name, run_folder=str(out.relative_to(ROOT)),
                              config=config, metrics=metrics))
        print(json.dumps(metrics, indent=2), flush=True)
    signature = hashlib.sha256(json.dumps(manifests, sort_keys=True).encode()).hexdigest()[:12]
    out = ROOT/'runs'/f'robin_domain_comparison_{signature}'
    out.mkdir(parents=True, exist_ok=True)
    (out/'summary.json').write_text(json.dumps(dict(cases=manifests,
        comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')
    plot_comparison(cases, out)
    animate_comparison(cases, out)
    print('Comparison saved:', out, flush=True)


if __name__ == '__main__':
    main()
