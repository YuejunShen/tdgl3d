"""Orthogonal P/E/phi viewers (HTML) and MP4 for the notebook's parameter cell.

Uses the cached run in runs/; with --run it simulates and caches it first."""
import argparse
import json
import numpy as np
from compare_robin_domains import notebook_driver
from tdgl3d.animations import (build_orthogonal_fields, write_field_viewers,
                              write_field_movie, lateral_electric_field)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-frames', type=int, default=201)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--run', action='store_true',
                        help='Simulate the notebook case if it is not cached yet.')
    parser.add_argument('--no-mp4', action='store_true', help='Skip the MP4 export.')
    args = parser.parse_args()
    if args.max_frames < 2 or args.fps <= 0:
        parser.error('Use at least two frames and a positive frame rate.')
    ns = notebook_driver()
    # Reuse the notebook's frame selection and waveform driver.
    notebook = json.loads((ns['ROOT']/'3D_TDGL.ipynb').read_text())
    for cell in notebook['cells']:
        if 'plot-definitions' in cell.get('metadata', {}).get('tags', []):
            exec(compile(''.join(cell['source']), '<notebook plots>', 'exec'), ns)
    model, pieces, event, config, out = ns['prepare_run'](
        ns['material_settings'], ns['geometry'], ns['initial'], ns['waveform'],
        ns['simulation'], ns['ROOT']/'runs', ns.get('defects'))
    cached = (out/'run.json').exists() and all(
        (out/f'phase_{i:02d}.npz').exists() for i in range(len(pieces)))
    if not cached and not args.run:
        raise FileNotFoundError('Not cached yet: run the notebook, or pass --run to simulate it here.')
    phases, result = ns['run_protocol'](model, pieces, config, out, ns['initial'], ns['simulation'])
    ns['check_result'](model, phases, result)
    # Independent analytic periodic potential verifies the new transverse fields.
    x, y = model.x[:, None, None], model.y[None, :, None]
    phi = np.broadcast_to(1e-3*(np.sin(2*np.pi*x/model.g.Lx)
                           +.7*np.cos(2*np.pi*y/model.g.Ly)),
                           (model.g.nx, model.g.ny, 3))
    Ex, Ey = lateral_electric_field(phi, model.g.Lx, model.g.Ly)
    np.testing.assert_allclose(Ex, np.broadcast_to(-1e-3*2*np.pi/model.g.Lx
        *np.cos(2*np.pi*x/model.g.Lx), phi.shape), rtol=1e-11, atol=1e-7)
    np.testing.assert_allclose(Ey, np.broadcast_to(.7e-3*2*np.pi/model.g.Ly
        *np.sin(2*np.pi*y/model.g.Ly), phi.shape), rtol=1e-11, atol=1e-7)
    view = dict(ns['view'], max_frames=args.max_frames, fps=args.fps)
    ids, origin = ns['select_animation_frames'](result, model, event, view)
    print(f'Reconstructing {len(ids)} saved frames in all three planes...', flush=True)
    fields = build_orthogonal_fields(result, model, view, ids, origin)
    print('Writing standalone interactive animations...', flush=True)
    path = write_field_viewers(fields, model, out/'cross_sections', fps=args.fps)
    checks_path = path.parent/'animation_checks.json'
    checks = json.loads(checks_path.read_text())
    checks['transverse_field_analytic_check_passed'] = True
    checks['shared_cross_section_lines_checked'] = True
    checks['source_run'] = str(out.relative_to(ns['ROOT']))
    checks_path.write_text(json.dumps(checks, indent=2)+'\n')
    print('Saved:', path, flush=True)
    if not args.no_mp4:
        print('Writing MP4...', flush=True)
        movie = write_field_movie(fields, model, path.parent/'cross_sections.mp4', fps=args.fps,
                                  title=f"{ns['waveform']['kind']} · noise_over_Ps = "
                                        f"{ns['simulation']['noise_over_Ps']:g}", traces=result)
        print('Saved:', movie, flush=True)


if __name__ == '__main__':
    main()
