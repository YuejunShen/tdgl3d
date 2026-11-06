"""Synchronized orthogonal P/E/potential animations from saved TDGL fields.

The HTML viewers are standalone, need no server or external assets, and use
16-bit display quantization. The NPZ export retains the unquantized float32
cross sections in SI units. These are visualization helpers, not solver changes.
"""
from pathlib import Path
import base64
import json
import numpy as np


QUANTITIES = {
    'P': ('Pz', 'C/m²', 1., 'RdBu_r'),
    'Ex': ('Ex', 'MV/m', 1e-6, 'RdBu_r'),
    'Ey': ('Ey', 'MV/m', 1e-6, 'RdBu_r'),
    'Ez': ('Ez', 'MV/m', 1e-6, 'RdBu_r'),
    'Emag': ('|E|', 'MV/m', 1e-6, 'viridis'),
    'phi': ('φ', 'mV', 1e3, 'RdBu_r'),
}


def lateral_electric_field(phi, Lx, Ly):
    """-grad_xy(phi), using the solver's periodic Fourier discretization."""
    kx = 2*np.pi*np.fft.fftfreq(phi.shape[0], Lx/phi.shape[0])
    ky = 2*np.pi*np.fft.fftfreq(phi.shape[1], Ly/phi.shape[1])
    Ex = np.fft.ifft(-1j*kx[:, None, None]*np.fft.fft(phi, axis=0), axis=0).real
    Ey = np.fft.ifft(-1j*ky[None, :, None]*np.fft.fft(phi, axis=1), axis=1).real
    return Ex, Ey


def build_orthogonal_fields(result, model, view, frame_ids, origin_s=0.):
    """Central or user-selected orthogonal planes; full insulating stack in z."""
    m, g = model.m, model.g
    def cut(key, coords, length):
        value = view.get(key)
        if value is None:
            return len(coords)//2
        if not np.isfinite(value) or not 0 <= value*1e-9 <= length:
            raise ValueError(f'{key} must lie inside the corresponding physical box.')
        return int(np.argmin(abs(coords-value*1e-9)))
    zi = cut('xy_z_nm', model.z, m.h_f)
    yi = cut('xz_y_nm', model.y, g.Ly)
    xi = cut('yz_x_nm', model.x, g.Lx)
    fields = {f'{q}_{plane}': [] for q in QUANTITIES for plane in ('xy', 'xz', 'yz')}
    max_gauss = 0.
    for i in frame_ids:
        P = result['P_snapshots'][i].astype(float)
        voltage = float(result['snapshot_V'][i])
        electrical = model.electrostatics(P, voltage)
        phi, D = electrical['phi'], electrical['D_faces']
        max_gauss = max(max_gauss, electrical['gauss_relative_residual'])
        p_stack = np.zeros_like(phi)
        p_stack[..., model.fe] = P
        Ez = (.5*(D[..., :-1]+D[..., 1:])-p_stack)/model.eps
        np.testing.assert_allclose(Ez[..., model.fe], electrical['E_z'], rtol=1e-10, atol=1e-6)
        # Recover the actual prescribed electrode values from the half cells.
        np.testing.assert_allclose(phi[..., 0]+(D[..., 0]-p_stack[..., 0])
            *model.widths[0]/(2*model.eps[0]), voltage, atol=1e-11)
        np.testing.assert_allclose(phi[..., -1]-(D[..., -1]-p_stack[..., -1])
            *model.widths[-1]/(2*model.eps[-1]), 0., atol=1e-11)
        Ex, Ey = lateral_electric_field(phi, g.Lx, g.Ly)
        P_full = np.full_like(phi, np.nan)
        P_full[..., model.fe] = P
        quantities = dict(P=P_full, Ex=Ex, Ey=Ey, Ez=Ez,
                          Emag=np.sqrt(Ex**2+Ey**2+Ez**2), phi=phi)
        for q, a in quantities.items():
            fields[f'{q}_xy'].append(a[:, :, model.fe.start+zi].T)
            fields[f'{q}_xz'].append(a[:, yi, :].T)
            fields[f'{q}_yz'].append(a[xi, :, :].T)
    fields = {key: np.asarray(value, dtype=np.float32) for key, value in fields.items()}
    fields.update(t=np.asarray(result['snapshot_t'])[frame_ids],
        V=np.asarray(result['snapshot_V'])[frame_ids], frame_ids=np.asarray(frame_ids),
        origin_s=np.array(origin_s), t0_s=np.array(m.t0), z_faces_m=model.z_faces,
        x_faces_m=np.linspace(0, g.Lx, g.nx+1), y_faces_m=np.linspace(0, g.Ly, g.ny+1),
        xy_z_m=np.array(model.z[zi]), xz_y_m=np.array(model.y[yi]), yz_x_m=np.array(model.x[xi]),
        fe_thickness_m=np.array(m.h_f), max_gauss_relative_residual=np.array(max_gauss))
    if max_gauss > 1e-9:
        raise ArithmeticError('Gauss-law check failed during animation-field reconstruction.')
    # Shared lines must agree between the three independently extracted planes.
    for q in QUANTITIES:
        np.testing.assert_allclose(fields[f'{q}_xy'][:, yi, :],
                                   fields[f'{q}_xz'][:, model.fe.start+zi, :], equal_nan=True)
        np.testing.assert_allclose(fields[f'{q}_xy'][:, :, xi],
                                   fields[f'{q}_yz'][:, model.fe.start+zi, :], equal_nan=True)
        np.testing.assert_allclose(fields[f'{q}_xz'][:, :, xi],
                                   fields[f'{q}_yz'][:, :, yi], equal_nan=True)
    return fields


def write_field_viewers(fields, model, out, fps=8):
    """Export a combined viewer, one viewer per plane, exact arrays, and preview."""
    from matplotlib import colormaps
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out/'orthogonal_fields.npz', **fields)
    xf, yf, zf = [fields[key]*1e9 for key in ('x_faces_m', 'y_faces_m', 'z_faces_m')]
    planes = {
        'xy': dict(label='x–y', cut=f'z = {float(fields["xy_z_m"])*1e9:.2f} nm',
                   horizontal='x', vertical='y', hf=xf.tolist(), vf=yf.tolist()),
        'xz': dict(label='x–z', cut=f'y = {float(fields["xz_y_m"])*1e9:.2f} nm',
                   horizontal='x', vertical='z', hf=xf.tolist(), vf=zf.tolist()),
        'yz': dict(label='y–z', cut=f'x = {float(fields["yz_x_m"])*1e9:.2f} nm',
                   horizontal='y', vertical='z', hf=yf.tolist(), vf=zf.tolist()),
    }
    quantities = {}
    encoded = {}
    max_error = {}
    for q, (label, unit, factor, cmap) in QUANTITIES.items():
        maximum = max(float(np.nanmax(abs(fields[f'{q}_{p}']))) for p in planes)*factor
        maximum = max(maximum, 1e-12)
        low, high = (0., maximum) if q == 'Emag' else (-maximum, maximum)
        palette = np.rint(colormaps[cmap](np.linspace(0, 1, 256))[:, :3]*255).astype(int)
        quantities[q] = dict(label=label, unit=unit, low=low, high=high, palette=palette.tolist())
        if q.startswith('E'):
            zcenters = (fields['z_faces_m'][:-1]+fields['z_faces_m'][1:])/2
            inside = (zcenters >= 0) & (zcenters <= model.m.h_f)
            bto_max = max(float(np.max(abs(fields[f'{q}_xy']))),
                *(float(np.max(abs(fields[f'{q}_{p}'][:, inside, :]))) for p in ('xz', 'yz')))*factor
            bto_max = max(bto_max, 1e-12)
            quantities[q]['bto'] = dict(low=0. if q == 'Emag' else -bto_max, high=bto_max)
        max_error[q] = 0.
        for p in planes:
            a = fields[f'{q}_{p}'].astype(float)*factor
            finite = np.isfinite(a)
            codes = np.zeros(a.shape, dtype='<u2')
            codes[finite] = 1+np.rint(np.clip((a[finite]-low)/(high-low), 0, 1)*65534).astype(np.uint16)
            recovered = low+(codes[finite].astype(float)-1)/65534*(high-low)
            max_error[q] = max(max_error[q], float(np.max(abs(recovered-a[finite]))))
            assert max_error[q] <= (high-low)/65534*.501
            encoded[f'{q}_{p}'] = dict(shape=list(a.shape), b64=base64.b64encode(codes.tobytes()).decode())
    metadata = dict(planes=planes, quantities=quantities,
        time=((fields['t']-float(fields['origin_s']))/model.m.t0).tolist(),
        voltage=fields['V'].tolist(), fps=fps, hf_nm=model.m.h_f*1e9,
        lambda_bottom_nm=None if model.m.lambda_bottom is None else model.m.lambda_bottom*1e9,
        lambda_top_nm=None if model.m.lambda_top is None else model.m.lambda_top*1e9)
    for selected, name in [(list(planes), 'cross_sections.html')]+[
            ([p], f'cross_section_{p}.html') for p in planes]:
        payload = dict(metadata, planes={p: planes[p] for p in selected},
                       arrays={key: value for key, value in encoded.items() if key.rsplit('_', 1)[1] in selected})
        html = HTML_TEMPLATE.replace('/*DATA*/', json.dumps(payload, separators=(',', ':')))
        (out/name).write_text(html)
    checks = dict(frames=len(fields['t']), planes=list(planes), quantities=list(QUANTITIES),
        max_gauss_relative_residual=float(fields['max_gauss_relative_residual']),
        electrode_potentials_checked=True, Ez_matches_solver_checked=True,
        html_quantization='16-bit; code 0 reserved for dielectric P mask',
        max_display_error_in_display_units=max_error,
        array_units='SI: P C/m², E V/m, phi V, coordinates m, time s',
        shared_cross_section_lines_checked=True,
        color_scales='Fixed per quantity across all selected times and all three planes; optional BTO electric-field range clips dielectric extrema explicitly')
    (out/'animation_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    plot_final_sections(fields, out)
    return out/'cross_sections.html'


def plot_final_sections(fields, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 3, figsize=(12, 11), layout='constrained')
    for row, q in enumerate(('P', 'Ez', 'phi')):
        label, unit, factor, cmap = QUANTITIES[q]
        limit = max(float(np.nanmax(abs(fields[f'{q}_{p}']))) for p in ('xy', 'xz', 'yz'))*factor
        limit = max(limit, 1e-12)
        for col, p in enumerate(('xy', 'xz', 'yz')):
            horizontal = 'y' if p == 'yz' else 'x'
            vertical = 'y' if p == 'xy' else 'z'
            hf = fields[f'{horizontal}_faces_m']*1e9
            vf = fields[f'{vertical}_faces_m']*1e9
            ax = axes[row, col]
            ax.set_facecolor('#dedede')
            im = ax.pcolormesh(hf, vf, fields[f'{q}_{p}'][-1]*factor,
                cmap=cmap, vmin=-limit, vmax=limit, shading='flat', rasterized=True)
            if p != 'xy':
                for z in (0, float(fields['fe_thickness_m'])*1e9):
                    ax.axhline(z, color='#333', ls='--', lw=.65)
            ax.set(xlabel=horizontal+' (nm)', ylabel=vertical+' (nm)', aspect='equal')
            if row == 0:
                ax.set_title(p[0]+'–'+p[1]+' section')
            ax.grid(False)
        fig.colorbar(im, ax=axes[row, :].tolist(), label=f'{label} ({unit})', shrink=.85)
    tau=(fields['t'][-1]-float(fields['origin_s']))/float(fields['t0_s'])
    fig.suptitle(f'Robin spontaneous-domain run · t/t₀ = {tau:g} · V = {fields["V"][-1]:g} V\n'
                 'Gray in P panels: dielectric; dashed lines: BTO interfaces')
    fig.savefig(out/'final_cross_sections.png', dpi=160)
    fig.savefig(out/'final_cross_sections.pdf')
    plt.close(fig)


def _centers(faces):
    faces = np.asarray(faces, dtype=float)
    return (faces[:-1]+faces[1:])/2


def _site_outlines(fields, model, in_bto):
    """delta_alpha in the three cut planes (zero outside BTO), or (None, None)."""
    delta = getattr(model, 'delta_alpha', None)
    if delta is None or not np.any(delta):
        return None, None
    zi = int(np.argmin(abs(model.z-float(fields['xy_z_m']))))
    yi = int(np.argmin(abs(model.y-float(fields['xz_y_m']))))
    xi = int(np.argmin(abs(model.x-float(fields['yz_x_m']))))
    stack = np.zeros((model.g.nx, model.g.ny, len(in_bto)))
    stack[..., in_bto] = delta
    planes = {'xy': delta[:, :, zi].T, 'xz': stack[:, yi, :].T, 'yz': stack[xi, :, :].T}
    peak = float(delta.flat[np.argmax(abs(delta))])
    return planes, .5*peak


ROW_LABELS = {'P': 'P$_z$', 'Ex': 'E$_x$', 'Ey': 'E$_y$', 'Ez': 'E$_z$', 'Emag': '|E|', 'phi': 'φ'}


def write_field_movie(fields, model, path, fps=8, quantities=('P', 'Ez', 'Emag', 'phi'),
                      dpi=100, title=None, traces=None):
    """MP4 (H.264, needs ffmpeg) of P/E/phi in the x-y, x-z and y-z planes.

    Rows are quantities, labeled once on the left. Columns are planes, labeled
    once on top. Every panel has its own color scale, fixed over time and set by
    the largest |value| reached in that panel's BTO cells. The BTO interior is
    therefore never clipped and fills the color range. Only the 1 nm dielectric
    layers can exceed it and appear saturated. Exact values remain in
    orthogonal_fields.npz.

    A bottom strip shows the applied voltage waveform with a moving cursor.
    traces (optional) is the run result with SI 't', 'V' and 'P' arrays. It
    gives the full-resolution waveform and adds the volume-averaged P. Without
    it, the waveform uses only the frame samples.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    if not FFMpegWriter.isAvailable():
        raise RuntimeError('ffmpeg was not found on PATH; it is required for MP4 export.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    planes = ('xy', 'xz', 'yz')
    t0 = float(fields['t0_s'])
    origin = float(fields['origin_s'])
    tau = (np.asarray(fields['t'])-origin)/t0
    nrow = len(quantities)
    zc = (fields['z_faces_m'][:-1]+fields['z_faces_m'][1:])/2
    in_bto = (zc >= 0) & (zc <= float(fields['fe_thickness_m']))
    headers = {'xy': f'x–y plane at z = {float(fields["xy_z_m"])*1e9:.1f} nm\n(axes: x, y in nm)',
               'xz': f'x–z plane at y = {float(fields["xz_y_m"])*1e9:.1f} nm\n(axes: x, z in nm)',
               'yz': f'y–z plane at x = {float(fields["yz_x_m"])*1e9:.1f} nm\n(axes: y, z in nm)'}
    outlines, outline_level = _site_outlines(fields, model, in_bto)
    with matplotlib.rc_context({'axes.grid': False, 'font.size': 12}):
        fig = plt.figure(figsize=(15, 4.3*nrow+3.4), dpi=dpi, layout='constrained')
        rows = np.atleast_1d(fig.subfigures(nrow+1, 1, height_ratios=[1]*nrow+[.5]))
        meshes = []
        for row, (q, sub) in enumerate(zip(quantities, rows)):
            label, unit, factor, cmap = QUANTITIES[q]
            sub.supylabel(f'{ROW_LABELS.get(q, label)}  ({unit})', fontsize=20, fontweight='bold')
            axes = sub.subplots(1, 3)
            for ax, p in zip(axes, planes):
                data = fields[f'{q}_{p}']
                bto = data if p == 'xy' else data[:, in_bto, :]
                limit = max(float(np.nanmax(abs(bto)))*factor, 1e-12)
                low, high = (0., limit) if q == 'Emag' else (-limit, limit)
                horizontal = 'y' if p == 'yz' else 'x'
                vertical = 'y' if p == 'xy' else 'z'
                ax.set_facecolor('#c8c8c8')
                mesh = ax.pcolormesh(fields[f'{horizontal}_faces_m']*1e9,
                    fields[f'{vertical}_faces_m']*1e9, data[0]*factor,
                    cmap=cmap, vmin=low, vmax=high, shading='flat')
                if p != 'xy':
                    for z in (0, float(fields['fe_thickness_m'])*1e9):
                        ax.axhline(z, color='#222', ls='--', lw=.7)
                if outlines is not None:
                    plane = outlines[p]
                    inside = plane < outline_level if outline_level < 0 else plane > outline_level
                    if inside.any():
                        ax.contour(_centers(fields[f'{horizontal}_faces_m'])*1e9,
                                   _centers(fields[f'{vertical}_faces_m'])*1e9, plane,
                                   levels=[outline_level], colors='k', linewidths=.9, linestyles='solid')
                ax.set_aspect('equal')
                if row == nrow-1:
                    ax.set_xlabel(horizontal+' (nm)')
                if row == 0:
                    ax.set_title(headers[p], fontsize=15, pad=10)
                sub.colorbar(mesh, ax=ax, shrink=.85, pad=.02)
                meshes.append((mesh, f'{q}_{p}', factor))
        strip = rows[-1]
        strip.supylabel('Waveform', fontsize=20, fontweight='bold')
        ax = strip.subplots()
        if traces is not None:
            keep = np.r_[True, np.diff(traces['t']) > 0]
            wave_t = (np.asarray(traces['t'])[keep]-origin)/t0
            wave_V = np.asarray(traces['V'])[keep]
        else:
            wave_t, wave_V = tau, np.asarray(fields['V'])
        ax.plot(wave_t, wave_V, color='#333', lw=2)
        voltage_dot, = ax.plot([], [], 'o', color='#333', ms=9)
        ax.set_xlabel('Time since voltage removal / t₀' if origin else 'Time / t₀')
        ax.set_ylabel('V (V)')
        span = max(float(np.ptp(wave_V)), 1e-3)
        ax.set_ylim(float(wave_V.min())-.12*span, float(wave_V.max())+.12*span)
        ax.set_xlim(tau[0], tau[-1])
        ax.grid(True, alpha=.25)
        cursor = ax.axvline(tau[0], color='#c0392b', lw=1.5)
        polarization_dot = None
        if traces is not None:
            twin = ax.twinx()
            P_mean = np.asarray(traces['P'])[keep]
            twin.plot(wave_t, P_mean, color='#2f6aa0', lw=2)
            polarization_dot, = twin.plot([], [], 'o', color='#2f6aa0', ms=9)
            twin.set_ylabel('mean P (C/m²)', color='#2f6aa0')
            twin.tick_params(axis='y', colors='#2f6aa0')
            pspan = max(float(np.ptp(P_mean)), 1e-6)
            twin.set_ylim(float(P_mean.min())-.12*pspan, float(P_mean.max())+.12*pspan)
            ax.set_ylabel('V (V), black')
        heading = fig.suptitle('', fontsize=17)
        footer = ('gray: dielectric (no P)  ·  dashed: BTO interfaces  ·  each panel has its own '
                  'fixed color scale from its BTO cells; the 1 nm dielectric layers may saturate')
        if outlines is not None:
            footer += '\nblack contours: defect sites (half of the peak local change of α₁)'
        fig.supxlabel(footer, fontsize=11, color='#555')
        prefix = (title+'      ') if title else ''

        def draw(i):
            for mesh, key, factor in meshes:
                mesh.set_array((fields[key][i]*factor).ravel())
            cursor.set_xdata([tau[i], tau[i]])
            voltage_dot.set_data([tau[i]], [float(fields['V'][i])])
            if polarization_dot is not None:
                j = int(np.argmin(abs(wave_t-tau[i])))
                polarization_dot.set_data([wave_t[j]], [P_mean[j]])
            heading.set_text(f'{prefix}t / t₀ = {tau[i]:6.2f}      V = {float(fields["V"][i]):+.3f} V')

        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=-1,
                              extra_args=['-pix_fmt', 'yuv420p', '-crf', '18',
                                          '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2'])
        with writer.saving(fig, str(path), dpi):
            for i in range(len(tau)):
                draw(i)
                writer.grab_frame()
        plt.close(fig)
    return path


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robin TDGL: orthogonal field animations</title>
<style>
*{box-sizing:border-box}body{font:15px Arial,sans-serif;color:#263039;background:#f5f6f7;margin:0;padding:22px}
header,.controls,.note{max-width:1440px;margin:0 auto 14px}h1{font-size:23px;margin:0 0 7px}p{margin:6px 0;line-height:1.5}
.controls{background:white;border:1px solid #dce1e5;border-radius:8px;padding:13px;display:flex;align-items:center;gap:14px;flex-wrap:wrap;position:sticky;top:0;z-index:2}
button,select{font:inherit;padding:6px 10px;border:1px solid #bfc8cf;border-radius:5px;background:white;color:#263039}button{cursor:pointer}
input[type=range]{flex:1;min-width:180px}#time{min-width:205px;font-variant-numeric:tabular-nums}.row{max-width:1440px;margin:0 auto 16px}
.row h2{font-size:17px;margin:8px 0}.grid{display:grid;grid-template-columns:repeat(var(--cols),minmax(0,1fr));gap:14px}
.card{background:white;border:1px solid #dce1e5;border-radius:8px;overflow:hidden}.card h3{font-size:15px;text-align:center;margin:13px 0 0}
canvas{display:block;width:100%;height:auto}.note{font-size:13px;color:#52616c}a{color:#275e8a} @media(max-width:750px){.grid{grid-template-columns:1fr}body{padding:10px}.controls{position:static}}
</style>
<header><h1>Polarization, electric field and potential</h1><p id="description"></p></header>
<div class="controls"><button id="play">▶ Play</button><button id="back" title="Previous frame">‹</button><button id="forward" title="Next frame">›</button>
<input id="frame" aria-label="Time frame" type="range" min="0" value="0"><span id="time"></span>
<label>Field <select id="field"><option value="Ez">Ez</option><option value="Ex">Ex</option><option value="Ey">Ey</option><option value="Emag">|E|</option></select></label>
<label>E range <select id="contrast"><option value="full">Full stack</option><option value="bto">BTO contrast</option></select></label>
<label>Speed <select id="speed"><option>4</option><option selected>8</option><option>12</option><option>20</option></select> fps</label>
<label><input type="checkbox" id="all"> Show all field components</label><label><input type="checkbox" id="loop" checked> Loop</label>
<button id="download">Save current view</button></div><main id="main"></main>
<p class="note">Pz is defined only inside BTO; gray indicates dielectric in the Pz panels. E and φ span the full insulating stack. Dashed lines mark the BTO interfaces. Coordinates and dielectric thicknesses are physical; color scales stay fixed across all planes and frames. “BTO contrast” sets the E range from BTO values; larger dielectric values are clipped to the marked color limits. φ is referenced to the grounded top electrode. The display uses 16-bit quantization; the accompanying orthogonal_fields.npz retains the unquantized cross sections in SI units.</p>
<script>
'use strict';const D=/*DATA*/;
const planeNames=Object.keys(D.planes),arrays={};
for(const [key,item] of Object.entries(D.arrays)){
 const raw=atob(item.b64),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);
 arrays[key]={values:new Uint16Array(bytes.buffer),shape:item.shape};delete item.b64;
}
const $=id=>document.getElementById(id);let frame=0,playing=false,last=0,cards=[];
$('frame').max=D.time.length-1;$('speed').value=String(D.fps);
const length=x=>x===null?'∞':Number(x).toString();
$('description').textContent=`Spontaneous domains · Robin λbottom = ${length(D.lambda_bottom_nm)} nm, λtop = ${length(D.lambda_top_nm)} nm · ${D.time.length} saved frames · central orthogonal sections`;
function fmt(v){return Math.abs(v)<1e-12?'0':Number(v.toPrecision(3)).toString()}
function rows(){return $('all').checked?['P','Ex','Ey','Ez','Emag','phi']:['P',$('field').value,'phi']}
function makeCards(){
 cards=[];$('main').replaceChildren();
 for(const q of rows()){
  const section=document.createElement('section');section.className='row';if(planeNames.length===1)section.style.maxWidth='620px';
  const title=document.createElement('h2');title.textContent=`${D.quantities[q].label} (${D.quantities[q].unit})`;section.append(title);
  const grid=document.createElement('div');grid.className='grid';grid.style.setProperty('--cols',planeNames.length);section.append(grid);
  for(const p of planeNames){const card=document.createElement('div');card.className='card';
   const h=document.createElement('h3');h.textContent=`${D.planes[p].label} · ${D.planes[p].cut}`;card.append(h);
   const canvas=document.createElement('canvas');canvas.width=520;canvas.height=510;card.append(canvas);grid.append(card);cards.push({q,p,canvas});
  }$('main').append(section);
 }draw();
}
function drawCard({q,p,canvas}){
 const ctx=canvas.getContext('2d'),plane=D.planes[p],quantity=D.quantities[q],a=arrays[q+'_'+p];
 const clipped=$('contrast').value==='bto'&&quantity.bto,limits=clipped?quantity.bto:quantity;
 const H=a.shape[1],W=a.shape[2],offset=frame*H*W,hf=plane.hf,vf=plane.vf;
 const physW=hf[hf.length-1]-hf[0],physH=vf[vf.length-1]-vf[0];
 const scale=Math.min(325/physW,365/physH),w=Math.round(physW*scale),h=Math.round(physH*scale),left=65+(325-w)/2,top=35+(365-h)/2;
 ctx.fillStyle='white';ctx.fillRect(0,0,520,510);
 const pixels=ctx.createImageData(w,h),palette=quantity.palette;
 for(let y=0;y<h;y++){
  const coord=vf[vf.length-1]-(y+.5)/h*physH;let iy=0;while(iy<H-1&&vf[iy+1]<=coord)iy++;
  for(let x=0;x<w;x++){
   const ix=Math.min(W-1,Math.floor((x+.5)/w*W)),code=a.values[offset+iy*W+ix],idx=4*(y*w+x);
   const value=quantity.low+(code-1)/65534*(quantity.high-quantity.low);
   const color=code===0?[220,223,226]:palette[Math.round(Math.max(0,Math.min(1,(value-limits.low)/(limits.high-limits.low)))*255)];
   pixels.data[idx]=color[0];pixels.data[idx+1]=color[1];pixels.data[idx+2]=color[2];pixels.data[idx+3]=255;
  }
 }ctx.putImageData(pixels,left,top);ctx.strokeStyle='#525e67';ctx.lineWidth=1;ctx.strokeRect(left,top,w,h);
 ctx.font='15px Arial';ctx.fillStyle='#34424c';ctx.textAlign='center';ctx.textBaseline='top';
 for(let i=0;i<=4;i++){const v=hf[0]+physW*i/4,x=left+w*i/4;ctx.fillText(fmt(v),x,top+h+9)}
 ctx.fillText(plane.horizontal+' (nm)',left+w/2,top+h+34);
 ctx.textAlign='right';ctx.textBaseline='middle';
 for(let i=0;i<=4;i++){const v=vf[0]+physH*i/4,y=top+h-h*i/4;ctx.fillText(fmt(v),left-9,y)}
 ctx.save();ctx.translate(18,top+h/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText(plane.vertical+' (nm)',0,0);ctx.restore();
 if(p!=='xy'){
  ctx.setLineDash([5,4]);ctx.strokeStyle='#47515a';for(const z of [0,D.hf_nm]){const y=top+h-(z-vf[0])/physH*h;ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+w,y);ctx.stroke()}ctx.setLineDash([]);
  ctx.textAlign='center';ctx.font='13px Arial';ctx.fillText('Top electrode: 0 V',left+w/2,top-16);ctx.fillText('Bottom electrode: '+fmt(D.voltage[frame])+' V',left+w/2,top+h+63);
 }
 const cx=left+w+22,cw=14;
 for(let i=0;i<h;i++){const rgb=palette[Math.round((h-1-i)/(h-1)*255)];ctx.fillStyle=`rgb(${rgb.join(',')})`;ctx.fillRect(cx,top+i,cw,1)}
 ctx.fillStyle='#34424c';ctx.textAlign='left';ctx.font='14px Arial';ctx.fillText((clipped?'≥':'')+fmt(limits.high),cx+cw+6,top+5);
 ctx.fillText((clipped&&limits.low<0?'≤':'')+fmt(limits.low),cx+cw+6,top+h-5);ctx.fillText(fmt((limits.low+limits.high)/2),cx+cw+6,top+h/2);
}
function draw(){ $('frame').value=frame;$('time').textContent=`t/t₀ = ${D.time[frame].toFixed(2)} · V = ${fmt(D.voltage[frame])} V`;for(const c of cards)drawCard(c);document.documentElement.dataset.ready='true';document.documentElement.dataset.frame=frame;document.documentElement.dataset.panels=cards.length }
function stop(){playing=false;$('play').textContent='▶ Play'}
$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'❚❚ Pause':'▶ Play';last=performance.now()};
$('frame').oninput=()=>{stop();frame=+$('frame').value;draw()};
$('back').onclick=()=>{stop();frame=Math.max(0,frame-1);draw()};$('forward').onclick=()=>{stop();frame=Math.min(D.time.length-1,frame+1);draw()};
$('all').onchange=makeCards;$('field').onchange=makeCards;$('contrast').onchange=draw;
$('download').onclick=()=>{
 const cols=planeNames.length,nrows=cards.length/cols,canvas=document.createElement('canvas');canvas.width=520*cols;canvas.height=510*nrows+60;
 const ctx=canvas.getContext('2d');ctx.fillStyle='white';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#263039';ctx.font='22px Arial';ctx.fillText($('time').textContent,25,35);
 cards.forEach((c,i)=>{const x=i%cols*520,y=Math.floor(i/cols)*510+60;ctx.drawImage(c.canvas,x,y);ctx.font='18px Arial';ctx.fillText(`${D.quantities[c.q].label} (${D.quantities[c.q].unit}) · ${D.planes[c.p].label}`,x+25,y+20)});
 const a=document.createElement('a');a.download='cross-sections-frame-'+frame+'.png';a.href=canvas.toDataURL('image/png');a.click();
};
function tick(now){if(playing&&now-last>=1000/+$('speed').value){last=now;if(frame===D.time.length-1){if($('loop').checked)frame=0;else stop()}else frame++;draw()}requestAnimationFrame(tick)}
const initial=new URLSearchParams(location.hash.slice(1));frame=Math.max(0,Math.min(D.time.length-1,Number(initial.get('frame'))||0));$('all').checked=initial.get('all')==='1';if(['Ex','Ey','Ez','Emag'].includes(initial.get('field')))$('field').value=initial.get('field');if(initial.get('contrast')==='bto')$('contrast').value='bto';
makeCards();requestAnimationFrame(tick);
</script></html>'''
