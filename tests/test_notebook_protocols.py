"""Physical continuity and waveform checks for the unified notebook driver."""
from pathlib import Path
from copy import deepcopy
import contextlib,io,json,sys,tempfile,types,unittest
from unittest.mock import patch
import numpy as np
from tdgl3d import Material,Grid,StackModel
from reference_waveforms import PiecewiseVoltage


class NotebookProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        nb=json.loads((root/'3D_TDGL.ipynb').read_text())
        module=types.ModuleType('tdgl_notebook_test');sys.modules[module.__name__]=module
        cls.ns=module.__dict__
        for cell in nb['cells']:
            if any(t in cell.get('metadata',{}).get('tags',[]) for t in
                    ['imports','parameters','definitions','plot-definitions']):
                exec(compile(''.join(cell['source']),'<unified notebook>','exec'),cls.ns)

    def config(self,kind='hold_release'):
        initial=dict(kind='positive',rms_over_Ps=.01,correlation_length_nm=1.,seed=7)
        w=deepcopy(self.ns['waveform']);w.update(kind=kind,hold_t0=.04,relaxation_t0=.06,
            rise_t0=.02,fall_t0=.02,gap_t0=.02,period_t0=.2)
        simulation=dict(noise_over_Ps=0.,seed=42,max_step_t0=.005,output_dt_t0=.01,
                        snapshot_dt_t0=.01,reuse_cache=True)
        return initial,w,simulation

    def run_small(self,folder,kind='hold_release',hd=1e-9):
        initial,w,sim=self.config(kind)
        model,pieces,event,config,out=self.ns['prepare_run'](
            dict(h_d=hd),dict(nx=4,ny=4,nz=8,Lx=8e-9,Ly=8e-9),initial,w,sim,folder)
        with contextlib.redirect_stdout(io.StringIO()):
            phases,result=self.ns['run_protocol'](model,pieces,config,out,initial,sim)
            self.ns['check_result'](model,phases,result)
        return model,pieces,event,config,out,initial,sim,phases,result

    def test_all_waveform_presets_build_finite_voltages(self):
        for kind in ['zero','step_hold','hold_release','triangle','sine','trapezoid','two_step','pund','custom']:
            with self.subTest(kind=kind):
                _,w,_=self.config(kind);pieces,event=self.ns['build_protocol'](w)
                self.assertGreater(sum(p.duration_t0 for p in pieces),0)
                for p in pieces:
                    self.assertTrue(np.isfinite(p.callable(1.)(np.linspace(0,p.duration_t0,7))).all())
                if kind=='hold_release':self.assertEqual(event,w['hold_t0'])

    def test_ideal_release_preserves_field_and_accounts_for_charge(self):
        with tempfile.TemporaryDirectory() as folder:
            model,pieces,event,config,out,initial,sim,phases,r=self.run_small(folder)
            self.assertEqual(len(r['jumps']),1)
            np.testing.assert_allclose(r['jumps'][0,1:],[-.4,-.4*model.m.C_area],atol=1e-12)
            np.testing.assert_allclose(phases[1]['P_snapshots'][0],phases[0]['P_final'],rtol=0,atol=5e-8)
            np.testing.assert_allclose(phases[1]['V'],0)
            np.testing.assert_allclose(phases[1]['J_background'],0)

    def test_matching_cache_reuses_dynamics_and_changes_get_new_folder(self):
        with tempfile.TemporaryDirectory() as folder:
            model,pieces,event,config,out,initial,sim,phases,r=self.run_small(folder)
            with patch.object(model,'simulate',side_effect=AssertionError('Cache not reused')):
                with contextlib.redirect_stdout(io.StringIO()):
                    reused,merged=self.ns['run_protocol'](model,pieces,config,out,initial,sim)
            np.testing.assert_array_equal(merged['P_final'],r['P_final'])
            changed=deepcopy(sim);changed['noise_over_Ps']=.01
            _,_,_,_,new_out=self.ns['prepare_run'](dict(h_d=1e-9),
                dict(nx=4,ny=4,nz=8,Lx=8e-9,Ly=8e-9),initial,
                self.config()[1],changed,folder)
            self.assertNotEqual(out,new_out)

    def test_custom_held_knots_make_exact_jumps(self):
        _,w,_=self.config('custom')
        w.update(custom_times_t0=[0,.1,.2],custom_voltages_V=[.4,-.2,0],custom_interpolation='hold')
        pieces,event=self.ns['build_protocol'](w)
        self.assertEqual([(p.v_start,p.v_end) for p in pieces],[(.4,.4),(-.2,-.2),(0,0)])
        w['relaxation_t0']=0.
        with self.assertRaisesRegex(ValueError,'last held-voltage jump'):self.ns['build_protocol'](w)

    def test_robin_zero_voltage_run_and_cache_identity(self):
        initial,w,sim=self.config('zero');initial['kind']='near_zero'
        geometry=dict(nx=4,ny=4,nz=8,Lx=8e-9,Ly=8e-9)
        material=dict(lambda_bottom=5e-9,lambda_top=5e-9)
        with tempfile.TemporaryDirectory() as folder:
            model,pieces,_,config,out=self.ns['prepare_run'](
                material,geometry,initial,w,sim,folder)
            with contextlib.redirect_stdout(io.StringIO()):
                phases,result=self.ns['run_protocol'](model,pieces,config,out,initial,sim)
                self.ns['check_result'](model,phases,result)
            self.assertEqual(config['material']['lambda_top'],5e-9)
            self.assertLess(np.max(np.diff(result['G_area'])),1e-11)
            _,_,_,_,other=self.ns['prepare_run'](
                dict(lambda_bottom=None,lambda_top=None),geometry,initial,w,sim,folder)
            self.assertNotEqual(out,other)

    def test_continuous_segmented_triangle_matches_single_waveform(self):
        with tempfile.TemporaryDirectory() as folder:
            model,pieces,event,config,out,initial,sim,phases,r=self.run_small(folder,'triangle')
            knots=[0.];volts=[0.]
            for p in pieces:
                knots.append(knots[-1]+p.duration_t0*model.m.t0);volts.append(p.v_end)
            wave=PiecewiseVoltage(tuple(knots),tuple(volts))
            times=np.unique(np.concatenate([phase['t']+offset for phase,offset in
                zip(phases,np.r_[0.,np.cumsum([p.duration_t0*model.m.t0 for p in pieces])[:-1]])]))
            reference=model.simulate(wave,times,P0=self.ns['make_initial_state'](model,initial),
                temperature=0,max_step_over_t0=sim['max_step_t0'])
            np.testing.assert_allclose(r['P_final'],reference['P_final'],rtol=1e-10,atol=1e-11)
            self.assertEqual(len(r['jumps']),0)

    def test_pund_blocks_have_matched_current_bins(self):
        with tempfile.TemporaryDirectory() as folder:
            model,pieces,event,config,out,initial,sim,phases,r=self.run_small(folder,'pund')
            widths=[]
            for label in 'PUND':
                ids=[i for i,p in enumerate(pieces) if p.name.startswith(label+'_')]
                widths.append(np.concatenate([np.diff(phases[i]['t']) for i in ids]))
            for dt in widths[1:]:np.testing.assert_allclose(dt,widths[0],atol=1e-15)
            self.assertEqual(len(r['jumps']),0)

    def test_full_stack_slices_and_direct_electrode_limit(self):
        for hd in [0.,1e-9]:
            with self.subTest(hd=hd),tempfile.TemporaryDirectory() as folder:
                model,pieces,event,config,out,initial,sim,phases,r=self.run_small(folder,hd=hd)
                view=deepcopy(self.ns['view']);view.update(max_frames=6)
                fields=self.ns['build_animation_fields'](r,model,event,view,out)
                self.assertEqual(fields['phi_xz'].shape[1:],(len(model.widths),4))
                np.testing.assert_allclose(fields['z_faces_m'][[0,-1]],[-hd,model.m.h_f+hd])
                np.testing.assert_allclose(fields['V'][0],.4);np.testing.assert_allclose(fields['V'][-1],0)

    def test_optional_stationarity_stops_on_stationary_uniform_state(self):
        model=StackModel(Material(),Grid(nx=4,ny=4,nz=8),sites=())
        options=dict(enabled=True,max_t0=.1,block_t0=.05,max_step_t0=.005,
                     rate_tolerance=1e-3,energy_tolerance=1e-4)
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            result=self.ns['stationary_check'](model,model.uniform_initial(1),0.,options,Path(folder))
            self.assertTrue(result['converged'])
            self.assertLess(result['residual'][-1],1e-3)

if __name__=='__main__':unittest.main()
