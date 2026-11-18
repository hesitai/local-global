"""CPU regression suite. Run with the joint-cu118-test Python and unittest.

Tests never launch training or touch real caches/checkpoints. The user's copy
is a read-only pre-change reference where present.
"""
import csv
import dataclasses
import importlib.util
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

STUDENT = Path(__file__).resolve().parents[1]
GEN = STUDENT.parent / 'motion-flow-syn-python-flat'
BASELINE = STUDENT.parent.parent / 'joint2D copy'
sys.path.insert(0, str(STUDENT))
sys.path.insert(0, str(GEN))
import generate_online_2d_epoch as generator
import kernel_response_labels as labels
from kernel_response import augment_response, response_loss, ResponseConditionedLocal
from dataset.dataset_motiondeblur import Online2DPatchTrainDataset
from model import Uformer

torch.set_num_threads(2)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class LabelTests(unittest.TestCase):
    def test_identity_and_opencv_plane_response(self):
        identity = labels.response_from_homographies([np.eye(3)], [1], 32, 32, 64)
        np.testing.assert_array_equal(identity[::2], 1)
        np.testing.assert_array_equal(identity[1::2], 0)
        h = np.asarray([[1.002, .013, 2.13], [-.008, .997, -1.37], [0, 0, 1.]])
        response = labels.response_from_homographies([h], [1], 32, 32, 64)
        yy, xx = np.mgrid[:128, :128]
        to_patch = np.array([[1.,0,-32],[0,1,-32],[0,0,1]])
        for k, f in enumerate(labels.FREQUENCIES):
            plane = np.exp(2j*np.pi*(xx*f[0]+yy*f[1]))
            warped = cv2.warpPerspective(plane.real, to_patch @ h, (64,64)) + 1j*cv2.warpPerspective(plane.imag, to_patch @ h, (64,64))
            measured = warped / plane[32:96,32:96]
            grid = np.stack([measured[dy::16,dx::16] for dy in (7,8) for dx in (7,8)]).mean(0)
            np.testing.assert_allclose(response[2*k]+1j*response[2*k+1], grid, atol=3e-5)

    def test_all_eight_geometric_augmentations(self):
        h = np.asarray([[1.01,.02,1.3],[-.01,.99,-.7],[.0001,-.0002,1.]])
        source = labels.response_from_homographies([np.eye(3),h],[.3,.7],0,0,64)
        for mode in range(8):
            a = np.eye(3)
            if mode & 1:
                a = np.array([[-1,0,63],[0,1,0],[0,0,1]]) @ a
            if mode & 2:
                a = np.array([[1,0,0],[0,-1,63],[0,0,1]]) @ a
            if mode & 4:
                a = np.array([[0,1,0],[1,0,0],[0,0,1]]) @ a
            expected = labels.response_from_homographies([np.eye(3),a@h@np.linalg.inv(a)],[.3,.7],0,0,64)
            actual = augment_response(torch.from_numpy(source), mode).numpy()
            np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_label_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'a.npz'
            x = labels.response_from_homographies([np.eye(3)],[1],0,0,64)
            labels.save_label(p,x,{'seed':1234})
            np.testing.assert_array_equal(labels.load_label(p,64,expected_metadata={'seed':1234}),x)
            with self.assertRaises(ValueError):
                labels.load_label(p,64,expected_metadata={'seed':1})
            with self.assertRaises(ValueError):
                labels.load_label(p,64,backend='cuda')


class ModelTests(unittest.TestCase):
    @staticmethod
    def make(enabled):
        return Uformer(img_size=256, embed_dim=8, depths=[1]*9,
                       num_heads=[1,1,2,4,8,8,4,2,1], use_ddem=True,
                       ddem_dim=8, ddem_blocks=1, use_local_branch=True,
                       use_channel_aware=True, double_process=True,
                       use_kernel_response=enabled)

    def test_original_initialization_preserved_and_new_gradients(self):
        torch.manual_seed(1234)
        old = self.make(False)
        rng = torch.get_rng_state().clone()
        torch.manual_seed(1234)
        new = self.make(True)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name,value in old.state_dict().items():
            self.assertTrue(torch.equal(value, new.state_dict()[name]), name)
        controllers = [(n,m) for n,m in new.named_modules() if isinstance(m,ResponseConditionedLocal)]
        self.assertEqual(len(controllers),8)
        self.assertFalse(any(n.startswith('conv.') for n,m in controllers))
        x = torch.rand(1,3,256,256)
        restored,prior = new(x,return_ddem_features=True)
        self.assertEqual(prior['kernel_response'].shape,(1,64,16,16))
        target = torch.zeros_like(prior['kernel_response']); target[:,::2]=1
        aux,stats=response_loss(prior['kernel_response'],target)
        loss=(restored-x).square().mean()+.01*aux
        loss.backward()
        for name,param in new.named_parameters():
            if 'response_' in name:
                self.assertIsNotNone(param.grad,name)
                self.assertTrue(torch.isfinite(param.grad).all(),name)
        for name,module in controllers:
            self.assertGreater(float(module.gate.weight.grad.abs().sum()),0,name)
            self.assertGreater(float(module.experts[0].weight.grad.abs().sum()),0,name)
        self.assertGreater(float(new.ddem.response_head[-1].weight.grad.abs().sum()),0)
        new.eval()
        with torch.no_grad():
            result,prior=new(torch.rand(1,3,128,256),return_ddem_features=True)
        self.assertEqual(result.shape,(1,3,128,256))
        self.assertEqual(prior['kernel_response'].shape,(1,64,8,16))

    def test_shifted_control_alignment(self):
        module=ResponseConditionedLocal(4).eval()
        x=torch.rand(1,32,32,4)
        response=torch.rand(1,64,2,2)
        captured=[]
        handle=module.gate.register_forward_pre_hook(lambda m,args: captured.append(args[0].detach()))
        with torch.no_grad():
            module(x,response,0)
            module(torch.roll(x,(-4,-4),(1,2)),response,4)
        handle.remove()
        torch.testing.assert_close(captured[1],torch.roll(captured[0],(-4,-4),(-2,-1)))

    def test_disabled_matches_prechange_model_output(self):
        old_path=BASELINE/'only student  3000/model.py'
        if not old_path.exists():
            self.skipTest('Pre-change model unavailable')
        old_module=import_file('_prechange_model',old_path)
        kw=dict(img_size=256,embed_dim=8,depths=[1]*9,num_heads=[1,1,2,4,8,8,4,2,1],
                use_ddem=True,ddem_dim=8,ddem_blocks=1,use_local_branch=True,
                use_channel_aware=True,double_process=True)
        torch.manual_seed(1234); old=old_module.Uformer(**kw).eval()
        torch.manual_seed(1234); new=Uformer(**kw).eval()
        self.assertEqual(set(old.state_dict()),set(new.state_dict()))
        x=torch.rand(1,3,128,256)
        with torch.no_grad():
            torch.testing.assert_close(old(x),new(x),rtol=0,atol=0)

    def test_losses(self):
        target=torch.zeros(1,64,16,16); target[:,::2]=1
        pred=torch.zeros_like(target); pred[:,1::2]=1
        self.assertAlmostEqual(float(response_loss(pred,target,'magnitude')[0]),0)
        self.assertGreater(float(response_loss(pred,target,'complex')[0]),0)


class GenerationParityTests(unittest.TestCase):
    def test_15_cases_original_off_on_and_epoch_selection(self):
        reference=BASELINE/'motion-flow-syn-python-flat/generate_online_2d_epoch.py'
        if not reference.exists():
            self.skipTest('Read-only pre-change copy unavailable')
        old=import_file('_legacy_generator',reference)
        old_dataset=import_file('_legacy_dataset',BASELINE/'only student  3000/dataset/dataset_motiondeblur.py')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            yy,xx=np.mgrid[:480,:640]
            source=np.stack((xx%251,yy%251,(xx+yy)%251),-1).astype(np.uint8)
            src=root/'source.png'; cv2.imwrite(str(src),source)
            manifests={'old':[],'off':[],'on':[]}
            for i,case in enumerate(generator.motion.blur_cases):
                common=dict(source_index=0,variant_index=i,source_path=str(src),output_name=f'v{i:02}.png',
                            epoch=1,base_seed=1234,patch_size=256,cases=(case,),strength_min=1.,strength_max=1.8,
                            adaptive_subframes=True,fixed_subframes=31,interpolation_margin=2,crop_trials=256)
                for name,mod in [('old',old),('off',generator),('on',generator)]:
                    folder=root/name
                    (folder/'input').mkdir(parents=True,exist_ok=True)
                    (folder/'target').mkdir(exist_ok=True)
                    task=mod.PatchTask(**common,staging_input_path=str(folder/'input'/common['output_name']),
                                       staging_target_path=str(folder/'target'/common['output_name']),
                                       **({'kernel_response':True} if name=='on' else {}))
                    row=mod.render_task(task)
                    row.pop('output_name')
                    row.update(input_path=task.staging_input_path,target_path=task.staging_target_path)
                    manifests[name].append(row)
                for name in ('off','on'):
                    for sub in ('input','target'):
                        self.assertEqual((root/'old'/sub/common['output_name']).read_bytes(),
                                         (root/name/sub/common['output_name']).read_bytes())
                    for key in old.MANIFEST_FIELDS:
                        if key not in ('input_path','target_path'):
                            self.assertEqual(manifests['old'][-1][key],manifests[name][-1][key],key)
                print('PARITY PASS',case,flush=True)
            for name,rows in manifests.items():
                with (root/name/'pairs.tsv').open('w',newline='') as f:
                    writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t')
                    writer.writeheader();writer.writerows(rows[:12])
            for epoch in range(1,501):
                a=old_dataset.Online2DPatchTrainDataset(root/'old/pairs.tsv',{'patch_size':256},epoch)
                b=Online2DPatchTrainDataset(root/'on/pairs.tsv',{'patch_size':256,'use_kernel_response':True},epoch)
                self.assertEqual(a.rows[0]['variant_index'],b.rows[0]['variant_index'])
                if epoch in (1,50,51,500):
                    random.seed(epoch); aa=a[0]
                    random.seed(epoch); bb=b[0]
                    torch.testing.assert_close(aa[0],bb[0],rtol=0,atol=0)
                    torch.testing.assert_close(aa[1],bb[1],rtol=0,atol=0)
                    self.assertEqual(bb[4].shape,(64,16,16))
            print('PARITY PASS epoch selection 1..500 and paired augmentation',flush=True)

    def test_real_gt_cli_pool_and_resume(self):
        import subprocess
        gt=STUDENT/'dataset/UAV_single_image_pose_split/fixed_protocol_first_half_60_78_visdrone_all/train_full_7520'
        python=GEN/'.venv/bin/python'
        if not gt.exists() or not python.exists():
            self.skipTest('Original training data/generator environment unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            common=[str(python),'--epoch','51','--base-seed','1234','--gt-root',str(gt),
                    '--max-images','1','--variants-per-gt','12','--workers','2','--backend','cpu']
            for name,script in [('old',BASELINE/'motion-flow-syn-python-flat/generate_online_2d_epoch.py'),
                                ('on',GEN/'generate_online_2d_epoch.py')]:
                command=[common[0],str(script),*common[1:],'--cache-root',str(root/name)]
                if name=='on':
                    command.append('--kernel-response')
                result=subprocess.run(command,cwd=str(script.parent),capture_output=True,text=True)
                self.assertEqual(result.returncode,0,result.stderr)
            old_root=root/'old/epoch_000051'; new_root=root/'on/epoch_000051'
            def rows(folder):
                with (folder/'pairs.tsv').open() as f:
                    return list(csv.DictReader(f,delimiter='\t'))
            a,b=rows(old_root),rows(new_root)
            self.assertEqual(len(b),12)
            responses=[]
            for ra,rb in zip(a,b):
                for key in generator.MANIFEST_FIELDS:
                    if key not in ('input_path','target_path'):
                        self.assertEqual(ra[key],rb[key],key)
                for key in ('input_path','target_path'):
                    self.assertEqual(Path(ra[key]).read_bytes(),Path(rb[key]).read_bytes())
                responses.append(labels.load_label(rb['kernel_response_path'],256))
            print('REAL GT PARITY PASS: cache epoch 51, 12 candidates, workers=2, original generator environment',flush=True)
            amp=np.sqrt(np.stack(responses)[:,::2]**2+np.stack(responses)[:,1::2]**2)
            print('RESPONSE RADIUS MEAN MAGNITUDES',amp.reshape(12,4,8,16,16).mean((0,2,3,4)).tolist(),flush=True)
            result=subprocess.run(command,cwd=str(GEN),capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Reusing complete',result.stdout)
            bad=command.copy(); bad[bad.index('--base-seed')+1]='4321'
            result=subprocess.run(bad,cwd=str(GEN),capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertTrue((new_root/'pairs.tsv').exists())
            print('CACHE PASS: matching reuse; mismatched seed refused without deleting cache',flush=True)


if __name__=='__main__':
    unittest.main(verbosity=2)
