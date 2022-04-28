import random

from geom.mesh.op.cpu.remesh import trunc_to_vertex_mask,centralize_mesh
import geom.mesh.io.base
import geom.mesh.io.animate
import glob
import numpy as np

class CompletionSaver:

    def __init__(self, exp_dir, testset_names, extended_save, f,segmentation_manager,centralize_mesh_clouds=True):
        from cfg import SAVE_MESH_AS
        self.save_func = getattr(geom.mesh.io.base, f'write_{SAVE_MESH_AS}')
        self.read_func = getattr(geom.mesh.io.base, f'read_{SAVE_MESH_AS}')
        self.centralize_mesh_clouds = centralize_mesh_clouds
        self.extended_save = extended_save
        self.segmentation_manger = segmentation_manager
        self.f = f  # Might be None

        self.dump_dirs = []
        for ts_name in testset_names:
            dp = exp_dir / 'completions' / ts_name
            dp.mkdir(parents=True, exist_ok=True)
            self.dump_dirs.append(dp)
            dp = exp_dir / 'completions' / ts_name / 'RightArm'
            dp.mkdir(parents=True, exist_ok=True)
            dp = exp_dir / 'completions' / ts_name / 'RightLeg'
            dp.mkdir(parents=True, exist_ok=True)
            dp = exp_dir / 'completions' / ts_name / 'Head'
            dp.mkdir(parents=True, exist_ok=True)
            dp = dp / "test"
            dp.mkdir(parents=True, exist_ok=True)
            dp = dp / "best"
            dp.mkdir(parents=True, exist_ok=True)
            dp = dp / "worst"
            dp.mkdir(parents=True, exist_ok=True)
            dp = dp / "rand"
            dp.mkdir(parents=True, exist_ok=True)
    
    def get_completions_as_pil(self, pred, b):
        # TODO - Make this generic, and not key dependent. Insert support for P2P
        gtrb = pred['completion_xyz']

        # if self.centralize_mesh_clouds:
        #     gt_com = self.segmentation_manger.get_center_of_mass_points_of_segments(b['gt'])['Torso']
        #     tp_com = self.segmentation_manger.get_center_of_mass_points_of_segments(b['tp'])['Torso']
        #     recon_com = self.segmentation_manger.get_center_of_mass_points_of_segments(gtrb)['Torso']
        #     b['gt'] = centralize_mesh(b['gt'],gt_com)
        #     b['tp'] = centralize_mesh(b['tp'],tp_com)
        #     gtrb = centralize_mesh(pred['completion_xyz'],recon_com)
        gt = b['gt']
        tp = b['tp']
        gt_hi = b['gt_hi']
        tp_hi = b['tp_hi']
        if len(gtrb.shape) > 3:
            gtrb = gtrb.reshape(-1, gtrb.shape[-2], gtrb.shape[-1])
        if len(gt.shape) > 3:
            gt = gt.reshape(-1, gt.shape[-2], gt.shape[-1])
        if len(tp.shape) > 3:
            tp = tp.reshape(-1, tp.shape[-2], tp.shape[-1])

        gtrb = gtrb.cpu().numpy()
        pils = []
        
        for i in range(len(b['gt_hi'])):
            gtr_v = gtrb[i, :, :3]
            gt_v = gt[i, :, :3].cpu().numpy()
            tp_v = tp[i, :, :3].cpu().numpy()
            cur_gt_hi = gt_hi[i]
            cur_tp_hi = tp_hi[i]
            if 'gt_f' in b:
                gt_f = b['gt_f'][i]
            else:
                gt_f = self.f
            pils += [geom.mesh.io.base.numpy_to_pil(cur_gt_hi,cur_tp_hi,gtr_v,gt_v,tp_v, gt_f)]
        return pils

    def load_completions(self, set_id=0,test_step=False):
        dump_dp = self.dump_dirs[set_id]
        if test_step:
            dump_dp = dump_dp/"test"
        completions = glob.glob(f"{str(dump_dp)}/*.ply")
        subjects = {}
        for file in completions:
            filename = file.split("/")[-1]
            name_split = filename.split("_")
            subject = name_split[1]
            frame_loc = [i for i, n in enumerate(name_split) if n.isdigit()][1]
            frame = name_split[frame_loc]
            pose = name_split[2:frame_loc]
            pose = '_'.join(str(x) for x in pose)
            if subject not in subjects:
                subjects[subject] = {}
            if pose not in subjects[subject]:
                subjects[subject][pose] = {}
            if frame not in subjects[subject][pose]:
                subjects[subject][pose][frame] = []
            subjects[subject][pose][frame] += [file]
        for subject in subjects:
            for pose in subjects[subject]:
                frame_paths = []
                for frame in sorted(subjects[subject][pose]):
                    frame_paths += [subjects[subject][pose][frame][0]]
                subjects[subject][pose] = frame_paths
                geometries_comp = [geom.mesh.io.base.read_ply_verts(path) for path in subjects[subject][pose]]
                geom.mesh.io.animate.animate(geometries_comp, self.f, str(dump_dp / f"{subject}_{pose}.gif"),
                                             titles=[f"{subject}_{pose}"] * len(frame_paths))
                yield str(dump_dp / f"{subject}_{pose}.gif"), geometries_comp, f"{subject}_{pose}"



    def save_completions_by_batch(self, pred, b, set_id,test_step_folder=False, organ=None, selection=None,metric = None):
        dump_dp = self.dump_dirs[set_id]
        if test_step_folder:
            dump_dp = dump_dp / "test"
        
        if selection != None:
            dump_dp = dump_dp / selection
        if metric != None:
            dump_dp = dump_dp / metric
        if organ != None:
            dump_dp = dump_dp / organ
            dump_dp.mkdir(parents=True, exist_ok=True)

        if len(gtrb.shape) > 3:
            gtrb = gtrb.reshape(-1, gtrb.shape[-2], gtrb.shape[-1])
    
        # TODO - Make this generic, and not key dependent. Insert support for P2P
        gtrb = pred['completion_xyz']
        if not isinstance(gtrb,np.ndarray):
            gtrb = gtrb.cpu().numpy()
        for i, (gt_hi, tp_hi) in enumerate(zip(b['gt_hi'], b['tp_hi'])):
            postfix = f"{gt_hi}_tp_{tp_hi}_{organ}" if organ != None else f"{gt_hi}_tp_{tp_hi}"
            gt_hi, tp_hi = '_'.join(str(x) for x in gt_hi), '_'.join(str(x) for x in tp_hi)
            gtr_v = gtrb[i, :, :3]
            if 'gt_f' in b:
                gt_f = b['gt_f'][i]
            else:
                gt_f = self.f
            self.save_func(dump_dp / f'gt_{postfix}_res', gtr_v, gt_f)

            if self.extended_save:
                gt_v = b['gt'][i, :, :3]
                self.save_func(dump_dp / f'gt_{postfix}_gt', gt_v, gt_f)
                if 'gt_part_f' in b and 'gt_part_v' in b: 
                    gt_part_v = b['gt_part_v'][i][:,:3]
                    gt_part_f = b['gt_f'][i]
                    self.save_func(dump_dp / f'gt_{postfix}_gtpart', gt_part_v, gt_part_f)
                if 'gt_mask' in b:
                    gt_part_v, gt_part_f = trunc_to_vertex_mask(gt_v, gt_f, b['gt_mask'][i])
                    self.save_func(dump_dp / f'gt_{postfix}_gtpart', gt_part_v, gt_part_f)
                elif 'gt_noise' in b:  # TODO - Quick hack for gt_noise. Fix this
                    self.save_func(dump_dp / f'gt_{postfix}_gtnoise', b['gt_noise'][i], None)

                if 'tp_f' in b:
                    tp_f = b['tp_f'][i]
                else:
                    tp_f = self.f
                tp_v = b['tp'][i, :, :3]
                self.save_func(dump_dp / f'gt_{postfix}_tp', tp_v, tp_f)
