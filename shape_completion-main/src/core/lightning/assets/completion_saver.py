import random

from geom.mesh.op.cpu.remesh import trunc_to_vertex_mask
import geom.mesh.io.base
import geom.mesh.io.animate
import glob


class CompletionSaver:

    def __init__(self, exp_dir, testset_names, extended_save, f):
        from cfg import SAVE_MESH_AS
        self.save_func = getattr(geom.mesh.io.base, f'write_{SAVE_MESH_AS}')
        self.read_func = getattr(geom.mesh.io.base, f'read_{SAVE_MESH_AS}')

        self.extended_save = extended_save
        self.f = f  # Might be None

        self.dump_dirs = []
        for ts_name in testset_names:
            dp = exp_dir / 'completions' / ts_name
            dp.mkdir(parents=True, exist_ok=True)
            self.dump_dirs.append(dp)
            dp = dp / "test"
            dp.mkdir(parents=True, exist_ok=True)
            

    def get_completions_as_pil(self, pred, b):
        # TODO - Make this generic, and not key dependent. Insert support for P2P
        gtrb = pred['completion_xyz']
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


    def save_completions_by_batch(self, pred, b, set_id,test_step_folder=False):
        dump_dp = self.dump_dirs[set_id]
        if test_step_folder:
            dump_dp = dump_dp / "test"

        # TODO - Make this generic, and not key dependent. Insert support for P2P
        gtrb = pred['completion_xyz']
        if len(gtrb.shape) > 3:
            gtrb = gtrb.reshape(-1, gtrb.shape[-2], gtrb.shape[-1])
        gtrb = gtrb.cpu().numpy()
        for i, (gt_hi, tp_hi) in enumerate(zip(b['gt_hi'], b['tp_hi'])):
            gt_hi, tp_hi = '_'.join(str(x) for x in gt_hi), '_'.join(str(x) for x in tp_hi)
            gtr_v = gtrb[i, :, :3]
            if 'gt_f' in b:
                gt_f = b['gt_f'][i]
            else:
                gt_f = self.f
            self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_res', gtr_v, gt_f)

            if self.extended_save:
                gt_v = b['gt'][i, :, :3]
                self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_gt', gt_v, gt_f)

                if 'gt_mask' in b:
                    gt_part_v, gt_part_f = trunc_to_vertex_mask(gt_v, gt_f, b['gt_mask'][i])
                    self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_gtpart', gt_part_v, gt_part_f)
                elif 'gt_noise' in b:  # TODO - Quick hack for gt_noise. Fix this
                    self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_gtnoise', b['gt_noise'][i], None)

                if 'tp_f' in b:
                    tp_f = b['tp_f'][i]
                else:
                    tp_f = self.f
                tp_v = b['tp'][i, :, :3]
                self.save_func(dump_dp / f'gt_{gt_hi}_tp_{tp_hi}_tp', tp_v, tp_f)
