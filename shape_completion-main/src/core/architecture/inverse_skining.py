from architecture.decoders import SkinningDecoder
from architecture.f2p import *
import pickle

class F2PEncoderDecoderSkinning(F2PEncoderDecoderBase):
    def _build_model(self):
        data = pickle.load(open(r"R:\MixamoSkinned\index\subject_joint_meta\joints_metadata.pkl", "rb"))

        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = self.encoder_full
        self.decoder = SkinningDecoder(code_size=2 * self.hp.code_size, device=self.hp.dev, joints_parents=data['parents_by_index'], is_global=self.hp.is_global)


    # noinspection PyUnresolvedReferences
    def _init_model(self):
        self.encoder_full.init_weights()
        if self.encoder_part != self.encoder_full:
            self.encoder_part.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=512, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']
        skinning_weights = input_dict['skinning_weights']

        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]

        global_code = torch.cat((part_code, full_code), 1).contiguous()  # [b x (2 * code_size)]
        deformed, global_joint_transformations = self.decoder(full[:,:,:3], global_code, skinning_weights)
        return {'completion_xyz': deformed, 'joint_trans': global_joint_transformations}

class F2FEncoderDecoderSkinning(F2PEncoderDecoderSkinning):

    def forward(self, input_dict):
        ground_truth_full = input_dict['gt']
        template_mesh_full = input_dict['rest_pose_verts']
        skinning_weights = input_dict['skinning_weights']

        ground_truth_code = self.encoder_part(ground_truth_full)  # [b x code_size]
        template_mesh_code = self.encoder_full(template_mesh_full)  # [b x code_size]

        global_code = torch.cat((ground_truth_code, template_mesh_code), 1).contiguous()  # [b x (2 * code_size)]
        deformed, global_joint_transformations = self.decoder(template_mesh_full[:,:,:3], global_code, skinning_weights)
        return {'completion_xyz': deformed, 'joint_trans': global_joint_transformations}



class F2PDisjointEncoderDecoderSkinning(F2PEncoderDecoderSkinning):
    def _build_model(self):
        data = pickle.load(open(r"R:\MixamoSkinned\index\subject_joint_meta\joints_metadata.pkl", "rb"))

        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.decoder = SkinningDecoder(code_size=2 * self.hp.code_size, device=self.hp.dev, joints_parents=data['parents_by_index'], is_global=self.hp.is_global)


class F2FDisjointEncoderDecoderSkinning(F2FEncoderDecoderSkinning):
    def _build_model(self):
        data = pickle.load(open(r"R:\MixamoSkinned\index\subject_joint_meta\joints_metadata.pkl", "rb"))

        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.decoder = SkinningDecoder(code_size=2 * self.hp.code_size, device=self.hp.dev, joints_parents=data['parents_by_index'], is_global=self.hp.is_global)
