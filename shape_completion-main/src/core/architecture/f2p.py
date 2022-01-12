from lightning.nn import CompletionLightningModel
from test_tube import HyperOptArgumentParser
import torch
from architecture.encoders import PointNetShapeEncoder, DgcnnShapeEncoder
from architecture.decoders import BasicShapeDecoder, LSTMDecoder
from architecture.base import Template, Regressor


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoderBase(CompletionLightningModel):
    def _build_model(self):
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = self.encoder_full
        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 2 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)

    # noinspection PyUnresolvedReferences
    def _init_model(self):
        self.decoder.init_weights()
        self.encoder_full.init_weights()
        if self.encoder_part != self.encoder_full:
            self.encoder_part.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=128, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        # TODO - Generalize this
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = full.size(0)
        nv = full.size(1)

        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]

        y = torch.cat((full, part_code, full_code), 2).contiguous()  # [b x nv x (in_channels + 2*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}


class F2PEncoderDecoderWindowed(CompletionLightningModel):
    def _build_model(self):
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = self.encoder_full
        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 2 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)

    # noinspection PyUnresolvedReferences
    def _init_model(self):
        self.decoder.init_weights()
        self.encoder_full.init_weights()
        if self.encoder_part != self.encoder_full:
            self.encoder_part.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=128, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]

        bs = full.size(0)
        window_size = full.size(1)
        nv = full.size(-2)

        full = full.reshape(bs*window_size,nv,-1)
        part = part.reshape(bs*window_size,nv,-1)

        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]

        part_code = part_code.unsqueeze(1).expand(bs*window_size, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs*window_size, nv, self.hp.code_size)  # [b x nv x code_size]

        y = torch.cat((full, part_code, full_code), 2).contiguous()  # [b x nv x (in_channels + 2*code_size)]
        y = self.decoder(y)
        #y = y.reshape(bs,window_size,nv,-1)
        return {'completion_xyz': y}

class F2PEncoderDecoderEncodingPre(CompletionLightningModel):
    def _build_model(self):
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = self.encoder_full
        self.encoder_pre = self.encoder_full

        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 3 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)

    # noinspection PyUnresolvedReferences
    def _init_model(self):
        self.decoder.init_weights()
        self.encoder_full.init_weights()

        if self.encoder_part != self.encoder_full:
            self.encoder_part.init_weights()
        if self.encoder_pre != self.encoder_full:
            self.encoder_pre.init_weights()
    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=128, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        # TODO - Generalize this
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = full.size(0)
        nv = full.size(1)
        part_shifted_right = torch.cat((part[0, :, :].unsqueeze(0), part[:-1, :, :]), dim=0)
        part_shifted_left = torch.cat((part[1:, :, :], part[-1, :, :].unsqueeze(0)), dim=0)
        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]
        pre_code = self.encoder_pre(part_shifted_right)

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        pre_code = pre_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]

        y = torch.cat((full, part_code, full_code, pre_code),
                      2).contiguous()  # [b x nv x (in_channels + 4*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}

class F2PEncoderDecoderEncodingPost(CompletionLightningModel):
    def _build_model(self):
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = self.encoder_full
        self.encoder_post = self.encoder_full

        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 3 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)

    # noinspection PyUnresolvedReferences
    def _init_model(self):
        self.decoder.init_weights()
        self.encoder_full.init_weights()

        if self.encoder_part != self.encoder_full:
            self.encoder_part.init_weights()
        if self.encoder_post != self.encoder_full:
            self.encoder_post.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=128, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        # TODO - Generalize this
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = full.size(0)
        nv = full.size(1)
        part_shifted_right = torch.cat((part[0, :, :].unsqueeze(0), part[:-1, :, :]), dim=0)
        part_shifted_left = torch.cat((part[1:, :, :], part[-1, :, :].unsqueeze(0)), dim=0)
        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]
        post_code = self.encoder_post(part_shifted_left) # [b x nv x code_size]

        part_shifted_left = torch.cat((part[1:, :, :], part[-1, :, :].unsqueeze(0)), dim=0)

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        post_code = post_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]

        y = torch.cat((full, part_code, full_code, post_code),
                      2).contiguous()  # [b x nv x (in_channels + 4*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}

class F2PEncoderDecoderEncodingPair(CompletionLightningModel):
    def _build_model(self):
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = self.encoder_full
        self.encoder_pre = self.encoder_full
        self.encoder_post = self.encoder_full

        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 4 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)

    # noinspection PyUnresolvedReferences
    def _init_model(self):
        self.decoder.init_weights()
        self.encoder_full.init_weights()

        if self.encoder_part != self.encoder_full:
            self.encoder_part.init_weights()
        if self.encoder_pre != self.encoder_full:
            self.encoder_pre.init_weights()
        if self.encoder_post != self.encoder_full:
            self.encoder_post.init_weights()
    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=128, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        # TODO - Generalize this
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = full.size(0)
        nv = full.size(1)
        part_shifted_right = torch.cat((part[0, :, :].unsqueeze(0), part[:-1, :, :]), dim=0)
        part_shifted_left = torch.cat((part[1:, :, :], part[-1, :, :].unsqueeze(0)), dim=0)
        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]
        pre_code = self.encoder_pre(part_shifted_right)
        post_code = self.encoder_post(part_shifted_left)

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        pre_code = pre_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        post_code = post_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        y = torch.cat((full, part_code, full_code, pre_code, post_code),
                      2).contiguous()  # [b x nv x (in_channels + 4*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}

class F2PEncoderDecoderEncodingPair(CompletionLightningModel):
    def _build_model(self):
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_pre = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_post = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)

        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 4 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)

    # noinspection PyUnresolvedReferences
    def _init_model(self):
        self.decoder.init_weights()
        self.encoder_full.init_weights()
        self.encoder_pre.init_weights()
        self.encoder_post.init_weights()

        if self.encoder_part != self.encoder_full:
            self.encoder_part.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=128, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        # TODO - Generalize this
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = full.size(0)
        nv = full.size(1)
        part_shifted_right = torch.cat((part[0, :, :].unsqueeze(0), part[:-1, :, :]), dim=0)
        part_shifted_left = torch.cat((part[1:, :, :], part[-1, :, :].unsqueeze(0)), dim=0)
        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]
        pre_code = self.encoder_pre(part_shifted_right)
        post_code = self.encoder_post(part_shifted_left)

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        pre_code = pre_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        post_code = post_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        y = torch.cat((full, part_code, full_code, pre_code, post_code),
                      2).contiguous()  # [b x nv x (in_channels + 4*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}


class F2PEncoderDecoderTemporal(F2PEncoderDecoderBase):
    def _build_model(self):
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = self.encoder_full
        self.encoder_pre = self.encoder_full

        self.decoder = LSTMDecoder(code_size=self.hp.in_channels + 3 * self.hp.code_size,
                                   out_channels=self.hp.out_channels, hidden_size=self.hp.decoder_hidden_size,
                                   dropout=self.hp.decoder_dropout, bidirectional=self.hp.decoder_bidirectional,
                                   layer_count=self.hp.decoder_layer_count, n_verts=6890)

    def forward(self, input_dict):
        # TODO - Generalize this
        part = input_dict['gt_part'] if 'gt_part' in input_dict else input_dict['gt_noise']
        full = input_dict['tp']

        # part, full [bs x nv x in_channels]
        bs = full.size(0)
        nv = full.size(1)
        part_shifted_right = torch.cat((part[0, :, :].unsqueeze(0), part[:-1, :, :]), dim=0)
        part_shifted_left = torch.cat((part[1:, :, :], part[-1, :, :].unsqueeze(0)), dim=0)
        part_code = self.encoder_part(part)  # [b x code_size]
        full_code = self.encoder_full(full)  # [b x code_size]
        pre_code = self.encoder_pre(part_shifted_right)

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        pre_code = pre_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        y = torch.cat((full, part_code, full_code, pre_code),
                      2).contiguous()  # [b x nv x (in_channels + 4*code_size)]
        y = self.decoder(y)
        return {'completion_xyz': y}

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=128, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_hidden_size', default=1024, type=int)
        p.add_argument('--decoder_bidirectional', default=False, type=bool)
        p.add_argument('--decoder_dropout', default=0.3, type=int)
        p.add_argument('--decoder_layer_count', default=2, type=int)
        p.add_argument('--decoder_convl', default=5, type=int)

        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p


# ----------------------------------------------------------------------------------------------------------------------
#                                                      Extensions
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoderRealistic(F2PEncoderDecoderBase):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = PointNetShapeEncoder(in_channels=3, code_size=self.hp.code_size)
        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 2 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)


class F2PDisjointEncoderDecoder(F2PEncoderDecoderBase):
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.encoder_full = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.encoder_part = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 2 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)


class F2PDGCNNEncoderDecoder(F2PEncoderDecoderBase):
    def _build_model(self):
        self.encoder_full = DgcnnShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size,
                                              k=20, device=self.hp.dev)
        self.encoder_part = self.encoder_full
        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + 2 * self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)


# ----------------------------------------------------------------------------------------------------------------------
#                                                      BASE
# ----------------------------------------------------------------------------------------------------------------------
class F2PEncoderDecoderTemplateBased(CompletionLightningModel):
    # TODO - Clean this up
    def _build_model(self):
        # Encoder takes a 3D point cloud as an input.
        # Note that a linear layer is applied to the global feature vector
        self.template = Template(self.hp.in_channels, self.hp.dev)
        self.encoder = PointNetShapeEncoder(in_channels=self.hp.in_channels, code_size=self.hp.code_size)
        self.decoder = BasicShapeDecoder(code_size=self.hp.in_channels + self.hp.code_size,
                                         out_channels=self.hp.out_channels, num_convl=self.hp.decoder_convl)
        self.regressor = Regressor(code_size=self.hp.code_size)

    # noinspection PyUnresolvedReferences
    def _init_model(self):
        super()._init_model()
        self.regressor.init_weights()

    @staticmethod
    def add_model_specific_args(parent_parser):
        p = HyperOptArgumentParser(parents=parent_parser, add_help=False, conflict_handler='resolve')
        p.add_argument('--code_size', default=512, type=int)
        p.add_argument('--out_channels', default=3, type=int)
        p.add_argument('--decoder_convl', default=3, type=int)
        if not parent_parser:  # Name clash with parent
            p.add_argument('--in_channels', default=3, type=int)
        return p

    def forward(self, input_dict):
        part = input_dict['gt_part']
        full = input_dict['tp']
        gt = input_dict['gt']

        # part, full, gt [bs x nv x in_channels]
        bs = part.size(0)
        nv = part.size(1)

        part_code = self.encoder(part)  # [b x code_size]
        full_code = self.encoder(full)  # [b x code_size]
        gt_code = self.encoder(gt)  # [b x code_size]
        comp_code = self.regressor(torch.cat((part_code, full_code), 1).contiguous())
        output_dict = {'comp_code': comp_code, 'gt_code': gt_code}

        part_code = part_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        full_code = full_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        comp_code = comp_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]
        gt_code = gt_code.unsqueeze(1).expand(bs, nv, self.hp.code_size)  # [b x nv x code_size]

        # all reconsturction (also completion are achieved by FIXED template deformation)
        template = self.template.get_template().expand(bs, nv, self.hp.in_channels)
        full_rec = self.decoder(
            torch.cat((template, full_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        part_rec = self.decoder(
            torch.cat((template, part_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        gt_rec = self.decoder(
            torch.cat((template, gt_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]
        completion = self.decoder(
            torch.cat((template, comp_code), 2).contiguous())  # decoder input: [b x nv x (in_channels + code_size)]

        output_dict.update({'completion_xyz': completion, 'full_rec': full_rec, 'part_rec': part_rec, 'gt_rec': gt_rec})
        return output_dict
