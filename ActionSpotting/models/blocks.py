import torch
import torch.nn as nn
import torch.nn.functional as F

from mstcn import MSTCN2
from basic import SALayer, SADecoder, SCALayer, SCADecoder, ActionUpdate_GRU, X2Y_map, logit2prob


class Block(nn.Module):
    """
    Base Block class for common functions
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        lines = f"{type(self).__name__}(\n  f:{self.frame_branch},\n  a:{self.action_branch},\n  a2f:{self.a2f_layer if hasattr(self, 'a2f_layer') else None},\n  f2a:{self.f2a_layer if hasattr(self, 'f2a_layer') else None}\n)"
        return lines

    def __repr__(self):
        return str(self)

    def process_feature(self, feature, nclass):
        # use the last several dimension as logit of action classes
        clogit = feature[:, :, -nclass:] # class logit
        feature = feature[:, :, :-nclass] # feature without clogit
        cprob = logit2prob(clogit, dim=-1)  # apply softmax
        feature = torch.cat([feature, cprob], dim=-1)

        return feature, clogit

    def create_fbranch(self, args, in_dim=None, f_inmap=False):
        # if in_dim is None:
        #     in_dim = cfg.f_dim

        # if cfg.f == 'm': # use MSTCN
        #     frame_branch = MSTCN(in_dim, cfg.f_dim, cfg.hid_dim, cfg.f_layers, 
        #                         dropout=cfg.dropout, ln=cfg.f_ln, ngroup=cfg.f_ngp, in_map=f_inmap)
        if args.f == 'm2': # use MSTCN++
            frame_branch = MSTCN2(in_dim = in_dim, num_f_maps = args.f_dim, out_dim = args.hid_dim, num_layers = args.f_layers, 
                                dropout = args.dropout, ln = args.f_ln, ngroup = args.f_ngp, in_map=f_inmap)
        
        return frame_branch

    def create_abranch(self, args):
        if args.a == 'sa': # self-attention layers, for update blocks
            layer = SALayer(q_dim = args.a_dim, nhead = args.a_num_heads, ff_dim = args.a_ffdim, dropout = args.dropout, attn_dropout = args.dropout)
            action_branch = SADecoder(in_dim = args.a_dim, hid_dim = args.a_dim, out_dim = args.hid_dim, decoder_layer = layer, num_layers = args.a_layers, in_map=False)
        elif args.a == 'sca': # self+cross-attention layers, for input blocks when video transcripts are not available
            layer = SCALayer(action_dim = args.a_dim, frame_dim = args.hid_dim, nhead = args.a_num_heads, ff_dim = agrs.a_ffdim, dropout=args.dropout, attn_dropout=args.dropout)
            norm = torch.nn.LayerNorm(args.a_dim)
            action_branch = SCADecoder(in_dim = args.a_dim, hid_dim = args.a_dim, out_dim = args.hid_dim, decoder_layer = layer, num_layers = args.a_layers, norm=norm, in_map=False)
        # elif args.a in ['gru', 'gru_om']: # GRU, for input blocks when video transcripts are available
        #     assert self.cfg.FACT.trans
        #     out_map = (cfg.a == 'gru_om')
        #     action_branch = basic.ActionUpdate_GRU(cfg.a_dim, cfg.a_dim, cfg.hid_dim, cfg.a_layers, dropout=cfg.dropout, out_map=out_map)
        else:
            raise ValueError(args.a)

        return action_branch

    def create_cross_attention(self, args, outdim, kq_pos=True):
        # one layer of cross-attention for cross-branch communication
        layer = X2Y_map(x_dim = args.hid_dim, y_dim = args.hid_dim, y_outdim = outdim, 
            head_dim=cfg.hid_dim, dropout=cfg.dropout, kq_pos=kq_pos)
        
        return layer

    @staticmethod
    def _eval(action_clogit, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)

        action_clogit = action_clogit.squeeze(1)
        a2f_attn = a2f_attn.squeeze(0)
        qtk_cpred = action_clogit.argmax(1) 
        null_cid = action_clogit.shape[-1] - 1
        action_loc = torch.where(qtk_cpred!=null_cid)[0]

        if len(action_loc) == 0:
            return fbranch_prob.argmax(1)

        qtk_prob = torch.softmax(action_clogit[:, :-1], dim=1) # remove logit of null classes
        action_pred = a2f_attn[:, action_loc].argmax(-1)
        action_pred = action_loc[action_pred]
        abranch_prob = qtk_prob[action_pred]

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        return prob.argmax(1)

    @staticmethod
    def _eval_w_transcript(transcript, a2f_attn, frame_clogit, weight):
        fbranch_prob = torch.softmax(frame_clogit.squeeze(1), dim=-1)
        fbranch_prob = fbranch_prob[:, transcript] 

        N = len(transcript)
        a2f_attn = a2f_attn[0, :, :N] # 1, f, a -> f, s'
        abranch_prob = torch.softmax(a2f_attn, dim=-1) # f, s'

        prob = (1-weight) * abranch_prob + weight * fbranch_prob
        pred = prob.argmax(1) # f
        pred = transcript[pred]
        return pred

    def eval(self, transcript=None):
        if not self.cfg.FACT.trans:
            return self._eval(self.action_clogit, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)
        else:
            return self._eval_w_transcript(transcript, self.a2f_attn, self.frame_clogit, self.cfg.FACT.mwt)


class InputBlock(Block):
    def __init__(self, args, in_dim, nclass):
        super().__init__()
        # self.args = args
        self.num_action_classes = nclass

        self.args = args.actionModel.inputBlock

        self.frame_branch = self.create_fbranch(args.actionModel.inputBlock, in_dim, f_inmap=True)
        self.action_branch = self.create_abranch(args.actionModel.inputBlock)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos, action_clogit=None):
        # frame branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.num_action_classes)

        # action branch
        action_feature = self.action_branch(action_feature, frame_feature, pos=frame_pos, query_pos=action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.num_action_classes)
        
        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit

        return frame_feature, action_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        atk_loss = criterion.action_token_loss(match, self.action_clogit)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss(frame_clogit)

        return frame_loss + atk_loss + self.args.loss_sw * smooth_loss
        

class UpdateBlock(Block):

    def __init__(self, args, nclass):
        super().__init__()
        self.args = args
        self.num_action_classes = nclass

        self.args = args.actionModel.updateBlock

        # fbranch
        self.frame_branch = self.create_fbranch(args.actionModel.updateBlock)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(args.actionModel.updateBlock, args.actionModel.updateBlock.a_dim)

        # abranch
        self.action_branch = self.create_abranch(args.actionModel.updateBlock)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(args.actionModel.updateBlock, args.actionModel.updateBlock.f_dim)

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # a->f
        action_feature = self.f2a_layer(frame_feature, action_feature, X_pos=frame_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.num_action_classes)

        # f->a
        frame_feature = self.a2f_layer(action_feature, frame_feature, X_pos=action_pos, Y_pos=frame_pos)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.num_action_classes)

        # save features for loss and evaluation
        self.frame_clogit = frame_clogit 
        self.action_clogit = action_clogit 
        self.f2a_attn = self.f2a_layer.attn[0]
        self.a2f_attn = self.a2f_layer.attn[0]
        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0)
        return frame_feature, action_feature

    def compute_loss(self, criterion: loss.MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1)) 
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss(match, torch.transpose(self.f2a_attn_logit, 1, 2), dim=1)
        a2f_loss = criterion.cross_attn_loss(match, self.a2f_attn_logit, dim=2)

        # temporal smoothing loss
        al = loss.smooth_loss( self.a2f_attn_logit )
        fl = loss.smooth_loss( torch.transpose(self.f2a_attn_logit, 1, 2) )
        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) # f, 1, c -> 1, f, c
        l = loss.smooth_loss( frame_clogit )
        smooth_loss = al + fl + l

        return atk_loss + f2a_loss + a2f_loss + frame_loss + self.args.loss_sw * smooth_loss


class UpdateBlockTDU(Block):
    """
    Update Block with Temporal Downsampling and Upsampling
    """

    def __init__(self, args, nclass):
        super().__init__()
        # self.cfg = cfg
        self.num_action_classes = nclass

        self.args = args.actionModel.upDownBlock

        # fbranch
        self.frame_branch = self.create_fbranch(args.actionModel.upDownBlock)

        # layers for temporal downsample and upsample
        self.seg_update = nn.GRU(args.actionModel.upDownBlock.hid_dim, args.actionModel.upDownBlock.hid_dim//2, args.actionModel.upDownBlock.s_layers, bidirectional=True)
        self.seg_combine = nn.Linear(args.actionModel.upDownBlock.hid_dim, args.actionModel.upDownBlock.hid_dim)

        # f2a: query is action
        self.f2a_layer = self.create_cross_attention(args.actionModel.upDownBlock, args.actionModel.upDownBlock.a_dim)

        # abranch
        self.action_branch = self.create_abranch(args.actionModel.upDownBlock)

        # a2f: query is frame
        self.a2f_layer = self.create_cross_attention(args.actionModel.upDownBlock, args.actionModel.upDownBlock.f_dim)

        # layers for temporal downsample and upsample
        self.sf_merge = nn.Sequential(nn.Linear((args.actionModel.upDownBlock.hid_dim + args.actionModel.upDownBlock.f_dim), args.actionModel.upDownBlock.f_dim), nn.ReLU())


    def temporal_downsample(self, frame_feature):

        # get action segments based on predictions
        cprob = frame_feature[:, :, -self.num_action_classes:]
        _, pred = cprob[:, 0].max(dim=-1)
        pred = utils.to_numpy(pred)
        segs = utils.parse_label(pred)

        tdu = basic.TemporalDownsampleUpsample(segs)
        tdu.to(cprob.device)

        # downsample frames to segments
        seg_feature = tdu.feature_frame2seg(frame_feature)

        # refine segment features
        seg_feature, hidden = self.seg_update(seg_feature)
        seg_feature = torch.relu(seg_feature)
        seg_feature = self.seg_combine(seg_feature) # combine forward and backward features
        seg_feature, seg_clogit = self.process_feature(seg_feature, self.num_action_classes)

        return tdu, seg_feature, seg_clogit

    def temporal_upsample(self, tdu, seg_feature, frame_feature):

        # upsample segments to frames
        s2f = tdu.feature_seg2frame(seg_feature)
        
        # merge with original framewise features to keep low-level details
        frame_feature = self.sf_merge(torch.cat([s2f, frame_feature], dim=-1))

        return frame_feature

    def forward(self, frame_feature, action_feature, frame_pos, action_pos):
        # downsample frame features to segment features
        tdu, seg_feature, seg_clogit = self.temporal_downsample(frame_feature) # seg_feature: S, 1, H

        # f->a
        seg_center = torch.LongTensor([ int( (s.start+s.end)/2 ) for s in tdu.segs ]).to(seg_feature.device)
        seg_pos = frame_pos[seg_center]
        action_feature = self.f2a_layer(seg_feature, action_feature, X_pos=seg_pos, Y_pos=action_pos)

        # a branch
        action_feature = self.action_branch(action_feature, action_pos)
        action_feature, action_clogit = self.process_feature(action_feature, self.num_action_classes)

        # a->f
        seg_feature = self.a2f_layer(action_feature, seg_feature, X_pos=action_pos, Y_pos=seg_pos)

        # upsample segment features to frame features
        frame_feature = self.temporal_upsample(tdu, seg_feature, frame_feature)

        # f branch
        frame_feature = self.frame_branch(frame_feature)
        frame_feature, frame_clogit = self.process_feature(frame_feature, self.num_action_classes)

        # save features for loss and evaluation       
        self.frame_clogit = frame_clogit 
        self.seg_clogit = seg_clogit
        self.tdu = tdu
        self.action_clogit = action_clogit 

        self.f2a_attn_logit = self.f2a_layer.attn_logit[0].unsqueeze(0)
        self.f2a_attn = tdu.attn_seg2frame(self.f2a_layer.attn[0].transpose(2, 1)).transpose(2, 1)
        self.a2f_attn_logit = self.a2f_layer.attn_logit[0].unsqueeze(0) 
        self.a2f_attn = tdu.attn_seg2frame(self.a2f_layer.attn[0])

        return frame_feature, action_feature

    def compute_loss(self, criterion: MatchCriterion, match=None):
        frame_loss = criterion.frame_loss(self.frame_clogit.squeeze(1))
        seg_loss = criterion.frame_loss_tdu(self.seg_clogit, self.tdu)
        atk_loss = criterion.action_token_loss(match, self.action_clogit)
        f2a_loss = criterion.cross_attn_loss_tdu(match, torch.transpose(self.f2a_attn_logit, 1, 2), self.tdu, dim=1)
        a2f_loss = criterion.cross_attn_loss_tdu(match, self.a2f_attn_logit, self.tdu, dim=2)

        frame_clogit = torch.transpose(self.frame_clogit, 0, 1) 
        smooth_loss = loss.smooth_loss( frame_clogit )

        return (frame_loss + seg_loss)/ 2 + atk_loss + f2a_loss + a2f_loss + self.args.loss_sw * smooth_loss