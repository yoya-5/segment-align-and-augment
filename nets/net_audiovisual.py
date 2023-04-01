import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingNCELoss(nn.Module):

    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingNCELoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return -torch.mean(torch.log(torch.sum(true_dist * pred, dim=self.dim)))


class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=512, nhead=1, dim_feedforward=512, dropout=0.1):
        super(Encoder, self).__init__()
        self.layers = []
        if encoder_layer == 'HANLayer':
            for i in range(num_layers):
                self.layers.append(HANLayer(d_model=d_model, nhead=nhead,
                                            dim_feedforward=dim_feedforward, dropout=dropout))
        else:
            raise ValueError('wrong encoder layer')
        self.layers = nn.ModuleList(self.layers)
        self.num_layers = num_layers
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm = norm

    def forward(self, src_a, src_v, mask=None, src_key_padding_mask=None, with_ca=True):
        output_a = src_a
        output_v = src_v

        for i in range(self.num_layers):
            output_a = self.layers[i](src_a, src_v, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            output_v = self.layers[i](src_v, src_a, src_mask=mask,
                                      src_key_padding_mask=src_key_padding_mask, with_ca=with_ca)
            src_a = output_a
            src_v = output_v

        if self.norm:
            output_a = self.norm1(output_a)
            output_v = self.norm2(output_v)

        return output_a, output_v


class HANLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super(HANLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cm_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src_q, src_v, src_mask=None, src_key_padding_mask=None, with_ca=True):
        """Pass the input through the encoder layer.

        Args:
            src_q: the sequence to the encoder layer (required).
            src_v: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            with_ca: whether to use audio-visual cross-attention
        Shape:
            see the docs in Transformer class.
        """
        src_q = src_q.permute(1, 0, 2)
        src_v = src_v.permute(1, 0, 2)

        if with_ca:
            src1 = self.cm_attn(src_q, src_v, src_v, attn_mask=src_mask,
                                key_padding_mask=src_key_padding_mask)[0]
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

            src_q = src_q + self.dropout11(src1) + self.dropout12(src2)
            src_q = self.norm1(src_q)
        else:
            src2 = self.self_attn(src_q, src_q, src_q, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)[0]

            src_q = src_q + self.dropout12(src2)
            src_q = self.norm1(src_q)

        src_q = src_q + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(src_q)))))
        src_q = self.norm2(src_q)
        return src_q.permute(1, 0, 2)


class MMIL_Net(nn.Module):

    def __init__(self, num_layers=1, temperature=0.2, att_dropout=0.1, cls_dropout=0.5):
        super(MMIL_Net, self).__init__()

        self.fc_prob = nn.Linear(512, 25)
        self.fc_frame_att = nn.Linear(512, 25)
        self.fc_av_att = nn.Linear(512, 25)
        self.fc_a = nn.Linear(128, 512)
        self.fc_v = nn.Linear(2048, 512)
        self.fc_st = nn.Linear(512, 512)
        self.fc_fusion = nn.Linear(1024, 512)
        self.hat_encoder = Encoder('HANLayer', num_layers, norm=None, d_model=512,
                                   nhead=1, dim_feedforward=512, dropout=att_dropout)
        self.alpha = 0.2

        self.audio_gated = nn.Sequential(
            nn.Linear(512, 1),

            nn.Sigmoid()
        )
        self.video_gated = nn.Sequential(
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
        self.temp = temperature
        if cls_dropout != 0:
            self.dropout = nn.Dropout(p=cls_dropout)
        else:
            self.dropout = None

    def forward(self, audio, visual, visual_st, with_ca=True):
        b, t, d = visual_st.size()
        x1 = self.fc_a(audio)
        # 2d and 3d visual feature fusion
        vid_s = self.fc_v(visual).permute(0, 2, 1).unsqueeze(-1)
        vid_s = F.avg_pool2d(vid_s, (8, 1)).squeeze(-1).permute(0, 2, 1)
        vid_st = self.fc_st(visual_st)
        x2 = torch.cat((vid_s, vid_st), dim=-1)
        x2 = self.fc_fusion(x2)

        # HAN
        x1, x2 = self.hat_encoder(x1, x2, with_ca=False)


        #先对齐特征，确保每个时刻两个模态特征都是相关性最高的
        #x1, x2 = self.hat_encoder(audio_query_output, video_query_output)
        xx2_after = x2
        xx1_after = x1
        # # ################step1###############
        # w=torch.rand(b,10,10).cuda()
        # for i in range(10):
        #     for j in range(10):
        #         w[:, i, j] = torch.sum(xx1_after[:, j, :].clone()*xx2_after[:, i, :].clone(),dim=1)
        #
        #
        # w=torch.softmax(w,dim=2)
        # _, max_indices = torch.max(w.clone(), dim=2, keepdim=True)
        #
        # w=w.clone().zero_()
        # w=w.clone().scatter_(2, max_indices, 1)
        #
        #
        # temp = xx1_after.clone()  # 复制 xx1_after，准备做后续计算
        # for i in range(10):
        #         xx1_after[:, i, :] = (w[:, i, 0].unsqueeze(1) * temp[:, 0, :].clone() +
        #                               w[:, i, 1].unsqueeze(1) * temp[:, 1, :].clone() +
        #                               w[:, i, 2].unsqueeze(1) * temp[:, 2, :].clone() +
        #                               w[:, i, 3].unsqueeze(1) * temp[:, 3, :].clone() +
        #                               w[:, i, 4].unsqueeze(1) * temp[:,  4, :].clone() +
        #                               w[:, i, 5].unsqueeze(1) * temp[:,  5, :].clone() +
        #                               w[:, i, 6].unsqueeze(1) * temp[:, 6, :].clone() +
        #                               w[:, i, 7].unsqueeze(1) * temp[:,  7, :].clone() +
        #                               w[:, i, 8].unsqueeze(1) * temp[:,  8, :].clone() +
        #                               w[:, i, 9].unsqueeze(1) * temp[:,  9, :].clone())
        # video_key_value_feature = xx2_after
        # audio_key_value_feature = xx1_after
        # video_query_output = xx2_after
        # audio_query_output = xx1_after
        # audio_gate = self.audio_gated(video_key_value_feature)
        # video_gate = self.video_gated(audio_key_value_feature)
        #
        # video_query_output = (1 - self.alpha) * video_query_output + audio_gate * video_query_output * self.alpha
        # audio_query_output = (1 - self.alpha) * audio_query_output + video_gate * audio_query_output * self.alpha
        #
        # xx1_after=audio_query_output
        # xx2_after=video_query_output
        #
        # xx2_after = F.normalize(xx2_after, p=2, dim=-1)
        # xx1_after = F.normalize(xx1_after, p=2, dim=-1)
        ################step1###############

        ######step3###############
        ##把对齐后的特征进行加强
        video_key_value_feature = xx2_after
        audio_key_value_feature = xx1_after
        video_query_output = xx2_after
        audio_query_output = xx1_after
        audio_gate = self.audio_gated(video_key_value_feature)
        video_gate = self.video_gated(audio_key_value_feature)

        video_query_output = (1 - self.alpha) * video_query_output + audio_gate * video_query_output * self.alpha
        audio_query_output = (1 - self.alpha) * audio_query_output + video_gate * audio_query_output * self.alpha
        # video_query_output = (video_query_output + audio_gate * video_query_output * self.alpha)/1.05
        # audio_query_output = (audio_query_output + video_gate * audio_query_output * self.alpha)/1.05
        x1,x2=self.hat_encoder(audio_query_output, video_query_output, with_ca=True)
        # x1 = audio_query_output
        # x2 = video_query_output
        xx2_after = x2
        xx1_after = x1

        xx2_after = F.normalize(xx2_after, p=2, dim=-1)
        xx1_after = F.normalize(xx1_after, p=2, dim=-1)
        ######step3###############



        sims_after = xx1_after.bmm(xx2_after.permute(0, 2, 1)).squeeze(1) / self.temp
        sims_after = sims_after.reshape(-1, 10)
        mask_after = torch.zeros(b, 10)
        mask_after = mask_after.long()
        for i in range(10):
            mask_after[:, i] = i
        mask_after = mask_after.cuda()
        mask_after = mask_after.reshape(-1)

        # prediction
        if self.dropout is not None:
            x2 = self.dropout(x2)
        x = torch.cat([x1.unsqueeze(-2), x2.unsqueeze(-2)], dim=-2)
        frame_prob = torch.sigmoid(self.fc_prob(x))
        # attentive MMIL pooling
        frame_att = torch.softmax(self.fc_frame_att(x), dim=1)
        av_att = torch.softmax(self.fc_av_att(x), dim=2)
        temporal_prob = frame_att * frame_prob
        global_prob = (temporal_prob * av_att).sum(dim=2).sum(dim=1)
        # frame-wise probability
        a_prob = temporal_prob[:, :, 0, :].sum(dim=1)
        v_prob = temporal_prob[:, :, 1, :].sum(dim=1)

        return global_prob, a_prob, v_prob, frame_prob, sims_after, mask_after
