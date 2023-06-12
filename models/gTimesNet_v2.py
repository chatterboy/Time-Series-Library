import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft
from layers.Embed import DataEmbedding
from layers.Conv_Blocks import Inception_Block_V1


class GroupModule(nn.Module):
    def __init__(self, configs):
        super(GroupModule, self).__init__()
        self.d_model = configs.d_model
        self.num_groups = configs.num_groups
        if self.d_model % self.num_groups != 0:
            raise ValueError("Expected {} is divisible by {}".format(self.d_model, self.num_groups))
        self.d_submodel = int(self.d_model / self.num_groups)
        self.block = TimesBlock(configs.seq_len, configs.pred_len, configs.top_k,
                                self.d_submodel, self.d_submodel, configs.num_kernels)

    def forward(self, x):
        # [B, T, F]
        outs = []
        for i in range(self.num_groups):
            outs.append(self.block(x[:, :, i * self.d_submodel:(i + 1) * self.d_submodel]))
        outs = torch.cat(outs, dim=-1)
        return outs


class GroupTimesBlock(nn.Module):
    def __init__(self, configs):
        super(GroupTimesBlock, self).__init__()
        self.gm = GroupModule(configs)
        self.dropout = nn.Dropout(configs.dropout)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.linear1 = nn.Linear(configs.d_model, configs.d_ff)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(configs.dropout)
        self.linear2 = nn.Linear(configs.d_ff, configs.d_model)
        self.dropout2 = nn.Dropout(configs.dropout)
        self.norm2 = nn.LayerNorm(configs.d_model)

    def forward(self, x):
        x = self.norm1(x + self.gm(x))
        x = self.norm2(x + self.ffn(x))
        return x

    def ffn(self, x):
        x = self.dropout1(self.act1(self.linear1(x)))
        x = self.dropout2(self.linear2(x))
        return x


def FFT_for_Period(x, k=2):
    # [B, T, C]
    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)  # 여기서, batch와 feature에 대해 주파수 평균을 계산해버리네..
    frequency_list[0] = 0  # 이 부분은 휴리스틱한 부분인 것 같음. '0'에 위치한 것이 항상 크다는 것을 관찰하고 이 부분을 배제하는?
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, k, d_model, d_ff, num_kernels):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)  # period_list: [k], period_weight: [B, k] => 왜 이렇지?

        res = []
        for i in range(self.k):
            period = period_list[i]
            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x
            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()  # [16, 32, 27, 2] => [B, C, ]
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :(self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        # res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.model = nn.ModuleList([GroupTimesBlock(configs)
                                    for _ in range(configs.e_layers)])
        # self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
        #                                    configs.dropout)
        self.enc_embedding = DataEmbedding(1, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers
        self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.predict_linear = nn.Linear(
                self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(
                configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            # self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            self.projection = nn.Linear(configs.d_model, 1, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # x_enc: [16, 36, 1] => [B, , ]
        # x_mark_enc: None
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]  # 여기서, 채널을 늘리고
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1)  # align temporal dimension  # 여기서 길이를 늘림
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
        # porject back
        dec_out = self.projection(enc_out)  # [B,L,D]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization from Non-stationary Transformer
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) /
                           torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)
        # porject back
        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        return dec_out

    def anomaly_detection(self, x_enc):  # [B,T,V]
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev  # [B,T,V]

        x_enc = x_enc.permute(0, 2, 1)  # [B,V,T]
        x_enc = torch.unsqueeze(x_enc, dim=-1)  # [B,V,T,1]
        u_enc = torch.reshape(x_enc, (x_enc.size(0) * x_enc.size(1), x_enc.size(2), x_enc.size(3)))  # [B*V,T,1]

        # embedding
        enc_out = self.enc_embedding(u_enc, None)  # [B*V,T,F]

        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)  # [B*V,T,F]

        # porject back
        dec_out = self.projection(enc_out)  # [B*V,T,1]

        dec_out = torch.reshape(dec_out, (x_enc.size(0), x_enc.size(1), x_enc.size(2), x_enc.size(3)))  # [B,V,T,1]
        dec_out = torch.squeeze(dec_out, dim=-1)  # [B,V,T]
        dec_out = dec_out.permute(0, 2, 1)  # [B,T,V]

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * \
                  (stdev[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))
        dec_out = dec_out + \
                  (means[:, 0, :].unsqueeze(1).repeat(
                      1, self.pred_len + self.seq_len, 1))

        return dec_out  # [B,T,V]

    def classification(self, x_enc, x_mark_enc):
        # embedding
        enc_out = self.enc_embedding(x_enc, None)  # [B,T,C]
        # TimesNet
        for i in range(self.layer):
            enc_out = self.model[i](enc_out)

        # Output
        # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.act(enc_out)
        output = self.dropout(output)
        # zero-out padding embeddings
        output = output * x_mark_enc.unsqueeze(-1)
        # (batch_size, seq_length * d_model)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(
                x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
