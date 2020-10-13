from typing import Optional, Tuple
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F



class XSeriesAttention(nn.Module):
    def __init__(self,
                 cond_size: int,
                 pred_size: int,
                 d_hidden: int,
                 n_rolls: int,
                 n_output: int,
                 d_feats: int = 0,
                 full_attention: bool = False,
                 symmetric: bool = False,
                 cum_value: bool = False,
                 add_autoreg: bool = True,
                 fix_ar_key: bool = True,
    ) -> None:
        super(XSeriesAttention, self).__init__()
        self.cond_size = cond_size
        self.pred_size = pred_size
        self.d_feats = d_feats
        self.d_hidden = d_hidden
        self.n_rolls = n_rolls
        self.n_output = n_output
        self.symmetric = symmetric
        self.full_attention = full_attention
        self.cum_value = cum_value
        self.add_autoreg = add_autoreg
        self.fix_ar_key = fix_ar_key
        
        self.q_weight = nn.Parameter(Tensor(d_hidden, self.d_feats+1, cond_size))
        self.q_bias = nn.Parameter(Tensor(d_hidden))
        if self.symmetric:
            self.k_weight = k_weight
            self.k_bias = k_bias
        else:
            self.k_weight = nn.Parameter(Tensor(d_hidden, self.d_feats+1, cond_size))
            self.k_bias = nn.Parameter(Tensor(d_hidden))
        self.v_weight = nn.Parameter(Tensor(d_hidden, 1, pred_size))
        self.v_bias = nn.Parameter(Tensor(d_hidden))
        if self.add_autoreg:
            self.m_weight = nn.Parameter(Tensor(n_output, 1, cond_size))
            self.m_bias = nn.Parameter(Tensor(n_output))
        else:
            self.register_parameter('m_weight', None)
            self.register_parameter('m_bias', None)
        
        if self.cum_value:
            self.o_weight = nn.Parameter(Tensor(n_output, d_hidden))
            self.o_bias = nn.Parameter(Tensor(n_output))
        else:
            self.o_weight = nn.Parameter(Tensor(n_output*pred_size, d_hidden))
            self.o_bias = nn.Parameter(Tensor(n_output*pred_size))

        self.r_key = nn.Parameter(Tensor(1,1,d_hidden))
        if self.add_autoreg and self.fix_ar_key:
            self.m_key = nn.Parameter(Tensor(1,1,d_hidden))
        else:
            self.register_parameter('m_key', None)
        
        self._reset_parameters()
        
        self.register_buffer('full_score', None, persistent=False)
        self.register_buffer('ref_score', None, persistent=False)
        if not self.full_attention:
            self.register_buffer('selected_t', None, persistent=False)
            self.register_buffer('selected_v', None, persistent=False)
    
    def _reset_parameters(self):
        init.xavier_normal_(self.q_weight)
        init.zeros_(self.q_bias)
        if self.symmetric:
            pass
        else:
            init.xavier_normal_(self.k_weight)
            init.zeros_(self.k_bias)
        if self.cum_value:
            init.ones_(self.v_weight)
            init.xavier_normal_(self.o_weight)
        else:
            orth = Tensor(self.d_hidden, self.pred_size)
            init.orthogonal_(orth)
            with pt.no_grad():
                self.v_weight.copy_(orth.unsqueeze(dim=1))
                self.o_weight.copy_(orth.T.repeat(self.n_output, 1))
        init.zeros_(self.v_bias)
        init.zeros_(self.o_bias)
        if self.add_autoreg:
            init.constant_(self.m_weight, 1.0 / self.cond_size)
            init.zeros_(self.m_bias)
        
        init.xavier_normal_(self.r_key)
        if self.add_autoreg and self.fix_ar_key:
            init.xavier_normal_(self.m_key)
        
    @staticmethod
    def minmax_conv1d(
        x: Tensor,                          # [N, D, T]
        w: nn.Parameter,                    # [H, D+F, K]
        b: Optional[nn.Parameter],          # [H]
        feats: Optional[Tensor] = None,     # [N, T, F]
        offset: Optional[Tensor] = None,    # [N, D, T-K+1]
        scale: Optional[Tensor] = None,     # [N, D, T-K+1]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        ksize = w.size(-1)
        length = x.size(-1)
        # [N, D, T] -> [N, D, K, T-K+1]
        x = pt.stack(
            [x[...,i:i+ksize] for i in range(length-ksize+1)],
            dim=-1,
        )
        if offset is None:
            offset = x.min(dim=2).values
        if scale is None:
            scale = x.max(dim=2).values - offset
            scale[scale.eq(0.0)] = 1.0
        x = x.sub(offset.unsqueeze(dim=2)).div(scale.unsqueeze(dim=2))
        # [N, D, K, T-K+1] -> [N, D+F, K, T-K+1]
        if feats is not None:
            feats = pt.stack(
                [feats[...,i:i+ksize] for i in range(length-ksize+1)],
                dim=-1,
            )
            x = pt.cat([x, feats], dim=1)
        # [N, 1, D+F, K, T-K+1] * [H, D+F, K, 1] -> [N, H, T-K+1]
        y = w.unsqueeze(dim=-1).mul(x.unsqueeze(dim=1)).sum(dim=3).sum(dim=2)
        if b is not None:
            y = y + b.unsqueeze(dim=-1)
        return y, offset, scale

    def fully_soft_attention(self,
                             full_score: Tensor,            # [m, q, k, n]
                             v: Tensor,                     # [n, k, h]
                             r_score: Optional[Tensor],     # [m, q, 1]
                             r: Optional[Tensor],           # [m, q, h]
                             m_score: Optional[Tensor],     # [m, q, 1]
                             m: Optional[Tensor],           # [m, q, h]
    ) -> Tensor:
        # flatten score: [m, q, k, n] -> [m, q, nk]
        full_score = full_score.transpose(-1,-2).contiguous()
        ref_score = full_score.view(*full_score.shape[:-2], -1)
        # flatten value: [n, k, h] -> [nk, h]
        v = v.contiguous().view(-1, self.d_hidden)
        # [nk, o]
        v = F.linear(v, self.o_weight, self.o_bias)
        # append self-prediction to ref list
        # [m, q, nk] + [m, q, 1] + [m, q, 1]
        if r_score is not None:
            ref_score = pt.cat([ref_score, r_score], dim=-1)
        if m_score is not None:
            ref_score = pt.cat([ref_score, m_score], dim=-1)
        ref_score = ref_score.softmax(dim=-1) 
        self.ref_score = ref_score.detach()
        bias = v.new_zeros(())
        # [m, q, o]
        if m_score is not None:
            size = ref_score.size(-1)
            ref_score, m_score = ref_score.split([size-1, 1], dim=-1)
            bias = bias + m_score * m
        if r_score is not None:
            size = ref_score.size(-1)
            ref_score, r_score = ref_score.split([size-1, 1], dim=-1)
            bias = bias + r_score * r
        p = ref_score @ v + bias
        return p
    
    def semi_hard_attention(self,
                            full_score: Tensor,            # [m, q, k, n]
                            v: Tensor,                     # [n, k, h]
                            r_score: Optional[Tensor],     # [m, q, 1]
                            r: Optional[Tensor],           # [m, q, h]
                            m_score: Optional[Tensor],     # [m, q, 1]
                            m: Optional[Tensor],           # [m, q, h]
    ) -> Tensor:
        # take max by key time index
        # [m, q, k, n] -> [m, q, n]
        ref_score, max_idx_per_ref = full_score.max(dim=2)
        self.selected_t = max_idx_per_ref.detach() + self.cond_size
        # [m, q, n] -> [n, mq]
        v_idx = max_idx_per_ref.view(-1, max_idx_per_ref.size(-1)).transpose(0, 1)
        # [n, mq] -> [n, mq, h]
        v_idx = v_idx.unsqueeze(dim=-1).repeat(1,1,self.d_hidden)
        # [n, k, h] -> [n, mq, h] -> [m, q, n, h]
        selected_v = pt.gather(v, 1, v_idx).transpose(0, 1).view(
            *full_score.shape[:2],
            v.shape[0],
            self.d_hidden,
        )
        # [m, q, n, o]
        selected_v = F.linear(selected_v, self.o_weight, self.o_bias)
        self.selected_v = selected_v

        if r_score is not None:
            ref_score = pt.cat([ref_score, r_score], dim=-1)
        if m_score is not None:
            ref_score = pt.cat([ref_score, m_score], dim=-1)
        ref_score = ref_score.softmax(dim=-1) 
        self.ref_score = ref_score
        # [m, q, o]
        bias = v.new_zeros(())
        if m_score is not None:
            size = ref_score.size(-1)
            ref_score, m_score = ref_score.split([size-1, 1], dim=-1)
            bias = bias + m_score * m
        if r_score is not None:
            size = ref_score.size(-1)
            ref_score, r_score = ref_score.split([size-1, 1], dim=-1)
            bias = bias + r_score * r
        p = ref_score.unsqueeze(dim=-1).mul(selected_v).sum(dim=-2)
        return p
    
    def forward(self, 
                query: Tensor, 
                ref: Tensor, 
                local_est: Optional[Tensor] = None,
                query_space_feats: Optional[Tensor] = None,
                ref_space_feats: Optional[Tensor] = None,
                query_time_feats: Optional[Tensor] = None,
                ref_time_feats: Optional[Tensor] = None,
                attn_mask: Optional[BoolTensor] = None,
    ):
        if (query_time_feats is not None) and (ref_time_feats is not None):
            assert query_time_feats.size(2) == ref_time_feats.size(2) == self.d_feats
            assert query_time_feats.size(1) == query.size(1)
            assert ref_time_feats.size(1) == ref.size(1)
            query_time_feats = query_time_feats.transpose(1,2)
            ref_time_feats = ref_time_feats.transpose(1,2)
        else:
            assert self.d_feats == 0
        # [M, Q] -> [M, 1, Q]
        q_input = pt.cumsum(query, dim=1).unsqueeze(dim=1)
        # [M, 1, Q] -> [M, H, Q-C+1] -> [M, Q-C+1, H]
        q, q_offset, q_scale = self.minmax_conv1d(
            x=q_input,
            w=self.q_weight,
            b=self.q_bias,
            feats=query_time_feats,
        )
        q = q.transpose(1,2)
        
        if self.add_autoreg:
            # [M, Q] -> [M, 1, Q+C-1] * [E, 1, C] = [M, E, T]
            m = F.pad(query, (self.cond_size-1, 0)).unsqueeze(dim=1)
            m = F.conv1d(m, self.m_weight, self.m_bias)
            # [M, E, Q] -> [M, E, Q-C+P]
            m = m[..., (self.cond_size-self.pred_size):]
            # [M, E, Q-C+P] -> [M, E, P, Q-C+1]
            m = pt.stack(
                [m[...,i:i+self.pred_size] for i in range(m.size(2)-self.pred_size+1)],
                dim=-1,
            ).contiguous()
            if self.cum_value:
                # [M, E, P, Q-C+1] -> [M, E, Q-C+1]
                m = m.sum(dim=2)
            else:
                # [M, E, P, Q-C+1] -> [N, EP, Q-C+1]
                m = m.view(m.size(0), -1, m.size(3))
            # [M, Q-C+1, E/EP]
            m = m.transpose(1,2)
            m = m / q_scale.transpose(1,2)
        else:
            m = None
        if local_est is not None:
            r = local_est / q_scale.transpose(1,2)
        else:
            r = None
        
        # [N, K] -> [N, 1, K-P]
        k_input = pt.cumsum(ref, dim=1).unsqueeze(dim=1)
        k_input = k_input[..., :-self.pred_size]
        # [N, 1, K-P] -> [N, K-P-C+1, H]
        k, k_offset, k_scale = self.minmax_conv1d(
            x=k_input, 
            w=self.k_weight, 
            b=self.k_bias, 
            feats=ref_time_feats,
        )
        k = k.transpose(1,2)
        
        # [N, K] -> [N, 1, K-C]
        v_input = ref.unsqueeze(dim=1)
        v_input = v_input[..., self.cond_size:]
        # [N, 1, K-C] -> [N, K-P-C+1, H] 
        v, _, _ = self.minmax_conv1d(
            x=v_input, 
            w=self.v_weight,
            b=self.v_bias, 
            offset=pt.zeros_like(k_offset),
            scale=k_scale,
        )
        v = v.transpose(1,2)
        
        full_score = pt.einsum('mqh,nkh->mqkn', q, k)
        if (query_space_feats is not None) and (ref_space_feats is not None):
            space_score = query_space_feats @ ref_space_feats.transpose(0,1)
            space_score = space_score.unsqueeze(dim=1).unsqueeze(dim=2)
            full_score = full_score + space_score
        if attn_mask is not None:
            full_score = full_score.masked_fill(attn_mask, float('-inf'))
        self.full_score = full_score
        
        if local_est is not None:
            r_score = pt.sum(q * self.r_key, dim=-1, keepdim=True)
        else:
            r_score = None
            
        if self.add_autoreg:
            if self.fix_ar_key:
                m_score = pt.sum(q * self.m_key, dim=-1, keepdim=True)
            else:
                m_score = pt.sum(q * q, dim=-1, keepdim=True)
        else:
            m_score = None
            
        preds = []
        for skip in range(self.n_rolls):
            skip_size = self.pred_size * skip
            score = full_score[..., :(full_score.size(2)-skip_size), :]
            value = v[:, skip_size:]
            # [M, Q, E, R]
            if self.full_attention:
                pr = self.fully_soft_attention(
                    score, v, r_score, r, m_score, m,
                )
            else:
                pr = self.semi_hard_attention(
                    score, v, r_score, r, m_score, m,
                )
            if self.cum_value:
                pr = pr.unsqueeze(dim=3)
            else:
                pr = pr.view(*pr.shape[:-1], self.n_output, self.pred_size)
            preds.append(pr)
        pr = pt.cat(preds, dim=3)
        q_scale = q_scale.transpose(1,2).unsqueeze(dim=3)
        pr = pr * q_scale
        return pr