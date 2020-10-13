from typing import Optional, List, Dict, Tuple
from copy import deepcopy
import pandas as pd
import numpy as np
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F
from torch.optim import Adam

from data import *
from expsmooth import ExpSmooth
from attention import XSeriesAttention
from base import GlobalLocalModel


class CaseModel(GlobalLocalModel):
    def __init__(self,
                 cond_size: int,
                 pred_size: int,
                 n_rolls: int,
                 d_hidden: int,
                 n_location: int,
                 case_window: int,
                 quantiles: List[int],
                 d_feats: int = 0,
                 share_params: bool = False,
                 full_attention: bool = False,
                 symmetric: bool = False,
                 add_autoreg: bool = True,
                 fix_ar_key: bool = True,
                 var_penalty: float = 1.0,
    ) -> None:
        super(CaseModel, self).__init__(
            cond_size, pred_size, n_rolls, d_hidden, n_location,
            quantiles, share_params, var_penalty,
        )
        
        self.smoother = ExpSmooth(
            cond_size=cond_size//pred_size, 
            pred_size=n_rolls, 
            n_location=n_location,
            share_params=share_params,
        )
        
        self.attention = XSeriesAttention(
            cond_size=cond_size,
            pred_size=pred_size,
            d_hidden=d_hidden,
            n_rolls=n_rolls,
            n_output=self.n_output,
            d_feats=d_feats,
            full_attention=full_attention,
            symmetric=symmetric,
            cum_value=True,
            add_autoreg=add_autoreg,
            fix_ar_key=fix_ar_key,
        )
        
        self.register_buffer('smoothed', None, persistent=False)
        self.register_buffer('level_diffs', None, persistent=False)
        self.register_buffer('global_pr', None, persistent=False)
        self.register_buffer('local_pr', None, persistent=False)

    def forward(self,
                case_data: Tensor,
                query_time_feats: Optional[Tensor] = None,
                ref_time_feats: Optional[Tensor] = None,
                query_space_feats: Optional[Tensor] = None,
                ref_space_feats: Optional[Tensor] = None,
                test_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        case_query = case_data
        case_ref = case_data
        attn_mask = pt.ones(
            case_query.size(1)-self.cond_size+1,
            case_ref.size(1)-self.cond_size-self.pred_size+1,
            dtype=pt.bool, device=case_query.device,
        ).triu()
        attn_mask = attn_mask.view(1, *attn_mask.shape, 1)
        case_length = case_data.size(1)
        target_index = pt.tensor(
            np.arange(self.cond_size, case_length+1).reshape(-1,1)\
            + np.arange(self.pred_size * self.n_rolls).reshape(1,-1),
            dtype=pt.long, device=case_data.device,
        )
        target_mask = target_index >= case_length
        target_index = pt.where(target_mask, pt.zeros_like(target_index)-1, target_index)
        target = case_data[:, target_index]
        target = target.view(
            *target.shape[:-1], 
            self.n_rolls, 
            self.pred_size,
        ).sum(dim=3)
        target_mask = target_mask.view(
            *target_mask.shape[:-1], 
            self.n_rolls, 
            self.pred_size,
        ).any(dim=2)

        pad = case_data.size(1) % self.pred_size
        if pad > 0:
            smooth_input = pt.cat([
                case_data.new_zeros(case_data.size(0), self.pred_size - pad),
                case_data,
            ], dim=1)
        else:
            smooth_input = case_data
        smooth_input = smooth_input.view(
            *smooth_input.shape[:-1],
            -1,
            self.pred_size,
        ).sum(dim=-1)
        sm, local_pr, level_diffs = self.smoother(smooth_input)
        sm = pt.repeat_interleave(
            sm, 
            self.pred_size, 
            dim=1,
        )[:, -case_data.size(1):]
        local_pr = pt.repeat_interleave(
            local_pr,
            self.pred_size, 
            dim=1,
        )[:, -(case_data.size(1)-self.cond_size+1):]
        if test_size is not None:
            case_query = case_data[:, -(test_size+self.cond_size):]
            attn_mask = attn_mask[:, -(test_size+1):]
            local_pr = local_pr[:, -(test_size+1):]
            target = target[:, -(test_size+1):]
            target_mask = target_mask[-(test_size+1):]
    
        global_pr = self.attention(
            query=case_query,
            ref=case_ref,
            local_est=None,
            query_space_feats=query_space_feats,
            ref_space_feats=ref_space_feats,
            query_time_feats=query_time_feats,
            ref_time_feats=ref_time_feats,
            attn_mask=attn_mask,
        )

        pr = self.tradeoff * pt.clamp_min(global_pr, 0.0) + (1-self.tradeoff) * pt.clamp_min(local_pr, 0.0).unsqueeze(dim=2)
        loss = sum(
            self.quantile_error(p, target, q) 
            for q, p in zip(
                self.quantiles,
                pr.unbind(dim=2),
            )
        )
        loss = loss.masked_fill(target_mask, 0.0).mean()
        loss = loss + level_diffs.mean() * self.var_penalty
        self.smoothed = sm.detach()
        self.level_diffs = level_diffs.detach()
        self.global_pr = global_pr.detach()
        self.local_pr = local_pr.detach()
        return loss, pr


def load_data(
    start_date: str,
    end_date: str,
    device: int = -1,
    test_size: Optional[int] = None,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    cases = load_cdc_truth(
        death=False, cumulative=False,
        start_date=start_date,
        end_date=end_date,
    ).rename(columns=state2abbr)
    
    beds = load_bed_and_population_data().loc[cases.columns]
    normed_beds = (beds - beds.mean(axis=0)) / beds.std(axis=0)
    population = load_bed_and_population_data().loc[cases.columns]
    normed_population = (population - population.mean(axis=0)) / population.std(axis=0)
    normed_beds['65+%'] = population['population_65'] / population['adult_population']
    mobs = load_mobility_data().rename(columns=state2abbr)
    mobs = mobs - mobs.min(axis=0)
    
    feats = load_census_embedding().loc[:, [
        'ANC1P_252', 
        'ANC1P_290', 
        'ANC2P_252', 
        'ANC2P_290', 
        'HICOV_1',
        'HICOV_2',
        'LANP_1200'
    ]].reindex(beds.index).fillna(0.0)

    query_space_feats = np.c_[
        normed_beds.loc[
            cases.columns,
            [
                'adult_population', 
                'population_65', 
                'density',
            ]
        ].values,
        feats.loc[cases.columns].values,
    ]
    ref_space_feats = query_space_feats.copy()
    
    device = pt.device('cpu') if device < 0 else pt.device(f'cuda:{device}')
    data = {
        'case_data': pt.tensor(cases.values.T, dtype=pt.float, device=device),
        'query_space_feats': pt.tensor(query_space_feats, dtype=pt.float, device=device),
        'ref_space_feats': pt.tensor(ref_space_feats, dtype=pt.float, device=device),
    }
    if test_size is not None:
        train_data = deepcopy(data)
        train_data['case_data'] = train_data['case_data'][:, :-test_size]
        valid_data = data
    else:
        train_data = data
        valid_data = None
    return train_data, valid_data
