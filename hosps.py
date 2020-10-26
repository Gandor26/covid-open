from typing import Optional, Tuple, List, Dict
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


class CausalRegressor(nn.Module):
    def __init__(self, 
                 cond_size: int,
                 pred_size: int,
                 n_location: int,
                 n_output: int,
                 s_window: int,
                 d_hidden: int,
    ) -> None:
        super(CausalRegressor, self).__init__()
        self.cond_size = cond_size
        self.pred_size = pred_size
        self.n_location = n_location
        self.s_window = s_window
        self.d_hidden = d_hidden
        self.n_output = n_output
        
        self.temporal_weight = nn.Parameter(Tensor(n_output*pred_size, d_hidden, cond_size))
        self.ma_weight = nn.Parameter(Tensor(d_hidden, 1, s_window))
        self._reset_parameters()
        
    def _reset_parameters(self) -> None:
        init_weight = Tensor(self.pred_size).uniform_()
        init_weight = F.softmax(init_weight, dim=0)
        weights = []
        for day in range(self.cond_size):
            weights.append(init_weight)
            init_weight = init_weight[:-1]
            init_weight = pt.cat([
                1.0-pt.sum(init_weight, dim=0, keepdim=True), 
                init_weight,
            ], dim=0)
        weights = pt.stack(weights, dim=1)
        weights = pt.stack([weights] * self.d_hidden, dim=1)
        weights = pt.cat([weights] * self.n_output, dim=0)
        with pt.no_grad():
            self.temporal_weight.copy_(weights)
        
        init_weight = Tensor(self.d_hidden, 1, self.s_window)
        init.xavier_uniform_(init_weight)
        init_weight = F.softmax(init_weight, dim=2)
        with pt.no_grad():
            self.ma_weight.copy_(init_weight)

    def forward(self,
                new_cases: Tensor,
                total_beds: Tensor,
                senior_pop_rate: Tensor,
    ) -> Tensor:
        cases = new_cases.unsqueeze(dim=1)
        hidden = F.conv1d(cases, self.ma_weight)
        hidden = F.relu(hidden)
        senior_pop_rate = senior_pop_rate.view(-1, 1, 1)
        total_beds = total_beds.view(-1, 1, 1)
        hidden = 2 * total_beds / (1 + pt.exp(-(8*senior_pop_rate/total_beds) * hidden)) - total_beds
        preds = F.conv1d(hidden, self.temporal_weight)
        preds = preds.transpose(-1,-2)
        return preds


class HospModel(GlobalLocalModel):
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
        super(HospModel, self).__init__(
            cond_size, pred_size, n_rolls, d_hidden, n_location,
            quantiles, share_params, var_penalty,
        )
        
        self.smoother = ExpSmooth(
            cond_size=cond_size, 
            pred_size=pred_size*n_rolls, 
            n_location=n_location,
            share_params=share_params,
        )
        self.attention = XSeriesAttention(
            cond_size=cond_size,
            pred_size=pred_size,
            d_hidden=d_hidden,
            d_feats=d_feats,
            n_rolls=n_rolls,
            n_output=self.n_output,
            full_attention=full_attention,
            symmetric=symmetric,
            cum_value=False,
            add_autoreg=add_autoreg,
            fix_ar_key=fix_ar_key,
        )
        
        self.regression = CausalRegressor(
            cond_size=cond_size,
            pred_size=pred_size,
            n_location=n_location,
            n_output=self.n_output,
            s_window=case_window,
            d_hidden=d_hidden,
        )

        self.register_buffer('smoothed', None, persistent=False)
        self.register_buffer('level_diffs', None, persistent=False)
        self.register_buffer('global_pr', None, persistent=False)
        self.register_buffer('local_pr', None, persistent=False)

    def forward(self, 
                hosp_data: Tensor,
                case_data: Tensor,
                total_beds: Tensor,
                senior_pop_rate: Tensor,
                query_time_feats: Optional[Tensor] = None,
                ref_time_feats: Optional[Tensor] = None,
                query_space_feats: Optional[Tensor] = None,
                ref_space_feats: Optional[Tensor] = None,
                test_size: Optional[int] = None,
    ) -> Tuple[Tensor, Tensor, Dict]:
        case_query = case_data
        attn_mask = pt.ones(
            hosp_data.size(1)-self.cond_size+1,
            hosp_data.size(1)-self.cond_size-self.pred_size+1,
            dtype=pt.bool, device=hosp_data.device,
        ).triu()
        attn_mask = attn_mask.view(1, *attn_mask.shape, 1)
        hosp_length = hosp_data.size(1)
        target_index = pt.tensor(
            np.arange(self.cond_size, hosp_length+1).reshape(-1,1)\
            + np.arange(self.pred_size * self.n_rolls).reshape(1,-1),
            dtype=pt.long, device=hosp_data.device
        )
        target_mask = target_index >= hosp_length
        target_index = pt.where(target_mask, pt.zeros_like(target_index)-1, target_index)
        target = hosp_data[:, target_index]
        
        sm, local_pr, level_diffs = self.smoother(hosp_data)
        hosp_query = hosp_data
        hosp_ref = hosp_data
        if test_size is not None:
            hosp_query = hosp_data[:, -(test_size+self.cond_size):]
            case_query = case_data[:, -(test_size+self.cond_size+self.regression.s_window-1):]
            attn_mask = attn_mask[:, -(test_size+1):]
            local_pr = local_pr[:, -(test_size+1):]
            target = target[:, -(test_size+1):]
            target_mask = target_mask[-(test_size+1):]
    
        local_est = self.regression(
            new_cases=case_query,
            total_beds=total_beds,
            senior_pop_rate=senior_pop_rate,
        )
        
        global_pr = self.attention(
            query=hosp_query,
            ref=hosp_ref,
            local_est=local_est,
            query_space_feats=query_space_feats,
            ref_space_feats=ref_space_feats,
            query_time_feats=query_time_feats,
            ref_time_feats=ref_time_feats,
            attn_mask=attn_mask,
        )

        pr = self.tradeoff * pt.clamp_min(global_pr, 0.0) + (1 - self.tradeoff) * pt.clamp_min(local_pr, 0.0).unsqueeze(dim=2)
        # pr = pt.clamp_min(global_pr + local_pr.unsqueeze(dim=2), 0.0)
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
    case_window: int,
    device: int = -1,
    test_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    hosps = load_hospitalized_data(
        start_date=pd.to_datetime(start_date)-pd.Timedelta(1, unit='d'),
        end_date=end_date
    )
    hosps = hosps.bfill()
    hosps = hosps.diff(1).iloc[1:]
    hosps = hosps.rename(columns=state2abbr)
    
    cases = load_cdc_truth(
        death=False, cumulative=False,
        start_date=pd.to_datetime(start_date)-pd.Timedelta(case_window-1, unit='d'),
        end_date=end_date,
    ).rename(columns=state2abbr).loc[:, hosps.columns]
    beds = load_bed_and_population_data().loc[hosps.columns]
    normed_beds = (beds - beds.mean(axis=0)) / beds.std(axis=0)
    normed_beds['65+%'] = beds['population_65'] / beds['adult_population']
    # mobs = load_mobility_data().rename(columns=state2abbr)
    
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
            hosps.columns, 
            [
                'adult_population', 
                'population_65', 
                'density',
            ]
        ].values,
        feats.loc[hosps.columns].values,
    ]
    ref_space_feats = query_space_feats.copy()
    
    device = pt.device('cpu') if device < 0 else pt.device(f'cuda:{device}')
    data = {
        'hosp_data': pt.tensor(hosps.values.T, dtype=pt.float, device=device),
        'case_data': pt.tensor(cases.values.T, dtype=pt.float, device=device),
        'total_beds': pt.tensor(beds.loc[:, 'total_hospital_beds'].values, dtype=pt.float, device=device),
        'senior_pop_rate': pt.tensor(normed_beds.loc[hosps.columns, '65+%'].values, dtype=pt.float, device=device),
        'query_space_feats': pt.tensor(query_space_feats, dtype=pt.float, device=device),
        'ref_space_feats': pt.tensor(ref_space_feats, dtype=pt.float, device=device),
    }
    if test_size is not None:
        train_data = deepcopy(data)
        train_data['hosp_data'] = train_data['hosp_data'][:, :-test_size]
        train_data['case_data'] = train_data['case_data'][:, :-test_size]
        valid_data = data
    else:
        train_data = data
        valid_data = None
    return train_data, valid_data