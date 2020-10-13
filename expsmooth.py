from typing import Optional
import torch as pt
from torch import Tensor, BoolTensor, nn
from torch.nn import init, functional as F



class ExpSmooth(nn.Module):
    def __init__(self,
                 cond_size: int,
                 pred_size: int,
                 n_location: int,
                 share_params: bool = False,
                 season: Optional[int] = None,
    ) -> None:
        super(ExpSmooth, self).__init__()
        self.add_season = season is not None
        self.period = season or 0
        self.cond_size = cond_size
        self.pred_size = pred_size
        self._individual_params = nn.Parameter(
            Tensor(
                1 if share_params else n_location,
                4 + self.period + 1,
            )
        )
        init.zeros_(self._individual_params)
        
    def forward(self, query: Tensor):
        es_params = self._individual_params
        init_level = es_params[:, 0]
        level_smoother = es_params[:, 1].sigmoid()
        init_trend = es_params[:, 2]
        trend_smoother = es_params[:, 3].sigmoid()
        
        if self.add_season:
            season_smoother = es_params[:, 4]
            init_seasonalities = es_params[:, -self.period:].sigmoid()
            seasonalities = list(init_seasonalities.unbind(dim=1))
        levels = []
        trends = []
        prev_level = init_level
        prev_trend = init_trend
        
        level_diffs = []
        for t in range(query.size(1)):
            y = query[:, t]
            if self.add_season:
                new_level = y - seasonalities[t]
            else:
                new_level = y
            level = level_smoother * y + (1 - level_smoother) * (prev_level + prev_trend)
            level_diffs.append(pt.abs(level - prev_level))
            levels.append(level)
            trend = trend_smoother * (level - prev_level) + (1-trend_smoother) * prev_trend
            trends.append(trend)
            if self.add_season:
                seasonality = season_smoother * (y - (prev_level + prev_trend)) + (1-season_smoother) * seasonalities[t]
                seasonalities.append(seasonality)
            prev_level = level
            prev_trend = trend
        levels = pt.stack(levels, dim=1)
        trends = pt.stack(trends, dim=1)
        if self.add_season:
            seasonalities = pt.stack(pt.broadcast_tensors(*seasonalities), dim=1)
        level_diffs = pt.stack(level_diffs, dim=1).mean(dim=1)
        
        sm = query - levels
        if self.add_season:
            sm = sm - seasonalities[:, self.period:]
        last_levels = levels[:, (self.cond_size-1):].unsqueeze(dim=2)
        last_trends = trends[:, (self.cond_size-1):].unsqueeze(dim=2)
        
        pr = last_levels + last_trends * (pt.arange(self.pred_size).to(last_trends) + 1)
        if self.add_season:
            season_index = pt.arange(self.period, dtype=pt.long, device=seasonalities.device)
            season_index = pt.cat(
                [season_index]*(self.pred_size//self.period) + [season_index[:(self.pred_size%self.period)]],
                dim=0,
            )
            last_season = pt.stack(
                [
                    pt.index_select(seasonalities, dim=1, index=season_index+start)
                    for start in range(self.cond_size, seasonalities.size(1)-self.period+1)
                ],
                dim=1,
            )
            pr = pr + last_season
        return sm, pr, level_diffs