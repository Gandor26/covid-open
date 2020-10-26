from typing import Optional, Dict
from pathlib import Path
from copy import deepcopy
from tqdm import tqdm
import torch as pt
from torch import Tensor, nn
from torch.optim import Adam


def train(
    train_data: Dict[str, Tensor],
    valid_data: Dict[str, Tensor],
    model: nn.Module,
    optimizer: Adam, 
    model_path: Path,
    n_epochs: int,
    test_size: Optional[int] = None,
    log_step: int = 10,
    patience: int = 10,
) -> None:
    prog_bar = tqdm(total=n_epochs, unit='epoch')
    best_valid = float('inf')
    stop_counter = patience
    for epoch in range(n_epochs):
        prog_bar.update()
        model = model.train()
        loss_train, _ = model(**train_data, test_size=test_size)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
        postfix = {'train_loss': loss_train.item()}
        if (epoch+1) % log_step == 0:
            if valid_data is not None:
                model = model.eval()
                with pt.no_grad():
                    loss_valid, _ = model(**valid_data)
                    loss_valid = loss_valid.item()
                    postfix['valid_loss'] = loss_valid
                    if loss_valid < best_valid:
                        best_valid = loss_valid
                        stop_counter = patience
                    else:
                        stop_counter -= 1
                    if stop_counter == 0:
                        break
            prog_bar.set_postfix(**postfix)
    prog_bar.close()
    pt.save(model.state_dict(), model_path)


def inference(
    data: Dict[str, Tensor],
    model: nn.Module,
    model_path: Path,
):
    model.load_state_dict(pt.load(model_path))
    model = model.eval()
    with pt.no_grad():
        _, pr = model(**data, test_size=0)
        pr = pr.clamp_min_(0.0)
    return pr