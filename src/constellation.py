import torch
from src.config import CFG

CONSTELLATION = torch.tensor([
    [-0.948683, -0.948683], [-0.948683, -0.316228], [-0.948683,  0.316228], [-0.948683,  0.948683],
    [-0.316228, -0.948683], [-0.316228, -0.316228], [-0.316228,  0.316228], [-0.316228,  0.948683],
    [ 0.316228, -0.948683], [ 0.316228, -0.316228], [ 0.316228,  0.316228], [ 0.316228,  0.948683],
    [ 0.948683, -0.948683], [ 0.948683, -0.316228], [ 0.948683,  0.316228], [ 0.948683,  0.948683],
], device=CFG.DEVICE)

BIT_LABELS = torch.tensor([
    [0,0,0,0], [0,0,0,1], [0,0,1,1], [0,0,1,0],
    [0,1,0,0], [0,1,0,1], [0,1,1,1], [0,1,1,0],
    [1,1,0,0], [1,1,0,1], [1,1,1,1], [1,1,1,0],
    [1,0,0,0], [1,0,0,1], [1,0,1,1], [1,0,1,0]
], device=CFG.DEVICE, dtype=torch.uint8)

def symbols_to_classes(symbols):
    diff = symbols.unsqueeze(1) - CONSTELLATION.unsqueeze(0)
    dist = (diff**2).sum(dim=2)
    return dist.argmin(dim=1)

def ber(tx_cls, rx_cls):
    return (BIT_LABELS[tx_cls] != BIT_LABELS[rx_cls]).float().mean().item()
