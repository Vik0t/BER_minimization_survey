import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.loss import FocalLoss
from src.config import CFG


try:
    from torch.cuda.amp import GradScaler
except ImportError:
    from torch.amp import GradScaler

DEVICE = CFG.DEVICE

class Trainer:
    def __init__(self, model):
        self.model = model.to(DEVICE)
        self.opt = optim.AdamW(
            model.parameters(),
            lr=CFG.LEARNING_RATE,
            weight_decay=CFG.WEIGHT_DECAY
        )
        self.scheduler = ReduceLROnPlateau(
            self.opt,
            mode='min',
            factor=0.5,
            patience=25  # Updated from CFG.patience
        )
        self.crit = FocalLoss(CFG.GAMMA) if CFG.USE_FOCAL_LOSS else torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # For mixed precision training on CUDA
        self.scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None

        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "lr": []
        }

    def run_epoch(self, loader, train=True):
        if train:
            self.model.train()
        else:
            self.model.eval()
            
        total_loss, correct, total = 0, 0, 0

        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            
            if train:
                self.opt.zero_grad(set_to_none=True)
                
                if DEVICE.type == 'cuda':
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        logits = self.model(x)
                        loss = self.crit(logits, y)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.GRAD_CLIP)
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    logits = self.model(x)
                    loss = self.crit(logits, y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CFG.GRAD_CLIP)
                    self.opt.step()
            else:
                with torch.no_grad():
                    if DEVICE.type == 'cuda':
                        with torch.autocast(device_type='cuda', dtype=torch.float16):
                            logits = self.model(x)
                            loss = self.crit(logits, y)
                    else:
                        logits = self.model(x)
                        loss = self.crit(logits, y)

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)

        return total_loss / total, correct / total

    def validate(self, val_loader):
        """Выполняет валидацию модели и возвращает loss и accuracy"""
        val_loss, val_acc = self.run_epoch(val_loader, train=False)
        return val_loss, val_acc

    def step_scheduler(self, val_loss):
        """Обновляет learning rate на основе валидационной ошибки"""
        self.scheduler.step(val_loss)
        self.history["lr"].append(self.opt.param_groups[0]['lr'])
