import torch
from torch.utils.data import DataLoader, Subset
from src.dataset import load_all_files, WindowedSymbolDataset
from src.constellation import symbols_to_classes, ber
from models import build_model
from src.trainer import Trainer
from src.config import CFG

def prepare_datasets():
    print("="*80)
    print("Создаем датасет")
    print("="*80)

    TRAIN_FILE_INDICES = CFG.train_files  # [1, 2, 3, 4, 5, 6, 7, 8, 9]
    TEST_FILE_INDICES = CFG.test_files  # Только файл 10 для чистого теста

    print("\n[1/4] Загружаем тренировочные и валидационные файлы (1-9)...")
    tx_train, rx_train = load_all_files(TRAIN_FILE_INDICES)

    print(f"[2/4] Создаем окна (k={CFG.context_k} → {2*CFG.context_k+1} символов)...")
    if len(tx_train) > 10000:
        tx_classes_list = [
            symbols_to_classes(s.to(CFG.DEVICE)).cpu()
            for s in tx_train.split(10000)
        ]
        tx_classes = torch.cat(tx_classes_list)
    else:
        tx_classes = symbols_to_classes(tx_train.to(CFG.DEVICE)).cpu()

    # Создаём датасет с аугментацией
    dataset = WindowedSymbolDataset(
        rx_train,
        tx_classes,
        k=CFG.context_k,
        normalize=True,
        noise_std=CFG.NOISE_STD if CFG.USE_NOISE_AUG else 0.0
    )

    print("[3/4] Перетосовывем и разделяем окна на трейн/валидацию (80/20)...")
    indices = torch.randperm(len(dataset)).tolist()
    split_idx = int(0.8 * len(dataset))

    train_subset = Subset(dataset, indices[:split_idx])
    val_subset = Subset(dataset, indices[split_idx:])

    train_loader = DataLoader(train_subset, batch_size=CFG.batch_size, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    dataset.set_training(True)

    print("[4/4] Загружаем тестовый файл (модель его не видит)...")
    tx_test, rx_test = load_all_files(TEST_FILE_INDICES)
    tx_test_classes = symbols_to_classes(tx_test.to(CFG.DEVICE)).cpu()

    test_dataset = WindowedSymbolDataset(rx_test, tx_test_classes, k=CFG.context_k, normalize=False)

    # Применяем статистику нормализации из train
    if dataset.mean is not None and dataset.std is not None:
        test_dataset.X = (test_dataset.X - dataset.mean) / dataset.std

    test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    print("\n" + "="*80)
    print(f"Финальное разделение:")
    print(f"  Train windows: {len(train_subset):,} (80%)")
    print(f"  Val windows:   {len(val_subset):,} (20%)")
    print(f"  Test windows:  {len(test_dataset):,} (file 10)")
    print(f"  Input dim:     {(2*CFG.context_k+1)*2} features (k={CFG.context_k})")
    print(f"  Total symbols: {len(tx_train):,} (files 1-9) + {len(tx_test):,} (file 10)")
    print("="*80 + "\n")

    return train_loader, val_loader, test_loader, dataset.mean, dataset.std

def train_equalizer(train_loader: DataLoader, val_loader: DataLoader):
    model = build_model(CFG.model_name)
    trainer = Trainer(model)
    
    best, patience = 1e9, 0
    best_state = None
    
    for e in range(CFG.epochs):
        tr_loss, _ = trainer.run_epoch(train_loader, True)
        va_loss, va_acc = trainer.validate(val_loader)
        
        trainer.history["train_loss"].append(tr_loss)
        trainer.history["val_loss"].append(va_loss)
        trainer.history["val_acc"].append(va_acc)
        
        # Обновляем learning rate
        trainer.step_scheduler(va_loss)
        
        if va_loss < best:
            best, patience = va_loss, 0
            best_state = model.state_dict()
        else:
            patience += 1
            
        if patience > CFG.patience:
            print(f"Early stopping at epoch {e}")
            break
            
        if (e + 1) % 20 == 0 or e == 0 or e == CFG.epochs - 1:
            print(f"{e:03d} | train {tr_loss:.4f} | val {va_loss:.4f} | acc {va_acc:.3f} | lr {trainer.history['lr'][-1]:.2e}")
    
    # Загружаем лучшую модель
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"Loaded best model with val loss: {best:.6f}")
    
    return model, trainer

def comprehensive_evaluation(model, test_loader, tx_symbols, rx_symbols):
    model.eval()
    baseline_ber = ber(
        symbols_to_classes(tx_symbols.to(CFG.DEVICE)),
        symbols_to_classes(rx_symbols.to(CFG.DEVICE))
    )

    all_preds, all_labels = [], []
    for inputs, labels in test_loader:
        inputs = inputs.to(CFG.DEVICE)
        if CFG.DEVICE.type == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(inputs)
        else:
            logits = model(inputs)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels)

    pred_classes = torch.cat(all_preds)
    true_classes = torch.cat(all_labels)

    equalized_ber = ber(true_classes, pred_classes)
    ser = (pred_classes != true_classes).float().mean().item()

    return {
        'baseline_ber': baseline_ber,
        'equalized_ber': equalized_ber,
        'ser': ser,
        'improvement_abs': baseline_ber - equalized_ber,
        'improvement_rel': (1 - equalized_ber / baseline_ber) * 100,
        'improvement_db': 10 * torch.log10(torch.tensor(baseline_ber / equalized_ber)).item() if equalized_ber > 0 else float('inf')
    }

def main():
    # Print configuration information
    print(f"✓ Device: {CFG.DEVICE}")
    print(f"✓ Context window: ±{CFG.CONTEXT_K} symbols (total {2*CFG.CONTEXT_K+1})")
    print(f"✓ Architecture: {'1D-CNN' if CFG.USE_CNN else 'Deep MLP'}")
    print(f"✓ Focal Loss: {'ENABLED' if CFG.USE_FOCAL_LOSS else 'disabled'}")
    
    # Подготовка датасетов
    train_loader, val_loader, test_loader, mean, std = prepare_datasets()

    # Загружаем полные символы для теста (файл 10)
    tx_test_full, rx_test_full = load_all_files([10])

    # Обучение
    model, trainer = train_equalizer(train_loader, val_loader)

    # Оценка
    results = comprehensive_evaluation(model, test_loader, tx_test_full, rx_test_full)

    # Вывод результатов
    print("\n" + "="*80)
    print("FINAL EVALUATION (file 10 - completely unseen)")
    print("="*80)
    print(f"Baseline BER:          {results['baseline_ber']:.6e}")
    print(f"Equalized BER:         {results['equalized_ber']:.6e}")
    print(f"Absolute reduction:    {results['improvement_abs']*100:.4f} pp")
    print(f"Relative improvement:  {results['improvement_rel']:.2f}%  ← КЛЮЧЕВАЯ МЕТРИКА")
    print(f"SNR gain:              {results['improvement_db']:.2f} dB")
    print(f"Symbol Error Rate:     {results['ser']:.6f}")
    print("="*80)

    torch.save(trainer.history, "history.pt")
    print("✓ Training history saved to history.pt")

if __name__ == "__main__":
    main()
