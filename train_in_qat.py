'''
Finetune 一個Pretrain過的MobileNetV2，用的是QAT
Training (QAT) and export an INT8 quantized checkpoint.
'''

import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.quantization import get_default_qat_qconfig, prepare_qat, convert
from model_for_qat import MobileNetV2 

# Config and Parameters
NUM_CLASSES   = 43
BATCH_SIZE    = 64
PRETRAIN_PATH = "checkpoints/mobilenetv2_gtsrb.pth"          # FP32
OUT_INT8_PATH = "checkpoints/mobilenetv2_gtsrb_qat.pth"      # INT8
FINETUNE_EPOCHS = 2
LR_QAT       = 1e-4                                          
DATA_DIR     = "data/gtsrb/GTSRB/Training"
CURVE_PATH       = "qat_training_curves.png"

def get_dataloaders():
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    full = datasets.ImageFolder(DATA_DIR, transform=tfm)
    tr_size = int(0.8 * len(full))
    va_size = len(full) - tr_size
    tr_set, va_set = random_split(full, [tr_size, va_size],
                                  generator=torch.Generator().manual_seed(42))
    tr_loader = DataLoader(tr_set, batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(va_set, batch_size=BATCH_SIZE, shuffle=False)
    return tr_loader, va_loader

def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    return 100.0 * correct / total

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load FP32 pretrained model
    model = MobileNetV2(ch_in=3, n_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(PRETRAIN_PATH, map_location=device))

    # Eval → fuse → Train
    model.eval()
    model.fuse_model()
    model.train()

    # QAT config & prepare
    model.qconfig = get_default_qat_qconfig('fbgemm')  # x86 backend
    prepare_qat(model, inplace=True)
    print("Prepared model for QAT.")

    # Dataloaders, criterion, optimizer
    train_loader, val_loader = get_dataloaders()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR_QAT)

    # Training session
    best_acc = 0.0
    train_losses, val_accs = [], []
    for epoch in range(1, FINETUNE_EPOCHS+1):
        model.train()
        run_loss = 0.; correct = total = 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{FINETUNE_EPOCHS} [Train]"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward(); optimizer.step()
            run_loss += loss.item()
            correct += (out.argmax(1) == labels).sum().item(); total += labels.size(0)

        tr_loss = run_loss / len(train_loader)
        va_acc  = evaluate(model, val_loader, device)
        train_losses.append(tr_loss); val_accs.append(va_acc)

        print(f"[{epoch}/{FINETUNE_EPOCHS}] loss {tr_loss:.4f} | val acc {va_acc:.2f}%")

    # Convert to real INT8 (must be on CPU)
    model.to('cpu')
    int8_model = convert(model.eval(), inplace=False)
    torch.save(int8_model.state_dict(), OUT_INT8_PATH)
    print(f"INT8 model saved to {OUT_INT8_PATH}")

    # 6. Evaluate INT8 accuracy
    int8_acc = evaluate(int8_model, val_loader, device='cpu')
    print(f"INT8 Val Accuracy: {int8_acc:.2f}% (Best fake-quant {best_acc:.2f}%)")

    # ---------- plot curves ----------
    epochs = range(1, FINETUNE_EPOCHS+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, train_losses, '-o'); plt.title("QAT Train Loss")
    plt.xlabel("Epoch"); plt.ylabel("Loss")

    plt.subplot(1,2,2)
    plt.plot(epochs, val_accs, '-o'); plt.title("QAT Val Accuracy")
    plt.xlabel("Epoch"); plt.ylabel("Acc (%)")

    plt.tight_layout(); plt.savefig(CURVE_PATH)
    print(f"Curves saved → {CURVE_PATH}")

if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    main()