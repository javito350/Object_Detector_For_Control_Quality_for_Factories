import argparse
from utils.image_loader import MVTecStyleDataset
from models.anomaly_inspector import AnomalyInspector
from torch.utils.data import DataLoader
import torch
import json
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', type=str, default='bottle', choices=['bottle', 'carpet', 'phone'])
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='./results')
    args = parser.parse_args()
    
    # dataset
    train_dataset = MVTecStyleDataset(args.data_root, args.category, is_train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    # model
    inspector = AnomalyInspector(device=args.device)
    inspector.fit(train_loader)
    
    # save memory bank
    os.makedirs(f'{args.output_dir}/{args.category}', exist_ok=True)
    torch.save(inspector.memory_bank.bank, f'{args.output_dir}/{args.category}/memory_bank.pt')
    print(f"[INFO] Memory bank saved for {args.category}")
    
    # evaluate on test set
    test_dataset = MVTecStyleDataset(args.data_root, args.category, is_train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    scores = []
    labels = []
    for img, label, _ in test_loader:
        _, img_score = inspector.predict(img)
        scores.append(img_score.item())
        labels.append(label.item())
    
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(labels, scores)
    print(f"[RESULT] Category: {args.category}, Test AUROC: {auc:.4f}")
    
    with open(f'{args.output_dir}/{args.category}/metrics.json', 'w') as f:
        json.dump({'AUROC': auc}, f, indent=4)

if __name__ == '__main__':
    main()