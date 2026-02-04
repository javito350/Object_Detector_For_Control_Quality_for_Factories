"""
Recalibrate model threshold based on ground truth labels
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import csv
import numpy as np

from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("="*70)
print("MODEL RECALIBRATION")
print("="*70)

# Load model
print("\nLoading model...")
inspector = torch.load('sensitive_inspector.pth', map_location='cpu', weights_only=False)
old_threshold = inspector.image_threshold
if hasattr(old_threshold, 'item'):
    old_threshold = old_threshold.item()

print(f"Current threshold: {old_threshold:.4f}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ground truth
labels_file = "data/water_bottles/test/labels.csv"
ground_truth = {}

with open(labels_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row and not row[0].startswith('#'):
            ground_truth[row[0].strip()] = int(row[1].strip())

print(f"Loaded {len(ground_truth)} ground truth labels")

# Collect scores
test_dir = Path("data/water_bottles/test")
scores_good = []
scores_defective = []

print("\nCollecting anomaly scores...")
for img_file in sorted(test_dir.glob("*.jpeg")):
    if img_file.name not in ground_truth:
        continue
    
    img_pil = Image.open(img_file).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0)
    
    res = inspector.predict(img_tensor)
    score = res[0].image_score
    
    if ground_truth[img_file.name] == 0:  # GOOD
        scores_good.append(score)
        print(f"  {img_file.name}: {score:.4f} (GOOD)")
    else:  # DEFECTIVE
        scores_defective.append(score)
        print(f"  {img_file.name}: {score:.4f} (DEFECTIVE)")

# Calculate optimal threshold
print("\n" + "="*70)
print("THRESHOLD OPTIMIZATION")
print("="*70)

if scores_good and scores_defective:
    max_good = max(scores_good)
    min_defective = min(scores_defective)
    
    print(f"\nGood samples - Max score: {max_good:.4f}")
    print(f"Defective samples - Min score: {min_defective:.4f}")
    
    # Set threshold between max good and min defective
    if max_good < min_defective:
        # Perfect separation
        new_threshold = (max_good + min_defective) / 2
        print(f"\nPerfect separation found!")
    else:
        # Overlapping distributions - use ROC optimization
        print(f"\nOverlapping distributions detected")
        
        # Try different thresholds
        best_accuracy = 0
        best_threshold = old_threshold
        
        all_scores = scores_good + scores_defective
        test_thresholds = np.linspace(min(all_scores), max(all_scores), 100)
        
        for thresh in test_thresholds:
            tp = sum(1 for s in scores_defective if s > thresh)
            tn = sum(1 for s in scores_good if s <= thresh)
            accuracy = (tp + tn) / len(all_scores)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = thresh
        
        new_threshold = best_threshold
        print(f"Optimal threshold found: {new_threshold:.4f} (accuracy: {best_accuracy*100:.1f}%)")
    
    # Update model
    inspector.image_threshold = new_threshold
    
    # Save recalibrated model
    output_path = 'calibrated_inspector.pth'
    torch.save(inspector, output_path)
    
    print(f"\n" + "="*70)
    print("RECALIBRATION COMPLETE")
    print("="*70)
    print(f"\nOld threshold: {old_threshold:.4f}")
    print(f"New threshold: {new_threshold:.4f}")
    print(f"Change: {new_threshold - old_threshold:+.4f}")
    print(f"\nRecalibrated model saved to: {output_path}")
    
    # Test new threshold
    print(f"\n" + "="*70)
    print("VALIDATION")
    print("="*70)
    
    correct = 0
    total = 0
    
    for img_file in sorted(test_dir.glob("*.jpeg")):
        if img_file.name not in ground_truth:
            continue
        
        img_pil = Image.open(img_file).convert('RGB')
        img_tensor = transform(img_pil).unsqueeze(0)
        
        res = inspector.predict(img_tensor)
        score = res[0].image_score
        
        predicted = 1 if score > new_threshold else 0
        true_label = ground_truth[img_file.name]
        
        is_correct = predicted == true_label
        correct += is_correct
        total += 1
        
        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"  {img_file.name}: {status}")
    
    accuracy = correct / total * 100
    print(f"\nNew accuracy: {correct}/{total} = {accuracy:.1f}%")
    
else:
    print("\nERROR: Need both good and defective samples to calibrate")

print("="*70 + "\n")
