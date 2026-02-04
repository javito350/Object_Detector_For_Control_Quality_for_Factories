"""
Evaluate model accuracy on test set with ground truth labels
"""
import torch
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import csv

from models.anomaly_inspector import EnhancedAnomalyInspector
torch.serialization.add_safe_globals([EnhancedAnomalyInspector])

print("="*70)
print("MODEL EVALUATION ON TEST SET")
print("="*70)

# Load model
print("\nLoading model...")
inspector = torch.load('sensitive_inspector.pth', map_location='cpu', weights_only=False)
threshold = inspector.image_threshold
if hasattr(threshold, 'item'):
    threshold = threshold.item()

print(f"Model loaded. Threshold: {threshold:.4f}")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ground truth labels
labels_file = "data/water_bottles/test/labels.csv"
ground_truth = {}

print(f"\nLoading ground truth from {labels_file}...")
with open(labels_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if row and not row[0].startswith('#'):  # Skip comments and empty lines
            filename = row[0].strip()
            label = int(row[1].strip())
            ground_truth[filename] = label  # 0=GOOD, 1=DEFECTIVE

print(f"Loaded {len(ground_truth)} ground truth labels")

# Evaluate
test_dir = Path("data/water_bottles/test")
results = []

print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

for img_file in sorted(test_dir.glob("*.jpeg")):
    if img_file.name not in ground_truth:
        continue
    
    # Load and predict
    img_pil = Image.open(img_file).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0)
    
    res = inspector.predict(img_tensor)
    result = res[0]
    
    score = result.image_score
    predicted_defective = 1 if score > threshold else 0
    true_label = ground_truth[img_file.name]
    
    correct = predicted_defective == true_label
    
    print(f"\n{img_file.name}:")
    print(f"  True Label: {'DEFECTIVE' if true_label == 1 else 'GOOD'}")
    print(f"  Predicted: {'DEFECTIVE' if predicted_defective == 1 else 'GOOD'}")
    print(f"  Score: {score:.4f}")
    print(f"  Result: {'CORRECT' if correct else 'INCORRECT'}")
    
    results.append({
        'file': img_file.name,
        'true': true_label,
        'pred': predicted_defective,
        'score': score,
        'correct': correct
    })

# Calculate metrics
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

total = len(results)
correct_count = sum(1 for r in results if r['correct'])
accuracy = correct_count / total * 100

true_positives = sum(1 for r in results if r['true'] == 1 and r['pred'] == 1)
false_positives = sum(1 for r in results if r['true'] == 0 and r['pred'] == 1)
true_negatives = sum(1 for r in results if r['true'] == 0 and r['pred'] == 0)
false_negatives = sum(1 for r in results if r['true'] == 1 and r['pred'] == 0)

print(f"\nTotal Samples: {total}")
print(f"Correct: {correct_count}")
print(f"Incorrect: {total - correct_count}")
print(f"Accuracy: {accuracy:.1f}%")

print(f"\nConfusion Matrix:")
print(f"  True Positives (Defective detected as Defective): {true_positives}")
print(f"  True Negatives (Good detected as Good): {true_negatives}")
print(f"  False Positives (Good detected as Defective): {false_positives}")
print(f"  False Negatives (Defective detected as Good): {false_negatives}")

if (true_positives + false_positives) > 0:
    precision = true_positives / (true_positives + false_positives)
    print(f"\nPrecision: {precision:.1%}")

if (true_positives + false_negatives) > 0:
    recall = true_positives / (true_positives + false_negatives)
    print(f"Recall: {recall:.1%}")

print("\n" + "="*70)
print("ANALYSIS")
print("="*70)

if accuracy < 80:
    print("\nModel performance is below optimal threshold.")
    print("Recommendations:")
    print("  1. Recalibrate threshold using validation set")
    print("  2. Collect more diverse training samples")
    print("  3. Adjust model sensitivity parameters")
    print(f"\nCurrent threshold: {threshold:.4f}")
    
    # Find optimal threshold
    scores_good = [r['score'] for r in results if r['true'] == 0]
    scores_defect = [r['score'] for r in results if r['true'] == 1]
    
    if scores_good and scores_defect:
        suggested_threshold = (max(scores_good) + min(scores_defect)) / 2
        print(f"Suggested threshold: {suggested_threshold:.4f}")
else:
    print(f"\nModel performs well with {accuracy:.1f}% accuracy.")

print("="*70 + "\n")
