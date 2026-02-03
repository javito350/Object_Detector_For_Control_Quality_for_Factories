import argparse
import torch
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
from typing import List, Dict, Any

from models.anomaly_inspector import AnomalyInspector, InspectionResult
from utils.image_loader import MVTecStyleDataset
from utils.visualization import AnomalyVisualizer
from utils.metrics import AdvancedMetrics

class ProductionInspector:
    """
    PhD-level production inspection system.
    Handles batch processing, logging, and quality control.
    """
    
    def __init__(self, model_dir: str, category: str, device: str = "cuda"):
        self.category = category
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # Load trained model
        self.inspector = AnomalyInspector(device=self.device)
        self.inspector.memory_bank.load(os.path.join(model_dir, f"{category}_memory_bank.h5"))
        
        # Load thresholds
        thresholds_path = os.path.join(model_dir, f"{category}_thresholds.json")
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                thresholds = json.load(f)
                self.inspector.image_threshold = thresholds['image_threshold']
                self.inspector.pixel_threshold = thresholds['pixel_threshold']
        
        # Initialize logging
        self.results_dir = os.path.join("results", category, datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.log_file = os.path.join(self.results_dir, "inspection_log.jsonl")
        self.batch_results = []
        
    def inspect_single(self, image_path: str, save_visualization: bool = True) -> Dict[str, Any]:
        """
        Inspect single image.
        
        Returns:
            Dictionary with inspection results
        """
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self._preprocess_image(image)
        
        # Run inspection
        results = self.inspector.predict(image_tensor.unsqueeze(0))
        result = results[0]
        
        # Convert to dictionary
        result_dict = {
            'image_path': image_path,
            'timestamp': datetime.now().isoformat(),
            'image_score': float(result.image_score),
            'is_defective': bool(result.is_defective()),
            'defect_type': result.defect_type.value,
            'confidence': float(result.confidence),
            'severity': float(result.severity),
            'anomaly_pixels': int(result.metadata['num_anomaly_pixels']),
            'bbox': result.bbox,
            'decision': 'REJECT' if result.is_defective() else 'ACCEPT'
        }
        
        # Save visualization
        if save_visualization:
            vis_path = os.path.join(self.results_dir, 
                                   f"result_{os.path.basename(image_path)}.png")
            image_np = np.array(image)
            AnomalyVisualizer.visualize_result(image_np, result, save_path=vis_path)
            result_dict['visualization_path'] = vis_path
        
        # Log result
        self._log_result(result_dict)
        
        return result_dict
    
    def inspect_batch(self, image_dir: str, batch_size: int = 8) -> List[Dict[str, Any]]:
        """
        Inspect batch of images from directory.
        """
        from torch.utils.data import DataLoader, Dataset
        
        class ImageFolderDataset(Dataset):
            def __init__(self, image_dir):
                self.image_paths = []
                for ext in ['.jpg', '.png', '.jpeg', '.bmp']:
                    self.image_paths.extend(
                        [os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                         if f.lower().endswith(ext)]
                    )
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                image = Image.open(self.image_paths[idx]).convert('RGB')
                return self._preprocess_image(image), self.image_paths[idx]
            
            def _preprocess_image(self, image):
                # Same preprocessing as training
                from torchvision import transforms as T
                transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
                ])
                return transform(image)
        
        dataset = ImageFolderDataset(image_dir)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_results = []
        
        for batch_images, batch_paths in dataloader:
            # Run inspection
            batch_results = self.inspector.predict(batch_images)
            
            # Process each result
            for i, (result, image_path) in enumerate(zip(batch_results, batch_paths)):
                result_dict = {
                    'image_path': image_path,
                    'timestamp': datetime.now().isoformat(),
                    'image_score': float(result.image_score),
                    'is_defective': bool(result.is_defective()),
                    'defect_type': result.defect_type.value,
                    'confidence': float(result.confidence),
                    'severity': float(result.severity),
                    'anomaly_pixels': int(result.metadata['num_anomaly_pixels']),
                    'bbox': result.bbox,
                    'decision': 'REJECT' if result.is_defective() else 'ACCEPT'
                }
                
                all_results.append(result_dict)
                self._log_result(result_dict)
                
                # Save individual result
                result_file = os.path.join(self.results_dir, 
                                          f"{os.path.basename(image_path).split('.')[0]}_result.json")
                with open(result_file, 'w') as f:
                    json.dump(result_dict, f, indent=4)
        
        # Generate batch summary
        self._generate_batch_summary(all_results)
        
        return all_results
    
    def evaluate_on_test_set(self, test_data_root: str) -> Dict[str, float]:
        """
        Evaluate model on test set with ground truth.
        """
        from torch.utils.data import DataLoader
        
        test_dataset = MVTecStyleDataset(test_data_root, self.category, is_train=False)
        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        all_pred_scores = []
        all_gt_labels = []
        all_pred_masks = []
        all_gt_masks = []
        
        for images, labels, paths in test_loader:
            results = self.inspector.predict(images)
            
            for result, label in zip(results, labels):
                all_pred_scores.append(result.image_score)
                all_gt_labels.append(int(label))
                
                # For pixel-level evaluation, need ground truth masks
                # This assumes masks are available in the dataset
                # You'll need to extend MVTecStyleDataset to load masks
        
        # Convert to numpy
        pred_scores = np.array(all_pred_scores)
        gt_labels = np.array(all_gt_labels)
        
        # Compute metrics
        metrics = AdvancedMetrics.compute_all_metrics(
            pred_scores, gt_labels, 
            None, None  # Replace with masks if available
        )
        
        # Save metrics
        metrics_path = os.path.join(self.results_dir, "test_metrics.json")
        AdvancedMetrics.save_metrics(metrics, metrics_path)
        
        return metrics
    
    def _preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model"""
        from torchvision import transforms as T
        
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image)
    
    def _log_result(self, result: Dict[str, Any]) -> None:
        """Log inspection result to JSONL file"""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(result) + '\n')
    
    def _generate_batch_summary(self, results: List[Dict[str, Any]]) -> None:
        """Generate summary statistics for batch"""
        total = len(results)
        defective = sum(1 for r in results if r['is_defective'])
        acceptance_rate = (total - defective) / total * 100
        
        # Group by defect type
        defect_types = {}
        for r in results:
            if r['is_defective']:
                defect_type = r['defect_type']
                defect_types[defect_type] = defect_types.get(defect_type, 0) + 1
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_inspected': total,
            'defective_count': defective,
            'acceptance_rate': acceptance_rate,
            'defect_distribution': defect_types,
            'average_severity': np.mean([r['severity'] for r in results if r['is_defective']] or [0])
        }
        
        summary_path = os.path.join(self.results_dir, "batch_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\n{'='*50}")
        print(f"BATCH INSPECTION SUMMARY")
        print(f"{'='*50}")
        print(f"Total inspected: {total}")
        print(f"Defective: {defective}")
        print(f"Acceptance rate: {acceptance_rate:.2f}%")
        print(f"Defect distribution: {defect_types}")
        print(f"{'='*50}")

def main():
    parser = argparse.ArgumentParser(description='PhD-Level Industrial Inspection System')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['single', 'batch', 'evaluate'],
                       help='Inspection mode')
    parser.add_argument('--category', type=str, required=True,
                       help='Product category (phone_cases, water_bottles, etc.)')
    parser.add_argument('--input_path', type=str, required=True,
                       help='Path to image or directory')
    parser.add_argument('--model_dir', type=str, default='./results',
                       help='Directory containing trained models')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size for batch inspection')
    
    args = parser.parse_args()
    
    # Initialize inspector
    inspector = ProductionInspector(args.model_dir, args.category)
    
    if args.mode == 'single':
        result = inspector.inspect_single(args.input_path)
        print("\nINSPECTION RESULT:")
        print(json.dumps(result, indent=2))
        
    elif args.mode == 'batch':
        results = inspector.inspect_batch(args.input_path, args.batch_size)
        print(f"\nProcessed {len(results)} images")
        
    elif args.mode == 'evaluate':
        metrics = inspector.evaluate_on_test_set(args.input_path)
        print("\nEVALUATION METRICS:")
        print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()