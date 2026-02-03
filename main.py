import argparse
import sys
import os

from scripts.run_baseline import main as run_baseline
from scripts.run_inspection import main as run_inspection
from scripts.train_advanced import main as train_advanced
from scripts.optimize_thresholds import main as optimize_thresholds

def main():
    parser = argparse.ArgumentParser(
        description='PhD-Level Industrial Defect Inspection System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model for phone_cases
  python main.py train --category phone_cases --data_root ./data
  
  # Inspect single image
  python main.py inspect --mode single --category phone_cases --image_path ./test.jpg
  
  # Batch inspection
  python main.py inspect --mode batch --category water_bottles --input_dir ./batch_images/
  
  # Evaluate on test set
  python main.py evaluate --category phone_cases --test_root ./data/phone_cases/test
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train anomaly detection model')
    train_parser.add_argument('--category', type=str, required=True,
                            help='Product category (phone_cases, water_bottles)')
    train_parser.add_argument('--data_root', type=str, default='./data',
                            help='Root directory of dataset')
    train_parser.add_argument('--backbone', type=str, default='wide_resnet50_2',
                            choices=['wide_resnet50_2', 'resnet18', 'efficientnet_b0'])
    train_parser.add_argument('--coreset_percentage', type=float, default=0.1,
                            help='Percentage of features to keep in memory bank')
    train_parser.add_argument('--output_dir', type=str, default='./results',
                            help='Output directory for models and results')
    
    # Inspect command
    inspect_parser = subparsers.add_parser('inspect', help='Run inspection')
    inspect_parser.add_argument('--mode', type=str, required=True,
                              choices=['single', 'batch'],
                              help='Inspection mode')
    inspect_parser.add_argument('--category', type=str, required=True,
                              help='Product category')
    inspect_parser.add_argument('--image_path', type=str,
                              help='Path to single image (for single mode)')
    inspect_parser.add_argument('--input_dir', type=str,
                              help='Directory with images (for batch mode)')
    inspect_parser.add_argument('--model_dir', type=str, default='./results',
                              help='Directory containing trained models')
    inspect_parser.add_argument('--save_visualization', action='store_true',
                              help='Save visualization images')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--category', type=str, required=True,
                           help='Product category')
    eval_parser.add_argument('--test_root', type=str, required=True,
                           help='Test dataset root')
    eval_parser.add_argument('--model_dir', type=str, default='./results',
                           help='Directory containing trained models')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Optimize thresholds')
    optimize_parser.add_argument('--category', type=str, required=True,
                               help='Product category')
    optimize_parser.add_argument('--val_root', type=str, required=True,
                               help='Validation dataset root')
    optimize_parser.add_argument('--model_dir', type=str, default='./results',
                               help='Directory containing trained models')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        sys.argv = ['run_baseline.py', 
                   '--category', args.category,
                   '--data_root', args.data_root,
                   '--backbone', args.backbone,
                   '--coreset_percentage', str(args.coreset_percentage),
                   '--output_dir', args.output_dir]
        run_baseline()
        
    elif args.command == 'inspect':
        if args.mode == 'single' and not args.image_path:
            print("ERROR: --image_path required for single mode")
            sys.exit(1)
        elif args.mode == 'batch' and not args.input_dir:
            print("ERROR: --input_dir required for batch mode")
            sys.exit(1)
        
        sys.argv = ['run_inspection.py',
                   '--mode', args.mode,
                   '--category', args.category,
                   '--input_path', args.image_path if args.mode == 'single' else args.input_dir,
                   '--model_dir', args.model_dir]
        if args.save_visualization:
            sys.argv.append('--save_visualization')
        
        run_inspection()
        
    elif args.command == 'evaluate':
        sys.argv = ['run_inspection.py',
                   '--mode', 'evaluate',
                   '--category', args.category,
                   '--input_path', args.test_root,
                   '--model_dir', args.model_dir]
        run_inspection()
        
    elif args.command == 'optimize':
        sys.argv = ['optimize_thresholds.py',
                   '--category', args.category,
                   '--val_root', args.val_root,
                   '--model_dir', args.model_dir]
        optimize_thresholds()
        
    else:
        parser.print_help()

if __name__ == '__main__':
    main()