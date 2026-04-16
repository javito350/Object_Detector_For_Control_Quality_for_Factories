import shutil
import pandas as pd
from pathlib import Path

def convert_visa_to_mvtec(visa_root: str, output_root: str):
    visa_path = Path(visa_root)
    out_path = Path(output_root)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find all CSV files in the split_csv directory
    csv_files = list(visa_path.glob("split_csv/**/*.csv"))
    if not csv_files:
        print("Error: Could not find any CSV files in VisA/split_csv/")
        return

    print(f"Found {len(csv_files)} CSV files. Converting to MVTec format...")

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        
        # Ensure this is a dataset split CSV by checking for required columns
        if not all(col in df.columns for col in ['image', 'split', 'label']):
            continue
            
        for _, row in df.iterrows():
            img_rel = str(row['image'])
            split = str(row['split']) # 'train' or 'test'
            label = str(row['label']) # 'normal' or 'anomaly'
            
            # The category name is the first part of the path (e.g., 'candle')
            cat_name = Path(img_rel).parts[0]
            
            src_img = visa_path / img_rel
            if not src_img.exists():
                continue

            # Route to MVTec folder structure
            if split == 'train':
                dest_img = out_path / cat_name / "train" / "good" / src_img.name
            else:
                sub_folder = "good" if label == 'normal' else "bad"
                dest_img = out_path / cat_name / "test" / sub_folder / src_img.name

            dest_img.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_img, dest_img)

            # Route the Ground Truth Masks
            if split == 'test' and label == 'anomaly':
                mask_rel = row.get('mask')
                if pd.notna(mask_rel):
                    src_mask = visa_path / str(mask_rel)
                    if src_mask.exists():
                        dest_mask = out_path / cat_name / "ground_truth" / "bad" / f"{src_img.stem}_mask{src_mask.suffix}"
                        dest_mask.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(src_mask, dest_mask)

    print(f"\nSuccess! VisA dataset is now MVTec-compatible at: {out_path.resolve()}")

if __name__ == "__main__":
    convert_visa_to_mvtec("VisA", "data/visa_mvtec_format")