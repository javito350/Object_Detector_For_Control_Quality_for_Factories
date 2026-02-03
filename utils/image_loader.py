import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms as T

class MVTecStyleDataset(Dataset):
    def __init__(self, root_dir, category='bottle', is_train=True, img_size=224):
        self.root = os.path.join(root_dir, category, 'train' if is_train else 'test')
        self.is_train = is_train
        self.img_size = img_size
        
        self.samples = []
        # good samples
        good_dir = os.path.join(self.root, 'good')
        if os.path.exists(good_dir):
            for fname in os.listdir(good_dir):
                if fname.endswith(('.png', '.jpg')):
                    self.samples.append((os.path.join(good_dir, fname), 0))
        
        # defective samples (test only)
        if not is_train:
            for defect_type in os.listdir(self.root):
                if defect_type == 'good':
                    continue
                defect_dir = os.path.join(self.root, defect_type)
                if os.path.isdir(defect_dir):
                    for fname in os.listdir(defect_dir):
                        if fname.endswith(('.png', '.jpg')):
                            self.samples.append((os.path.join(defect_dir, fname), 1))
        
        # transformations
        if is_train:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, label, path