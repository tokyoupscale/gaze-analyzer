import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
from tqdm import tqdm
import time

class MPIIGazeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        
        # Загружаем все данные
        for participant in sorted(os.listdir(data_dir)):
            participant_dir = os.path.join(data_dir, participant)
            if not os.path.isdir(participant_dir):
                continue
                
            for day in sorted(os.listdir(participant_dir)):
                if not day.startswith('day'):
                    continue
                    
                day_dir = os.path.join(participant_dir, day)
                annotation_file = os.path.join(day_dir, 'annotation.txt')
                
                if not os.path.exists(annotation_file):
                    continue
                
                # Читаем аннотации
                with open(annotation_file, 'r') as f:
                    annotations = f.readlines()
                
                # Загружаем изображения
                for img_file in sorted(os.listdir(day_dir)):
                    if not img_file.endswith('.jpg'):
                        continue
                    
                    frame_idx = int(img_file.split('.')[0]) - 1
                    if frame_idx >= len(annotations):
                        continue
                    
                    values = annotations[frame_idx].strip().split()
                    if len(values) < 2:
                        continue
                    
                    yaw = float(values[-2])
                    pitch = float(values[-1])
                    
                    img_path = os.path.join(day_dir, img_file)
                    self.samples.append((img_path, yaw, pitch))
        
        print(f"Loaded {len(self.samples)} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, yaw, pitch = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor([yaw, pitch], dtype=torch.float32)
        return image, label


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def test_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    
    pbar = tqdm(loader, desc='Testing', leave=False)
    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)


def main():
    # Параметры
    DATA_DIR = 'MPIIGaze/Data/Original'
    BATCH_SIZE = 32
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001
    TRAIN_SPLIT = 0.8
    
    print("="*60)
    print("MPII Gaze Estimation Training")
    print("="*60)
    
    # Трансформации
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Загружаем датасет
    print("\n[1/5] Loading dataset...")
    dataset = MPIIGazeDataset(DATA_DIR, transform=transform)
    
    # Разделяем на train/test
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"Train: {train_size} samples, Test: {test_size} samples")
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, num_workers=4)
    
    # Модель
    print("\n[2/5] Building model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 2)  # 2 выхода: yaw, pitch
    model = model.to(device)
    
    # Оптимизатор и лосс
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Обучение
    print("\n[3/5] Training...")
    print("="*60)
    
    best_test_loss = float('inf')
    
    for epoch in tqdm(range(NUM_EPOCHS), desc='Epochs'):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Test
        test_loss = test_epoch(model, test_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        # Логирование
        tqdm.write(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        tqdm.write(f"  Train Loss: {train_loss:.6f}")
        tqdm.write(f"  Test Loss:  {test_loss:.6f}")
        tqdm.write(f"  Time: {epoch_time:.2f}s")
        tqdm.write("-"*60)
        
        # Сохраняем лучшую модель
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    # Финальное тестирование
    print("\n[4/5] Final testing...")
    final_test_loss = test_epoch(model, test_loader, criterion, device)
    print(f"Final Test Loss: {final_test_loss:.6f}")
    
    # Сохранение модели
    print("\n[5/5] Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_loss': final_test_loss,
    }, 'gaze_model_final.pth')
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best test loss: {best_test_loss:.6f}")
    print(f"Models saved:")
    print(f"  - best_model.pth (best during training)")
    print(f"  - gaze_model_final.pth (final model)")
    print("="*60)


if __name__ == '__main__':
    main()