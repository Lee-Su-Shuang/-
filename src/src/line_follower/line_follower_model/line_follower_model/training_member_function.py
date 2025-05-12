import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

# --- 辅助函数
def get_x(path, width):
    return (float(int(path.split("_")[1])) * 224.0 / 640.0 - width/2) / (width/2)

def get_y(path, height):
    return ((224 - float(int(path.split("_")[2]))) - height/2) / (height/2)

# --- 自定义 Dataset
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory, random_hflips=False):
        self.directory = directory
        self.random_hflips = random_hflips
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        self.color_jitter = transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path), 224))
        y = float(get_y(os.path.basename(image_path), 224))
      
        if self.random_hflips:
            if float(np.random.rand(1)) > 0.5:
                image = transforms.functional.hflip(image)
                x = -x
        
        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = image.numpy().copy()
        image = torch.from_numpy(image)
        image = transforms.functional.normalize(image,
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()

# --- 自定义小型CNN
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),  # 输出: 16x224x224
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 16x112x112
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),  # 输出: 32x112x112
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 32x56x56
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # 输出: 64x56x56
            nn.ReLU(),
            nn.MaxPool2d(2),  # 输出: 64x28x28
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 2)  # 输出 (x, y)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- 自定义混合损失函数
def hybrid_classification_regression_loss(outputs, labels, y_threshold=0.8707, w_class=1.0, w_reg=15.0):
    pred_x, pred_y = outputs[:, 0], outputs[:, 1]
    true_x, true_y = labels[:, 0], labels[:, 1]
    
    true_class = (true_y > y_threshold).float()
    pred_class = (pred_y > y_threshold).float()
    
    misclassified = (pred_class != true_class).float()
    class_loss = misclassified * (pred_y - true_y).abs()
    
    straight_mask = (true_y < y_threshold) & (pred_y < y_threshold)
    reg_loss = torch.zeros(1, device=outputs.device)
    if straight_mask.any():
        reg_loss = F.mse_loss(pred_x[straight_mask], true_x[straight_mask], reduction='mean')
    
    total_loss = w_class * class_loss.mean() + w_reg * reg_loss
    return total_loss, w_class * class_loss.mean(), w_reg * reg_loss

# --- 主程序
def main(args=None):
    dataset = XYDataset('./image_dataset', random_hflips=False)

    test_percent = 0.1
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=0
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=24,
        shuffle=True,
        num_workers=0
    )

    model = SmallCNN()
    device = torch.device('cpu')  # 或使用 'cuda' 如果有GPU
    model = model.to(device)

    NUM_EPOCHS = 100
    BEST_MODEL_PATH = './best_line_follower_model_xy.pth'
    best_loss = 1e9

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # --- 早停设置
    patience = 20
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        train_class_loss = 0.0
        train_reg_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            total_loss, weighted_class_loss, weighted_reg_loss = hybrid_classification_regression_loss(outputs, labels)
            train_loss += float(total_loss)
            train_class_loss += float(weighted_class_loss)
            train_reg_loss += float(weighted_reg_loss)
            total_loss.backward()
            optimizer.step()
        
        train_loss /= len(train_loader)
        train_class_loss /= len(train_loader)
        train_reg_loss /= len(train_loader)
        # 计算占比
        train_class_ratio = train_class_loss / train_loss * 100 if train_loss > 0 else 0
        train_reg_ratio = train_reg_loss / train_loss * 100 if train_loss > 0 else 0

        model.eval()
        test_loss = 0.0
        test_class_loss = 0.0
        test_reg_loss = 0.0
        with torch.no_grad():
            for images, labels in iter(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                total_loss, weighted_class_loss, weighted_reg_loss = hybrid_classification_regression_loss(outputs, labels)
                test_loss += float(total_loss)
                test_class_loss += float(weighted_class_loss)
                test_reg_loss += float(weighted_reg_loss)
        
        test_loss /= len(train_loader)
        test_class_loss /= len(train_loader)
        test_reg_loss /= len(train_loader)
        # 计算占比
        test_class_ratio = test_class_loss / test_loss * 100 if test_loss > 0 else 0
        test_reg_ratio = test_reg_loss / test_loss * 100 if test_loss > 0 else 0

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f} '
              f'(Class Loss: {train_class_loss:.4f}, {train_class_ratio:.2f}%; '
              f'Reg Loss: {train_reg_loss:.4f}, {train_reg_ratio:.2f}%), '
              f'Test Loss: {test_loss:.4f} '
              f'(Class Loss: {test_class_loss:.4f}, {test_class_ratio:.2f}%; '
              f'Reg Loss: {test_reg_loss:.4f}, {test_reg_ratio:.2f}%)')

        scheduler.step(test_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}, Current Learning Rate: {current_lr}')

        if test_loss < best_loss:
            print("Saving model with better test loss")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        if epochs_without_improvement >= patience:
            print(f"Validation loss didn't improve for {patience} consecutive epochs. Stopping training.")
            break

if __name__ == '__main__':
    main()
