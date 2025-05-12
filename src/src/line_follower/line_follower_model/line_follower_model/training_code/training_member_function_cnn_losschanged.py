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

# 定义混合损失函数    
def hybrid_classification_regression_loss(outputs, labels, y_threshold=0.8707, w_class=1.0, w_reg=2.0):
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
    return total_loss

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

    # 换成自己的小CNN
    model = SmallCNN()
    device = torch.device('cpu')  # 可以改成 'cuda' 如果你有GPU
    model = model.to(device)

    NUM_EPOCHS = 100
    BEST_MODEL_PATH = './best_line_follower_model.pth'
    best_loss = 1e9

    optimizer = optim.Adam(model.parameters())
    
    # 设置学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in iter(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = hybrid_classification_regression_loss(outputs, labels)  # 使用自定义损失函数
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        
        train_loss /= len(train_loader)

        # 在验证集上计算损失
        model.eval()
        test_loss = 0.0
        with torch.no_grad():  # 测试时禁用反向传播，更快
            for images, labels in iter(test_loader):
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = hybrid_classification_regression_loss(outputs, labels)  # 使用自定义损失函数
                test_loss += float(loss)
        
        test_loss /= len(test_loader)
        
        print('Epoch %d, Train Loss: %.4f, Test Loss: %.4f' % (epoch, train_loss, test_loss))
        
        # 使用 ReduceLROnPlateau 调整学习率
        scheduler.step(test_loss)
 	# 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch}, Current Learning Rate: {current_lr}')
        
        # 保存最优模型
        if test_loss < best_loss:
            print("Saving model with better test loss")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_loss


if __name__ == '__main__':
    main()

