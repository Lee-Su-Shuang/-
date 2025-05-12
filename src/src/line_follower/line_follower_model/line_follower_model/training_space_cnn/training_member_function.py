import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

# ---------------- Small Custom CNN ------------------
class SmallCNN(nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# ---------------- Hybrid Loss Function ------------------
def hybrid_classification_regression_loss(outputs, labels, y_threshold=0.8707, w_class=1.0, w_reg=9.0):
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

    class_loss_value = w_class * class_loss.mean()
    reg_loss_value = w_reg * reg_loss
    total_loss = class_loss_value + reg_loss_value
    return total_loss, class_loss_value, reg_loss_value

# ---------------- Dataset Class ------------------
def get_x(path, width):
    return (float(int(path.split("_")[1])) * 224.0 / 640.0 - width/2) / (width/2)

def get_y(path, height):
    return ((224 - float(int(path.split("_")[2]))) - height/2) / (height/2)

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

        if self.random_hflips and float(np.random.rand(1)) > 0.5:
            image = transforms.functional.hflip(image)
            x = -x

        image = self.color_jitter(image)
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image,
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        return image, torch.tensor([x, y]).float()

# ---------------- Main Training ------------------
def main():
    dataset = XYDataset('./image_dataset', random_hflips=False)
    test_percent = 0.1
    num_test = int(test_percent * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=24, shuffle=False, num_workers=0)

    model = SmallCNN()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    NUM_EPOCHS = 100
    BEST_MODEL_PATH = './best_line_follower_model_xy.pth'
    best_loss = 1e9

    optimizer = optim.Adam(model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, class_loss, reg_loss = hybrid_classification_regression_loss(outputs, labels)
            loss.backward()
            optimizer.step()
            train_total_loss += float(loss)

        train_total_loss /= len(train_loader)

        model.eval()
        test_total_loss = 0.0
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss, class_loss, reg_loss = hybrid_classification_regression_loss(outputs, labels)
            test_total_loss += float(loss)

        test_total_loss /= len(test_loader)

        scheduler.step(test_total_loss)

        total = class_loss.item() + reg_loss.item() + 1e-8
        print(f"Epoch {epoch+1} - Train Loss: {train_total_loss:.6f}, Test Loss: {test_total_loss:.6f}")
        print(f"           Class Loss: {class_loss.item():.6f} ({class_loss.item() / total:.2%}), Reg Loss: {reg_loss.item():.6f} ({reg_loss.item() / total:.2%})")
        print(f"           w_class * class_loss: {class_loss.item():.6f}, w_reg * reg_loss: {reg_loss.item():.6f}")

        current_lr = optimizer.param_groups[0]['lr']
        print(f"           Current Learning Rate: {current_lr:.6f}")

        if test_total_loss < best_loss:
            print("✅ Successfully saved best model!")
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_total_loss

if __name__ == '__main__':
    main()

