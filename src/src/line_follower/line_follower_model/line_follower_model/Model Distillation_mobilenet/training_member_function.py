import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import glob
import PIL.Image
import os
import numpy as np

# ---------------- Hybrid Loss Function ------------------
def hybrid_classification_regression_loss(outputs, labels, y_threshold=0.8707, w_class=1.0, w_reg=14.0):
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

# ---------------- Distillation Loss ------------------
def distillation_loss(student_outputs, teacher_outputs, labels, alpha=0.5):
    mse = torch.nn.MSELoss()
    soft_loss = mse(student_outputs, teacher_outputs)
    hard_loss, _, _ = hybrid_classification_regression_loss(student_outputs, labels)
    total_loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return total_loss, hard_loss.item(), soft_loss.item()

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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Teacher Model (ResNet18)
    teacher_model = models.resnet18(weights=None)
    teacher_model.fc = torch.nn.Linear(in_features=512, out_features=2)
    teacher_model.load_state_dict(torch.load('./best_line_follower_model_xy.pth'))
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Build Student Model (MobileNetV3 Small)
    student_model = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
    student_model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=2)
    student_model = student_model.to(device)

    optimizer = optim.Adam(student_model.parameters())
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)

    NUM_EPOCHS = 100
    BEST_MODEL_PATH = './mobilenetv3_student_distilled.pth'
    best_loss = 1e9
    alpha = 0.5  # Balance between hard and soft loss

    for epoch in range(NUM_EPOCHS):
        student_model.train()
        train_total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            student_outputs = student_model(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            loss, hard_loss_val, soft_loss_val = distillation_loss(student_outputs, teacher_outputs, labels, alpha)
            loss.backward()
            optimizer.step()
            train_total_loss += loss.item()
            total_hard_loss += hard_loss_val
            total_soft_loss += soft_loss_val

        train_total_loss /= len(train_loader)
        total_hard_loss /= len(train_loader)
        total_soft_loss /= len(train_loader)

        student_model.eval()
        test_total_loss = 0.0
        test_hard_loss = 0.0
        test_soft_loss = 0.0

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            student_outputs = student_model(images)
            with torch.no_grad():
                teacher_outputs = teacher_model(images)
            loss, hard_loss_val, soft_loss_val = distillation_loss(student_outputs, teacher_outputs, labels, alpha)
            test_total_loss += loss.item()
            test_hard_loss += hard_loss_val
            test_soft_loss += soft_loss_val

        test_total_loss /= len(test_loader)
        test_hard_loss /= len(test_loader)
        test_soft_loss /= len(test_loader)

        scheduler.step(test_total_loss)

        hard_pct = test_hard_loss / (test_hard_loss + test_soft_loss + 1e-8)
        soft_pct = test_soft_loss / (test_hard_loss + test_soft_loss + 1e-8)

        print(f"\n🧪 Epoch {epoch+1} Summary")
        print(f"Train Loss: {train_total_loss:.6f}, Test Loss: {test_total_loss:.6f}")
        print(f"Hard Loss:  {test_hard_loss:.6f} ({hard_pct:.2%}), Soft Loss: {test_soft_loss:.6f} ({soft_pct:.2%})")

        if test_total_loss < best_loss:
            print("✅ Saved best student model")
            torch.save(student_model.state_dict(), BEST_MODEL_PATH)
            best_loss = test_total_loss

if __name__ == '__main__':
    main()

