import torch
import numpy as np
import torchvision.transforms as transforms
import glob
import os
import PIL.Image


# 辅助函数，解析归一化坐标
def get_x(path, width):
    return (float(int(path.split("_")[1])) * 224.0 / 640.0 - width/2) / (width/2)

def get_y(path, height):
    return ((224 - float(int(path.split("_")[2]))) - height/2) / (height/2)


# 自定义 Dataset
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


# 自定义小型 CNN
class SmallCNN(torch.nn.Module):
    def __init__(self):
        super(SmallCNN, self).__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 28 * 28, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# 加载数据
dataset = XYDataset('./image_dataset', random_hflips=False)

# 划分训练 / 测试集
test_percent = 0.4
num_test = int(test_percent * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# 加载 SmallCNN 模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SmallCNN()
model.load_state_dict(torch.load('./best_line_follower_model_xy.pth', map_location=device))
model.to(device)
model.eval()

# 测试并计算准确率
threshold = 0.8707  # 对应归一化阈值
all_y_true = []
all_y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        y_true_batch = labels[:, 1].cpu().numpy()
        y_pred_batch = outputs[:, 1].cpu().numpy()

        all_y_true.append(y_true_batch)
        all_y_pred.append(y_pred_batch)

# 整合所有 batch
all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

# 用 all_y_true 和 all_y_pred 来算
is_real_intersection = (all_y_true > threshold)
is_pred_intersection = (all_y_pred > threshold)

TP = np.sum(is_real_intersection & is_pred_intersection)
TN = np.sum(~is_real_intersection & ~is_pred_intersection)
FP = np.sum(~is_real_intersection & is_pred_intersection)
FN = np.sum(is_real_intersection & ~is_pred_intersection)

accuracy = (TP + TN) / (TP + TN + FP + FN)

# 避免除以 0
intersection_total = np.sum(is_real_intersection)
non_intersection_total = np.sum(~is_real_intersection)

if intersection_total > 0:
    recall_intersection = TP / intersection_total
else:
    recall_intersection = float('nan')

if non_intersection_total > 0:
    recall_non_intersection = TN / non_intersection_total
else:
    recall_non_intersection = float('nan')

# 输出
print("\n========= 📊 ~完整测试集评估结果 📊 =========")
print(f"✅ 总体识别准确率 (Accuracy): {accuracy * 100:.2f}%")
print(f"✅ 路口识别率 (Recall for 路口): {recall_intersection * 100:.2f}%")
print(f"✅ 非路口识别率 (Recall for 非路口): {recall_non_intersection * 100:.2f}%")
print(f"✅ 测试集中真实路口样本比例: {np.mean(is_real_intersection) * 100:.2f}%")
print(f"✅ 测试集中真实非路口样本比例: {np.mean(~is_real_intersection) * 100:.2f}%")
print("\n--------- 混淆矩阵结果 ---------")
print(f"True Positive (TP - 路口识别成路口): {TP}")
print(f"True Negative (TN - 非路口识别成非路口): {TN}")
print(f"False Positive (FP - 非路口识别成路口): {FP}")
print(f"False Negative (FN - 路口识别成非路口): {FN}")
print("===================================")
