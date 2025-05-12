import torch
import numpy as np
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import glob
import os
import PIL.Image

# --------- 1. 解析像素坐标，不归一化 ---------
def get_x_pixel(filename):
    parts = filename.split("_")
    return int(parts[1])

def get_y_pixel(filename):
    parts = filename.split("_")
    return 224 - int(parts[2])  # 反转y轴

# --------- 2. 自定义Dataset ---------
class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path).convert('RGB')
        
        x_pixel = get_x_pixel(os.path.basename(image_path))
        y_pixel = get_y_pixel(os.path.basename(image_path))
        
        # 图像预处理
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(
            image, 
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
        
        return image, torch.tensor([x_pixel, y_pixel]).float()

# --------- 3. 加载数据 ---------
dataset = XYDataset('./image_dataset')

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

# --------- 4. 加载模型 ---------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(512, 2)  # 输出两个数 (x, y)
model.load_state_dict(torch.load('./best_line_follower_model_xy.pth', map_location=device))
model.to(device)
model.eval()
# --------- 5. 测试并计算准确率 ---------
threshold_pixel = 214  # y < 10 算路口
all_y_true = []
all_y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)

        y_true_batch = labels[:, 1].cpu().numpy()          # 真实y（像素）
        y_pred_batch = outputs[:, 1].cpu().numpy() * 112 + 112  # 反归一化回像素

        all_y_true.append(y_true_batch)
        all_y_pred.append(y_pred_batch)

# 整合所有batch
all_y_true = np.concatenate(all_y_true)
all_y_pred = np.concatenate(all_y_pred)

# 用 all_y_true 和 all_y_pred 来算
is_real_intersection = (all_y_true > threshold_pixel)   # 真实是路口
is_pred_intersection = (all_y_pred > threshold_pixel)   # 预测是路口

TP = np.sum(is_real_intersection & is_pred_intersection)
TN = np.sum(~is_real_intersection & ~is_pred_intersection)
FP = np.sum(~is_real_intersection & is_pred_intersection)
FN = np.sum(is_real_intersection & ~is_pred_intersection)

accuracy = (TP + TN) / (TP + TN + FP + FN)

# 避免除以0
intersection_total = np.sum(is_real_intersection)
non_intersection_total = np.sum(~is_real_intersection)

if intersection_total > 0:
    recall_intersection = TP / intersection_total
else:
    recall_intersection = float('nan')  # 无真实路口

if non_intersection_total > 0:
    recall_non_intersection = TN / non_intersection_total
else:
    recall_non_intersection = float('nan')  # 无真实非路口

# --------- 输出 ---------
print("\n========= 📊 完整测试集评估结果 📊 =========")
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
