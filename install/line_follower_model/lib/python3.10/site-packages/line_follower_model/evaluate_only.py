import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import torchvision.transforms as transforms
import glob
import os
import PIL.Image

# 你的 XYDataset 类
def get_x(path, width):
    return (float(int(path.split("_")[1])) * 224.0 / 640.0 - width/2) / (width/2)

def get_y(path, height):
    return ((224 - float(int(path.split("_")[2]))) - height/2) / (height/2)

class XYDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.image_paths = glob.glob(os.path.join(self.directory, '*.jpg'))
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = PIL.Image.open(image_path)
        x = float(get_x(os.path.basename(image_path), 224))
        y = float(get_y(os.path.basename(image_path), 224))
        
        image = transforms.functional.resize(image, (224, 224))
        image = transforms.functional.to_tensor(image)
        image = transforms.functional.normalize(image, 
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        return image, torch.tensor([x, y]).float()

# 加载测试集
dataset = XYDataset('./image_dataset')
test_percent = 0.1
num_test = int(test_percent * len(dataset))
_, test_dataset = torch.utils.data.random_split(dataset, [len(dataset) - num_test, num_test])

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# 加载模型
device = torch.device('cpu')
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = torch.nn.Linear(512, 2)
model.load_state_dict(torch.load('./best_line_follower_model_xy.pth', map_location=device))
model.to(device)
model.eval()

# 画图
preds = []
trues = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds.append(outputs.cpu().numpy())
        trues.append(labels.cpu().numpy())

preds = np.concatenate(preds, axis=0)
trues = np.concatenate(trues, axis=0)

plt.figure(figsize=(8, 8))
plt.scatter(trues[:, 0], trues[:, 1], color='blue', label='True', alpha=0.5)
plt.scatter(preds[:, 0], preds[:, 1], color='red', label='Predicted', alpha=0.5)
plt.legend()
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('Predicted vs True Coordinates on Test Set')
plt.grid()
plt.show()

