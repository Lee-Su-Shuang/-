# generate_onnx.py

import torch
import torch.nn as nn

# --- 小型CNN定义（要和训练时的一模一样）
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

# --- 导出函数
def generate_onnx():
    # 初始化模型
    model = SmallCNN()
    
    # 加载训练好的权重
    model_weights_path = './best_line_follower_model_xy.pth'
    model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    
    model.eval()
    model.cpu()
    
    # 创建虚拟输入
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # 导出为 ONNX
    output_onnx_path = './best_line_follower_model_xy.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        output_onnx_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✅ Successfully exported to {output_onnx_path}")

if __name__ == '__main__':
    generate_onnx()

