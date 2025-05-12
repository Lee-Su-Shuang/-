# Copyright (c) 2024, D-Robotics.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torchvision.models as models

def main():
    # --------- 1. 初始化 MobileNetV3-Small ---------
    model = models.mobilenet_v3_small(weights=None)
    
    # 修改分类头，输出2个数（x, y）
    model.classifier[3] = torch.nn.Linear(in_features=1024, out_features=2)
    
    # --------- 2. 加载训练好的权重 ---------
    model_weights_path = './best_line_follower_model_xy.pth'
    model.load_state_dict(torch.load(model_weights_path, map_location='cpu'))
    
    model.eval()
    model.cpu()
    
    # --------- 3. 创建虚拟输入 ---------
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # --------- 4. 导出为 ONNX 文件 ---------
    torch.onnx.export(
        model, 
        dummy_input, 
        './best_line_follower_model_xy.onnx',
        export_params=True,        # 保存训练好的参数
        opset_version=11,           # ONNX算子版本
        do_constant_folding=True,   # 常量折叠优化
        input_names=['input'], 
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'}, 
            'output': {0: 'batch_size'}
        }
    )
    print("✅ Successfully exported MobileNetV3 model to ONNX!")

if __name__ == '__main__':
    main()

