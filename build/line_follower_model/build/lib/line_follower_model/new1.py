import os
import re
import uuid
import cv2
import shutil

def rename_images(input_dir):
    # 创建临时目录用于安全操作
    temp_dir = os.path.join(input_dir, 'temp_rename')
    os.makedirs(temp_dir, exist_ok=True)
    
    # 定义文件名模式 - 匹配 type_x_y_uuid.jpg 格式
    pattern = re.compile(r'^(.+)_(\d+)_(\d+)_([a-f0-9]+)\.jpg$', re.IGNORECASE)
    
    for filename in os.listdir(input_dir):
        if not filename.lower().endswith('.jpg'):
            continue
            
        src_path = os.path.join(input_dir, filename)
        
        try:
            # 尝试从文件名提取信息
            match = pattern.match(filename)
            if not match:
                print(f"跳过：{filename} (不符合命名格式)")
                continue
                
            # 提取坐标和原有UUID
            _, x, y, old_uuid = match.groups()
            x = int(x)
            y = int(y)
            
            # 生成新文件名 (保留原坐标，只修改前缀和UUID)
            max_attempts = 10
            for _ in range(max_attempts):
                new_filename = f'xy_{x:03d}_{y:03d}_{uuid.uuid4().hex[:8]}.jpg'
                temp_path = os.path.join(temp_dir, new_filename)
                
                if not os.path.exists(os.path.join(input_dir, new_filename)):
                    # 复制到临时目录(确保安全)
                    shutil.copy2(src_path, temp_path)
                    
                    # 验证图片可读
                    img = cv2.imread(temp_path)
                    if img is not None:
                        # 移动回原目录
                        final_path = os.path.join(input_dir, new_filename)
                        shutil.move(temp_path, final_path)
                        os.remove(src_path)
                        print(f"重命名成功: {filename} -> {new_filename}")
                        break
                    else:
                        os.remove(temp_path)
                        raise ValueError("图片损坏")
            else:
                print(f"无法为 {filename} 生成唯一文件名")
                
        except Exception as e:
            print(f"处理 {filename} 失败: {str(e)}")
    
    # 清理临时目录
    shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == '__main__':
    image_dir = './image_dataset_temp'
    
    if not os.path.exists(image_dir):
        print(f"错误：目录不存在 {image_dir}")
        exit(1)
        
    print("=== 开始重命名图片 ===")
    print("规则: 从原文件名提取x,y坐标，格式化为 xy_XXX_YYY_NEWUUID.jpg")
    rename_images(image_dir)
    print("=== 处理完成 ===")
