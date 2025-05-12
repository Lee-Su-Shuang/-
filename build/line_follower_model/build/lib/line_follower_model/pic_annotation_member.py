import cv2 as cv
import uuid
import os
import numpy as np

class ImageProcessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.x = -1
        self.y = -1
        self.current_image = None
        self.original_image = None
        self.image_files = []
        self.current_index = 0
        self.annotation_round = 0  # 0-1 for 2 rounds
        self.annotation_counts = [0, 0]  # left, right counts
        self.round_names = [
            "left_t_junction",   # 左侧T字路口
            "right_t_junction"   # 右侧T字路口
        ]
        self.round_targets = [100, 100]  # 100 each

        # 检查目录
        if not os.path.exists(self.input_dir):
            raise FileNotFoundError(f"输入目录不存在: {self.input_dir}")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 获取图片文件
        self.image_files = sorted([f for f in os.listdir(self.input_dir) 
                                if f.lower().endswith(('png', 'jpg', 'jpeg'))])
        if not self.image_files:
            raise FileNotFoundError(f"目录中未找到图片文件: {self.input_dir}")

        # 初始化窗口
        cv.namedWindow("Road Annotation", cv.WINDOW_NORMAL)
        cv.resizeWindow("Road Annotation", 640, 224)

    def mouse_callback(self, event, x, y, flags, userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            display_image = self.original_image.copy()
            self.x = x
            self.y = y
            cv.circle(display_image, (x, y), 5, (0, 0, 255), -1)
            
            # 显示进度信息
            progress_text = f"{self.round_names[self.annotation_round]} ({self.annotation_counts[self.annotation_round]}/{self.round_targets[self.annotation_round]})"
            cv.putText(display_image, progress_text, (10, 30), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv.imshow("Road Annotation", display_image)

    def load_next_image(self):
        if not self.image_files:
            return False
            
        file_path = os.path.join(self.input_dir, self.image_files[self.current_index])
        print(f"正在处理: {file_path}")
        
        full_image = cv.imread(file_path)
        if full_image is None:
            print(f"读取失败: {file_path}")
            return False
            
        # 裁剪为640x224
        self.original_image = full_image[128:352, :, :].copy()
        
        # 显示图像（添加临时标注信息）
        display_image = self.original_image.copy()
        progress_text = f"{self.round_names[self.annotation_round]} ({self.annotation_counts[self.annotation_round]}/{self.round_targets[self.annotation_round]})"
        cv.putText(display_image, progress_text, (10, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv.setMouseCallback("Road Annotation", self.mouse_callback, display_image)
        cv.imshow("Road Annotation", display_image)
        return True

    def save_annotation(self):
        """确保生成唯一文件名后保存"""
        max_attempts = 10  # 最大尝试次数
        for _ in range(max_attempts):
            # 生成标准文件名
            filename = f"{self.round_names[self.annotation_round]}_{self.x:03d}_{self.y:03d}_{uuid.uuid4().hex[:8]}.jpg"
            output_path = os.path.join(self.output_dir, filename)
            
            # 检查文件是否已存在
            if not os.path.exists(output_path):
                cv.imwrite(output_path, self.original_image)
                print(f"已保存: {output_path}")
                return True
            
            print(f"检测到重名文件，重新生成: {filename}")
        
        print("错误: 无法生成唯一文件名，请手动清理目标目录")
        return False

    def process_images(self):
        while self.annotation_round < 2:  # Only 2 rounds now
            print(f"\n===== 第 {self.annotation_round+1}/2 轮: {self.round_names[self.annotation_round]} =====")
            print(f"目标数量: {self.round_targets[self.annotation_round]}")
            
            while self.annotation_counts[self.annotation_round] < self.round_targets[self.annotation_round]:
                if not self.load_next_image():
                    break
                    
                key = cv.waitKey(0)
                
                # 空格/回车确认标注
                if key in (32, 13):  
                    if self.x != -1 and self.y != -1:
                        if self.save_annotation():
                            self.annotation_counts[self.annotation_round] += 1

                    # 循环使用图片
                    self.current_index = (self.current_index + 1) % len(self.image_files)
                    self.x = -1
                    self.y = -1
                    
                # ESC跳过当前类型
                elif key == 27:  
                    print(f"跳过 {self.round_names[self.annotation_round]}")
                    break
            
            # 进入下一轮标注
            self.annotation_round += 1
            self.current_index = 0
            self.x = -1
            self.y = -1

        cv.destroyAllWindows()
        print("\n标注完成! 统计结果:")
        for i in range(2):
            print(f"{self.round_names[i]}: {self.annotation_counts[i]}/{self.round_targets[i]}")

if __name__ == '__main__':
    input_dir = "./input_images"    # 原始图片目录
    output_dir = "./image_dataset"  # 输出目录
    
    try:
        processor = ImageProcessor(input_dir, output_dir)
        processor.process_images()
    except Exception as e:
        print(f"错误: {str(e)}")
