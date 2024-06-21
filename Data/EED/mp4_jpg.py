import os
import cv2

file_path = '/Users/zhanghan/Desktop/1004_MTI_NEU_XX.flv_0001.jpg'
image = cv2.imread(file_path)
# 设置输入和输出目录
input_dir = '/Users/zhanghan/Desktop/EmpathEar_example/mp4'
output_dir = '/Users/zhanghan/Desktop/EmpathEar_example/jpg'

# 确保输出目录存在
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 遍历输入目录中的所有MP4文件
for filename in os.listdir(input_dir):
    if filename.endswith('.mp4'):
        # 构建输入和输出路径
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + '.jpg')

        # 使用 OpenCV 读取第一帧
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()

        # 保存第一帧为 JPEG 图像
        cv2.imwrite(output_path, frame)

        # 释放 VideoCapture 对象
        cap.release()

print('图片提取完成!')