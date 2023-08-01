# def cross_product(p1, p2, p3):
#     # 计算向量 (p2 - p1) 和 (p3 - p1) 的叉积
#     return (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
#
# def point_in_quadrilateral(A, B, C, D, E):
#     # 判断点 A 是否在四边形 BCDE 内部
#     cross1 = cross_product(B, C, A)
#     cross2 = cross_product(C, D, A)
#     cross3 = cross_product(D, E, A)
#     cross4 = cross_product(E, B, A)
#
#     if (cross1 >= 0 and cross2 >= 0 and cross3 >= 0 and cross4 >= 0) or (cross1 <= 0 and cross2 <= 0 and cross3 <= 0 and cross4 <= 0):
#         return True
#     else:
#         return False
#
# # 示例数据：定义点的坐标 (x, y)
# A = (6, 2)
# B = (0, 0)
# C = (4, 0)
# D = (4, 5)
# E = (0, 5)
#
# # 判断点 A 是否在四边形 BCDE 内部
# result = point_in_quadrilateral(A, B, C, D, E)
# print(result)  # 输出 True 或 False


import cv2
import numpy as np

# def fill_grayscale_outside_region(image, specified_points, grayscale_fill):
#     # 获取图像尺寸
#     height, width, _ = image.shape
#
#     # 创建一个掩码（mask）并将其设置为全黑
#     mask = np.zeros((height, width), dtype=np.uint8)
#
#     # 将指定区域内的像素点设置为白色（值为255）
#     cv2.fillPoly(mask, [np.array(specified_points)], 255)
#
#     # 将掩码取反，使指定区域外的像素值为白色（值为255），指定区域内的像素值为黑色（值为0）
#     mask_inverse = cv2.bitwise_not(mask)
#
#     # 创建一个与图像相同尺寸的灰度填充图像
#     filled_image = np.full((height, width, 3), grayscale_fill, dtype=np.uint8)
#
#     # 将指定区域内的像素从原始图像复制到灰度填充图像中
#     filled_image[mask == 255] = image[mask == 255]
#
#     return filled_image
#
# if __name__ == "__main__":
#     input_image_path = "input_image.jpg"  # 替换成您的输入图像路径
#     specified_points = [(100, 100), (200, 150), (300, 200), (250, 300)]  # 替换成指定区域的点列表
#     grayscale_fill_color = (128, 128, 128)  # 替换成要用于灰度填充的颜色，范围在0-255之间的RGB元组
#
#     # 假设您的输入图片是一个形状为(1, 480, 640, 3)的NumPy数组
#     input_image = np.random.randint(0, 256, size=(1, 480, 640, 3), dtype=np.uint8)
#
#     # 提取单张图片
#     input_image = input_image[0]
#
#     filled_image = fill_grayscale_outside_region(input_image, specified_points, grayscale_fill_color)
#
#     # 显示灰度填充后的图像

# for c_id, (f_time, l_time, t_time) in list(vehicle_info.items()):
#     elapsed_time = l_time - f_time
#     if elapsed_time > parking_duration:
#         if t_time == 0:
#             print(f"车辆 {c_id} 超时，持续时间：{elapsed_time}秒")
#         t_time += elapsed_time
#         vehicle_info[c_id] = (f_time, l_time, t_time)
#     if time_sync() - l_time > disappearance_time:
#         if t_time > 0:
#             print(f"车辆 {c_id} 消失超过 {2} 秒，总超时时长：{t_time}秒")
#         vehicle_info.pop(c_id)

# import setproctitle
# import sys
#
# if __name__ == "__main__":
#     new_process_name = "My Custom Process Name"  # 设置自定义的进程名称
#     setproctitle.setproctitle(sys.argv[1])
#     while 1:
#         print(sys.argv)


# import setproctitle
# import sys
#
# if __name__ == "__main__":
#     setproctitle.setproctitle(sys.argv[2])
#     while 1:
#         print(sys.argv)






