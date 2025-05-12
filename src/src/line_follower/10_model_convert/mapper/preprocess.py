# Copyright (c) 2024 D-Robotics.All Rights Reserved.
#
# The material in this file is confidential and contains trade secrets
# of D-Robotics Inc. This is proprietary information owned by
# D-Robotics Inc. No part of this work may be disclosed,
# reproduced, copied, transmitted, or used in any way for any purpose,
# without the express written permission of D-Robotics Inc.

import sys
sys.path.append("../../../01_common/python/data/")
from transformer import *
from dataloader import *


def calibration_transformers():
    """
    step：
        1、resize to 224x224
        2、convert RGB to BGR
        3、NHWC to NCHW
    """
    transformers = [
        ResizeTransformer(target_size=(224, 224)),  # 调整尺寸到224x224
        HWC2CHWTransformer(),  # 从 HWC 转为 CHW
        BGR2RGBTransformer(),  # 转换颜色通道：BGR->RGB
    ]
    return transformers


def infer_transformers(input_layout="NHWC"):
    """
    step：
        1、resize to 224x224
        2、convert RGB to BGR
        3、NHWC to NCHW (if input_layout=NCHW)
    """
    transformers = [
        ResizeTransformer(target_size=(224, 224)),  # 调整尺寸到224x224
        BGR2RGBTransformer(data_format="HWC"),  # 转换颜色通道：BGR->RGB
    ]
    if input_layout == "NCHW":
        transformers.append(HWC2CHWTransformer())  # 如果是NCHW布局，转换通道顺序
    return transformers


def infer_image_preprocess(image_file, input_layout):
    """
    image for single image inference
    note: imread_mode [skimage / opencv]
        opencv read image as 8-bit unsigned integers BGR in range [0, 255]
        skimage read image as float32 RGB in range [0, 1]
        make sure to use the same imread_mode as the model training
    :param image_file: image path
    :param input_layout: NCHW / NHWC
    :return: processed image (uint8, 0-255)
    """
    transformers = infer_transformers(input_layout)
    image = SingleImageDataLoader(transformers,
                                  image_file,
                                  imread_mode='opencv')
    return image


def eval_image_preprocess(image_path, label_path, input_layout):
    """
    image for full scale evaluation
    note: imread_mode [skimage / opencv]
        opencv read image as 8-bit unsigned integers BGR in range [0, 255]
        skimage read image as float32 RGB in range [0, 1]
        make sure to use the same imread_mode as the model training
    :param image_path: image path
    :param label_path: label path
    :param input_layout: input layout
    :return: data loader, evaluation label size
    """
    transformers = infer_transformers(input_layout)
    data_loader = ImageNetDataLoader(transformers,
                                     image_path,
                                     label_path,
                                     imread_mode='opencv',
                                     batch_size=1,
                                     return_img_name=True)
    loader_size = sum(1 for line in open(label_path))
    return data_loader, loader_size

