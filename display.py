import ast

import cv2
import numpy as np
import pandas as pd

def display_digit_number(image, x1, y1, x2, y2, digit_number):
    """
    Hiển thị digit_number lên hình ảnh.

    Args:
        image (numpy.ndarray): Hình ảnh đầu vào.
        x1, y1, x2, y2 (int): Tọa độ của bounding box.
        digit_number (str): Số nhận diện được.

    Returns:
        numpy.ndarray: Hình ảnh đã được hiển thị số.
        ----------------------------------------------------------------
    """
    # Vị trí để hiển thị digit_number
    org = (int(x1), int(y1) - 5)  # Hiển thị phía trên bounding box 
    # Việc trừ 10 từ y1 nhằm mục đích hiển thị văn bản ngay phía trên bounding box 
    # mà không che khuất đối tượng trong bounding box
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (0, 255, 0)  # Màu xanh lá cây
    thickness = 2

    # display bbox
    image = cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    # Hiển thị digit_number lên ảnh
    image = cv2.putText(image, digit_number, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return image

