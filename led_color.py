import cv2
from ultralytics import YOLO
import numpy as np


image_path = r'C:\\Users\\Admin\\Desktop\\Projectbh\\NEW_DATA\\69.jpg'
model = YOLO("best.pt")
color = (0, 255, 0)  # Màu xanh lá cây


def identify_led_colors(image_path):
    
    image = cv2.imread(image_path)
    detects = model(image)[0]
    detects_ = []

    for detect in detects.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detect
        if (class_id == 2): # class_id of led is 2
            detects_.append([x1, y1, x2, y2, score])

    cv2.imshow('original',image)
    if detects is None:
        print("Error: detect not found!")
        return

    led = image[int(y1):int(y2),int(x1):int(x2), :]
    cv2.imshow("crop",led)
  
    # pre-processing image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (15, 15), 0)
    
    _, thresholded_image = cv2.threshold(blurred_image, 200, 255, cv2.THRESH_BINARY)

    
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    
    color_ranges = {
        'red': [(0, 100, 100), (10, 255, 255)],
        'yellow': [(20, 100, 100), (30, 255, 255)],
        'green': [(40, 100, 100), (70, 255, 255)],
        'cyan': [(80, 100, 100), (90, 255, 255)],
        'blue': [(100, 100, 100), (130, 255, 255)],
        # 'magenta': [(140, 100, 100), (160, 255, 255)],
        # 'white': [(0, 0, 200), (180, 20, 255)],
        # 'black': [(0, 0, 0), (180, 255, 30)]
    }

    detected_colors = []

    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        led_region = hsv_image[y:y+h, x:x+w]
        
        
        for color_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype=np.uint8)
            upper = np.array(upper, dtype=np.uint8)
            mask = cv2.inRange(led_region, lower, upper)
            intensity = cv2.countNonZero(mask)
            
            if intensity > 0:
                detected_colors.append(color_name)
                
                
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)
                
                
                cv2.putText(image, color_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                break
    
    if detected_colors:
        print(f"Detected colors: {', '.join(detected_colors)}")
    else:
        print("No colors detected.")

    
    copy_image = image.copy()
    copy_image_path = image_path.split('.')
    copy_image_path[-2] += "_copy"
    copy_image_path = '.'.join(copy_image_path)
    cv2.imwrite(copy_image_path, copy_image)
    cv2.imshow('show',copy_image)
    cv2.waitKey(0)

    print(f"Image with LED colors added saved as: {copy_image_path}")


identify_led_colors(image_path)
