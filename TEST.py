from ultralytics import YOLO
import cv2
import torch
from utils import read_digit, write_csv

results = {}

#load models
model = YOLO("C:/Users/Admin/Desktop/New folder/best.pt")

#read image
image = cv2.imread("C:/Users/Admin/Desktop/New folder/75.png")
detects = model(image)[0]
detects_ = []

for detect in detects.boxes.data.tolist():
   x1, y1, x2, y2, score, class_id = detect
   if (class_id == 0): # class_id of Digital is 0
    detects_.append([x1, y1, x2, y2, score])

    # crop the digit of digital
    digit = image[int(y1):int(y2),int(x1):int(x2), :]
    digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    _, digit_thresh = cv2.threshold(digit_gray, 49, 255, cv2.THRESH_BINARY_INV)

    # read digit numbers
    digit_number, digit_score = read_digit(digit_thresh)
   
    if digit_number is not None:
      results[score] = {'Digital': {'bbox':[x1, y1, x2, y2],
                                     'value': digit_number,
                                     'bbox score': digit_score }}
    
# write the results
write_csv(results, 'C:/Users/Admin/Desktop/New folder/test.csv')

cv2.imshow("Detection", image)
cv2.destroyAllWindows()