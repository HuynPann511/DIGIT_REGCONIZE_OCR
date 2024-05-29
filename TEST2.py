# THIS FILE IS USING TO TEST PART READING DIGIT NUMBER OF DIGITAL
from ultralytics import YOLO
import cv2
import torch
from utils import read_digit, write_csv
import easyocr
import imutils

results = {}

#load models
model = YOLO("C:/Users/Admin/Desktop/New folder/best.pt")

#read image
image = cv2.imread("C:/Users/Admin/Desktop/New folder/6.jpg")
detects = model(image)[0]
detects_ = []

for detect in detects.boxes.data.tolist():
   x1, y1, x2, y2, score, class_id = detect
   if (class_id == 0): # class_id of Digital is 0
    detects_.append([x1, y1, x2, y2, score])

    # crop the digit of digital
    digit = image[int(y1):int(y2),int(x1):int(x2), :]
    digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(digit_gray, (5, 5), 0)
    cv2.imshow("blurred", blurred)
    edged = cv2.Canny(blurred, 150, 200, 255)
    cv2.imshow("edged", edged)
    _, digit_thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY_INV)

    digit_thresh = imutils.resize(digit_thresh, height=50)

    cv2.imshow("thresh",digit_thresh)
    
    # read digit numbers
    # digit_number, digit_score = read_digit(digit_thresh)
    text = ""
    digit_score = 0.0
    reader = easyocr.Reader(['en'], gpu=False)
    detections = reader.readtext(digit_thresh)
    
    if not detections:
        print("No detections found.")
    
    for detection in detections:
        print(detection)
        bbox, text, digit_score = detection


    print(f"Digit Number: {text}, Digit Score: {digit_score}")
    # if digit_number is not None:
    #   results[score] = {'Digital': {'bbox':[x1, y1, x2, y2],
    #                                  'value': digit_number,
    #                                  'bbox score': digit_score }}
    
# write the results
write_csv(results, 'C:/Users/Admin/Desktop/New folder/test.csv')

# print(digit_number)
# print(digit_score)
cv2.waitKey(0)
cv2.destroyAllWindows()