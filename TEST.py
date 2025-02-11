from ultralytics import YOLO
import cv2
# import torch
# import imutils
from utils import read_digit, write_csv
from display import display_digit_number

results = {}

#load models
model = YOLO("best.pt")

#read image
image = cv2.imread("E:/DO AN 1/PROJECT/DIGIT_REGCONIZE/DIGIT_REGCONIZE_OCR/image/50.jpg")
cv2.imshow("anh_goc", image)
detects = model(image)[0]
detects_ = []

for detect in detects.boxes.data.tolist():
  x1, y1, x2, y2, score, class_id = detect
  if (class_id == 0): # class_id of Digital is 0
    detects_.append([x1, y1, x2, y2, score])

    # crop the digit of digital
    digit = image[int(y1):int(y2),int(x1):int(x2), :]
    cv2.imshow("crop",digit)
    digit_gray = cv2.cvtColor(digit, cv2.COLOR_BGR2GRAY)
   
    # modify the image  
    blurred = cv2.GaussianBlur(digit_gray, (5, 5), 0)

    # Adjusting contrast
    alpha = 3.5  # Hệ số tăng cường độ tương phản (lớn hơn 1 để tăng cường độ tương phản)
    beta = -200   # Giá trị điều chỉnh độ sáng (giá trị âm để giảm cường độ pixel sáng)
    adjusted_image = cv2.convertScaleAbs(blurred, alpha=alpha, beta=beta)
    cv2.imshow('adjusted',adjusted_image)

    cv2.imshow("blurred", blurred)
    edged = cv2.Canny(adjusted_image, 45, 220, 255)
    cv2.imshow("edged", edged)
    _, digit_thresh = cv2.threshold(edged, 70, 255, cv2.THRESH_BINARY_INV)
    
    # digit_thresh = imutils.resize(digit_thresh, height=55)
    cv2.imshow("after_cut", digit_thresh)


    # read digit numbers
    digit_number, digit_score = read_digit(adjusted_image)
    print(f"Digit Number: {digit_number}, Digit Score: {digit_score}")
    
    # Display digit number on image
    if digit_number is not None:
      image = display_digit_number(image, x1, y1, x2, y2, digit_number)
    cv2.imshow("original", image)

    # Take the result into a dictionary (var)
    if digit_number is not None:
      results[score] = {'Digital': {'bbox':[x1, y1, x2, y2],
                                     'value': digit_number,
                                     'bbox score': digit_score }}
    
# write the results to file exel (csv file)
    # write_csv(results, 'C:/Users/Admin/Desktop/New folder/test.csv')

cv2.waitKey(0)  
cv2.destroyAllWindows()
