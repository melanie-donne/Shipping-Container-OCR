import cv2 # Import de la librairie OpenCV2
import numpy as np # Import de la librairie Numpy
import pytesseract # Import de la librairie PyTesseract
import imutils

img = cv2.imread('images/DSC_0067.JPG') # Chargement de l'image
img = cv2.resize(img, (620,480) ) # Resize de l'image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertion couleur vers nuance de gris
gray = cv2.bilateralFilter(gray, 13, 15, 15) # Floutage 
edged = cv2.Canny(gray, 30, 200)
#img= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)  Binarization adaptative

# Contours
contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:10]
# Tri du plus grand au plus petit et on ne considère que les 10 premiers résultats
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break
if screenCnt is None:
    detected = 0
    print ("No contour detected")
else:
     detected = 1
if detected == 1:
    cv2.drawContours(img, [screenCnt], -1, (0, 0, 255), 3)

# Masking
mask = np.zeros(gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)

# character Segmentation
# Segmentation du container en recadrant les caractères et en l'enregistrant en tant que nouvelle images
(x, y) = np.where(mask == 255)
(topx, topy) = (np.min(x), np.min(y))
(bottomx, bottomy) = (np.max(x), np.max(y))
Cropped = gray[topx:bottomx+1, topy:bottomy+1]

# Character recognition
conf = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6' # Configuration moteur OCR
text = pytesseract.image_to_string(Cropped, config=conf)
print("CONTAINER CODE RECOGNITIONS\n")
print("Detected container code is:",text)
img = cv2.resize(img,(500,300))
Cropped = cv2.resize(Cropped,(400,200))
cv2.imshow('container',img)
cv2.imshow('Cropped',Cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()