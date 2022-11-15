import pytesseract # Import de la librairie PyTesseract
import cv2 # Import de la librairie OpenCV2

conf = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6' # Configuration moteur OCR - psm: Specify page segmentation mode.

# Chargement de l'image
img = cv2.imread('images/DSC_0067.JPG')
assert not isinstance(img,type(None)), 'image not found'

# Preprocessing
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convertion couleur vers nuance de gris
img = cv2.GaussianBlur(img, (5, 5), 0) # Floutage 
img = cv2.resize(img, None, fx=1, fy=1) # Dilatation 
img= cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) # Binarization adaptative

# Exectution moteur PyTesseract OCR pour les caractères
print(pytesseract.image_to_string(img, config=conf))
cv2.imshow('Résultat1', img)
cv2.waitKey(0)


# Detection de ce qu'il y a autour
boxes = pytesseract.image_to_data(img, config = conf)
print(boxes)

cv2.imshow('Résultat2', img)
cv2.waitKey(0)



# chaineOCR = pytesseract.image_to_string(img, config=conf)  # Exectution moteur PyTesseract OCR







