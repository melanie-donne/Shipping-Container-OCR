from PIL import Image
import pytesseract
import numpy as np
import cv2
import re

def readImg(img):
	#set path to tesseract
	pytesseract.pytesseract.tesseract_cmd = 'tesseract'
	
	#initial bilateralFilter parameters
	d = 7
	sigmaColor = 150
	sigmaSpace = 115
	
	#get text from image
	text = extract(img, d, sigmaColor, sigmaSpace)
	
	#check format of string
	count = 0
	correct = check(text)
	while(correct == False):
		if(count == 0):
			#modify settings bilateralFilter parameters for cleaning the image to crop
			d = 7
			sigmaColor = 50
			sigmaSpace = 50
			
			#try to extract text again 
			text = extract(img, d, sigmaColor, sigmaSpace)
			
			#check if text is correct
			correct = check(text)
			count += 1
			
			
		elif(count == 1):
			#modify settings bilateralFilter parameters
			d = 6
			sigmaColor = 50
			sigmaSpace = 50
			
			#try to extract text again 
			text = extract(img, d, sigmaColor, sigmaSpace)
			
			#check if text is correct
			correct = check(text)
			count += 1
			
			
		elif(count == 2):
			#modify settings bilateralFilter parameters
			d = 9
			sigmaColor = 75
			sigmaSpace = 75
			
			#try to extract text again
			text = extract(img, d, sigmaColor, sigmaSpace)
			
			#check if text is correct
			correct = check(text)
			count += 1
			
		
		elif(count == 3):
                    #TO DO: compare partial text against database of labels to try to find match
                    failed = "UnreadableContainer"
                    return failed, count
                    break
			
	if(correct == True):		
		return text, count


	


def extract(img, d, sigmaColor, sigmaSpace):
    
    #denoise original img
    cleanIMG = cv2.fastNlMeansDenoisingColored(img)
    
    #convert image to grayscale and blur it
    gray = cv2.cvtColor(cleanIMG, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)
    cv2.imshow("gray", gray)
    cv2.waitKey(0)

    #detect edges in the image
    #this is for finding the bounded rectangle in order to crop
    edged = cv2.Canny(gray, 10, 250)
    edgeClean = cv2.bilateralFilter(edged, 11, 50, 50)
    
    cv2.imshow("edged", edged)
    cv2.waitKey(0)
    
    #find contours
    edgeClean, contours, hierarchy = cv2.findContours(edgeClean.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    cIMG = img.copy()
    #draw contours
    cv2.drawContours(cIMG, contours, -1, (0, 0, 255), 2)
    cnt = contours[4]
    cv2.imshow('contours', cIMG)
    cv2.waitKey(0)

    #find extreme points
    x, y = [], []
    for line in contours:
        for c in line:
            x.append(c[0][0])
            y.append(c[0][1])
            
    x1, x2, y1, y2 = min(x), max(x), min(y), max(y)
    
    #crop image
    cropped = img[y1:y2, x1:x2]
    
    #clean cropped image
    cleanCropped = cv2.fastNlMeansDenoisingColored(cropped)
    
    #convert image to grayscale and blur it
    gCropped = cv2.cvtColor(cleanCropped, cv2.COLOR_BGR2GRAY)
    gCropped = cv2.bilateralFilter(gCropped, 7, 100, 100)
    
    cv2.imshow('gCropped', gCropped)
    cv2.waitKey(0)
    
    #save modified img
    cv2.imwrite('label.png', gCropped)   
    
    
    #get text from image and clean string
    text = pytesseract.image_to_string(Image.open('label.png'))
    text = text.replace(" ", "")
    cleanedText = text.split("/")
    cleanedText = cleanedText[0].split("|")
    cleanedText = cleanedText[0].split("I")
    text = cleanedText[0].split("\n")[0]
    
    #strip string of any non alphanumeric characters
    text = re.sub('[^0-9a-zA-Z]+', '', text)
    print(text)
    return(text)

	
def check(text):
	#check to see if text follows regex
	regexp1 = re.compile('^[A-Z0-9]{8}UPS$')
	regexp2 = re.compile('^[A-Z0-9]{7}UPS$')
	if regexp1.search(text):
	    return True
	elif regexp2.search(text):
            return True
	else:
	    return False