import cv2 # Import de la librairie OpenCV2
import numpy as np # Import de la librairie Numpy
import pytesseract # Import de la librairie PyTesseract
import time
import pandas as pd


#pytesseract.pytesseract.tesseract_cmd = '/home/melanie/anaconda3/pkgs/pytesseract-0.3.10-py310h06a4308_0/bin/pytesseract'  #Chemin moteur OCR
conf = r'-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 --psm 6' # Configuration moteur OCR - psm: Specify page segmentation mode.
 
# Chargement de YOLO
net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg") # Poids des synapses + Fichier de configuration

# Liste des objets à detecter
classes = ["OCR_TAG"]

#Configuration de YOLO
layer_names = net.getLayerNames()
output_layers = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]


# Chemin de l'image
img_path="images/DSC_0067.JPG"

# Chargement de l'image
img = cv2.imread(img_path)
assert not isinstance(img,type(None)), 'image not found'
height, width, channels = img.shape # Dimension de la matrice de l'image


# Détection d'objet
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Affichage des informations sur une image + OCR
class_ids = []
confidences = []
boxes = []
for out in outs:

    for detection in out:

        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.3:

            # Objet detecté
            center_x = int(detection[0] * width) # Centre X de l'objet détecté
            center_y = int(detection[1] * height) # Centre Y de l'objet détecté
            w = int(detection[2] * width) # Largeur de l'objet détecté
            h = int(detection[3] * height) # Hauteur de l'objet détecté

            # Coordonnées coin supérieur droit du cadre
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.7, 0.3) #0.7 = Probabilité seuil / Niveau de confiance seuil

for i in range(len(boxes)):
    if i in indexes:
        #OCR
        x, y, w, h = boxes[i] #Dimension et Position des TAG OCR
        img_tag_OCR=img[y-10:y + h+10, x-10:x + w + 10] #Redimensionement des TAG OCR
        cv2.imshow("Tag Gray: " + str(i), img_tag_OCR) # Affichage des TAG OCR
        img_tag_OCR = cv2.cvtColor(img_tag_OCR, cv2.COLOR_BGR2GRAY) # Convertion couleur vers nuance de gris
        img_tag_OCR = cv2.GaussianBlur(img_tag_OCR, (1, 1), 0) # Floutage des TAG OCR
        (_, img_tag_OCR) = cv2.threshold(img_tag_OCR, 127, 255, cv2.THRESH_BINARY_INV) # Binarization TAG OCR
        img_tag_OCR = cv2.resize(img_tag_OCR, None, fx=1, fy=1) # Dilatation des TAG OCR
        chaineOCR = pytesseract.image_to_string(img_tag_OCR, config=conf)  # Exectution moteur PyTesseract OCR
        chaineOCR = ''.join(e for e in chaineOCR if e.isalnum()) # Jonction des caratères
        print("-------")
        print(i) # Numéro de TAG OCR trouvé
        print(chaineOCR) # TAG OCR lu

        # Les datas sont stockées dans un fichier CSV
        raw_data = {'Date': [time.asctime( time.localtime(time.time()) )],'Numéro container': [chaineOCR]}
        df = pd.DataFrame(raw_data, columns = ['date', 'v_number']) # On ajoute le nom des colomnes
        df.to_csv('data.csv') # Envoie dans le csv
        
        cv2.putText(img, chaineOCR, (100,100), cv2.FONT_HERSHEY_PLAIN, 6, (220,20,60), 6) # On écrit le résultat sur l'image

        #Affichage
        label = str(classes[class_ids[i]])#Etiquette de l'objet détecter
        cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 4) # Placer le cadre de délimitation
        cv2.rectangle(img, (x, y - 55), (x + w, y),(0,0,255) , -1) # Placer un rectangle au dessus de chaque cadre de délimitation
        cv2.putText(img, label, (x+5, y - 20), cv2.FONT_HERSHEY_SIMPLEX,1, (255,255,255), 3) # Placer Etiquette

img_r=cv2.resize(img,(int(img.shape[1]*480/img.shape[0]),480))#Redimensionement de l'image

cv2.imshow("Image", img_r) # Affichage de l'image rognée
key = cv2.waitKey(0) # Attends l'appuie d'une touche pour exectuter les ligne suivantes
cv2.destroyAllWindows() # Fermeture de toutes les images


