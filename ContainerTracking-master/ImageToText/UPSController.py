import ExtractText
import time
import cv2
import numpy
import os

#custom code imports
from DatabaseController import DatabaseController, DatabaseCredentials, Container
from ContainerAlert import alertPopUp


def main():
    start_time = time.time()
    picCommand = "fswebcam -r 352x288 -S 20 --no-banner image.jpg"
    os.system(picCommand)
    img = cv2.imread("image.jpg")
    takePictureTime = str(round(time.time() - start_time, 2))
    
    #extract text from label
    start_extract = time.time()
    label, count = ExtractText.readImg(img)
    end_extract = time.time()
    extract_time = str(round(end_extract - start_extract, 2))
    
    cameraid = getMAC()
    print(cameraid)
    
    #Future update: Change these to be stored in a config file, preferably encrypted
    host = "localhost"
    db = "upsdemo"
    user = "upsuser"
    passwd = "upsuser"
    
    #create an object with the database credentials
    dbCreds = DatabaseCredentials(host, db, user, passwd)
    #create the DBController
    dbCont = DatabaseController(dbCreds)
    
    dbCont.updateContainer(label, cameraid)
    container = dbCont.retrieveContainer(label)
    dbCont.endConnection()
    
    alertPopUp(container)
    
    
    
    finaltime = round((time.time() - start_time), 2)
    
    

# Return the MAC address of the specified interface
def getMAC(interface='eth0'):
  try:
    str = open('/sys/class/net/%s/address' %interface).read()
  except:
    str = "00:00:00:00:00:00"
  return str[0:17]

if __name__ =='__main__':
    main()