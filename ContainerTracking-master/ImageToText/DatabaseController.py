#Imports go here as needed
import MySQLdb
from string import Template

class DatabaseCredentials:

    def __init__(self, host, db, user, passwd):
        self.passwd = passwd
        self.host = host
        self.user = user
        self.db = db

class DatabaseController:

    selectContainer = Template("SELECT * FROM CONTAINERS WHERE ID = '$contID';")
    updateContain = Template("UPDATE CONTAINERS SET CAMERA_ID = '$camID' WHERE ID = '$contID';")


    def __init__(self, credentials):
        self.dbObj = MySQLdb.connect(host=credentials.host,
                                     user=credentials.user,
                                     passwd=credentials.passwd,
                                     db=credentials.db)

    def retrieveContainer(self, containerID):
        cur = self.dbObj.cursor()
        dbCommand = self.selectContainer.substitute(contID=containerID)
        success = cur.execute(dbCommand)

        if(success >= 1):
            print("successfully pulled from DB")
            row = cur.fetchall()[0]
            container = Container(row[0], row[1], row[2], row[3], row[4])
        else:
            container = 1
        cur.close()
        return container

    def updateContainer(self, containerID, cameraID):
        cur = self.dbObj.cursor()
        dbCommand = self.updateContain.substitute(camID=cameraID, contID=containerID)
        success = cur.execute(dbCommand)

        if(success >=1):
            print("updated container with Camera id")
            self.dbObj.commit()
            for row in cur.fetchall():
                for cell in row:
                    print (cell)
        cur.close()

    def endConnection(self):
        self.dbObj.close()


class Container:
    def __init__(self, containerID, fragile, priority, content, camID):
        self.camID = camID
        self.containerID = containerID
        if(fragile == 1):
            self.fragile = True
        else:
            self.fragile = False
        self.priority = priority
        self.content = content