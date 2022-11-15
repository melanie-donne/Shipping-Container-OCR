from DatabaseController import Container
from tkinter import Tk, Label, Button



def alertPopUp(container):
    if(container == 1):
        print("Error finding a container")
        return 0

    message = messageBuilder(container)
    root = Tk()
    containerPopUp = PopUp(root, message)
    root.mainloop()

def messageBuilder(container):
    #This is obviously very rough and is a stand in for something DB backed
    message = []
    message.append("Container: " + container.containerID + "\n")
    message.append("Contents: " + getContentsString(container.content) + "\n")

    #Add priority to message
    message.append("Priority: ")
    if((container.content in [1,2,6]) and (container.priority in [1,2])):
        message.append("Highest Priority,")
        if(container.content == 6):
            message.append(" unreadable container\n")
        else:
            message.append(" Easily spoiled contents\n")
    elif(container.content == 3):
        message.append("Low priority, no immediate action needed\n")
    elif(container.content == 4):
        message.append("Part of a larger order, wait for other containers\n")
    elif(container.content == 1):
        message.append("New shipment\n")
    elif(container.content == 2):
        message.append("High priority shipment\n")
    else:
        message.append("Special shipment\n")

    notes = []
    if(container.fragile):
        notes.append("Fragile shipment, please be gentle\n")
    if(container.content in {1,3}):
        notes.append("Temperature sensitive\n")

    if(len(notes) > 1):
        message.appened("\nContainer Notes:\n")
        for note in notes:
            message.append(note)

    return ' '.join(message)




def getContentsString(case):
    contentTypes = {
        1: "Medical supplies",
        2: "Chemicals",
        3: "Food",
        4: "Electronics",
        5: "Industrial components",
        6: "Unknown"
    }
    return contentTypes.get(case, "Miscellaneous")


class PopUp:
    def __init__(self, master, message):
        self.message = message
        self.master = master
        master.title("Container Info")

        self.label = Label(master, text=message)
        self.label.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()
