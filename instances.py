from MoveControl import MoveControl
from NonVolatile import Settings, Counter
from tkinter import *

counter = Counter("counter")
movecontrol = MoveControl()

# select = "CHK_N0144U"
# select = "Elektrolyt"

but_w = 15
but_h = 5
root = Tk()


def eleltrolyt():
    global select
    select = "Elektrolyt"
    root.destroy()


def chk_n0144u():
    global select
    select = "CHK_N0144U"
    root.destroy()


def lis_raz():
    global select
    select = "LisRaz"
    root.destroy()


elektrolyt = Button(root, text="Elektrolyt", width=but_w, height=but_h, font=("Courier", 45), command=eleltrolyt)
elektrolyt.pack(side="left")
CHK_N0144U = Button(root, text="CHK <=> N0144U", width=but_w, height=but_h, font=("Courier", 45), command=chk_n0144u)
CHK_N0144U.pack(side="left")
LisRaz = Button(root, text="Lisované\n<=>\n Rážované", width=but_w, height=but_h, font=("Courier", 45), command=lis_raz)
LisRaz.pack(side="left")

root.mainloop()

if select == "Elektrolyt":
    from Elektrolyt.ImageRecognizer import Decider, GuiProvider

    settings = Settings("Elektrolyt/settings")
    recognizer = Decider(settings.settings)
    gui_computer = GuiProvider(settings.settings)

    from Elektrolyt.GUI.StartGUI import StartGUI
    gui = StartGUI

elif select == "CHK_N0144U":
    from CHK_N0144U.ImageRecognizer import Decider, GuiProvider

    settings = Settings("CHK_N0144U/settings")
    recognizer = Decider(settings.settings)
    gui_computer = GuiProvider(settings.settings)

    from CHK_N0144U.GUI.StartGUI import StartGUI
    gui = StartGUI

elif select == "LisRaz":
    from LisRaz.ImageRecognizer import Decider, GuiProvider

    settings = Settings("LisRaz/settings")
    recognizer = Decider(settings.settings)
    gui_computer = GuiProvider(settings.settings)

    from LisRaz.GUI.StartGUI import StartGUI
    gui = StartGUI
