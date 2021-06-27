from tkinter import Tk, Button, Label
from functools import partial
from Elektrolyt.GUI.AdvGUI import AdvGUI
from Elektrolyt.GUI.SimpleGUI import SimpleGUI


class StartGUI:
    def __init__(self):
        # self.geometry = '200x300'
        self.title = "StartVisio"
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback_exit)
        self.root.title(self.title)
        # self.root.geometry(self.geometry)

        self.counter = 3
        self.countdown = Label(self.root)
        self.countdown.pack()

        self.but_Simple = Button(self.root, text="Jednoduché", font=("Courier", 36), command=partial(self.callback_button, "simple"))
        self.but_Simple.pack()
        self.but_Advanc = Button(self.root, text="Pokročilé", font=("Courier", 36), command=partial(self.callback_button, "advanced"))
        self.but_Advanc.pack()

        self.update_countdown()
        self.root.mainloop()

    def update_countdown(self):
        self.countdown.configure(text=str(self.counter))
        if self.counter > 0:
            self.root.after(1000, self.update_countdown)
        else:
            self.root.destroy()
            SimpleGUI()
        self.counter -= 1

    def callback_exit(self):
        self.root.destroy()

    def callback_button(self, name):
        self.root.destroy()
        if name == "simple":
            SimpleGUI()
        if name == "advanced":
            AdvGUI()
