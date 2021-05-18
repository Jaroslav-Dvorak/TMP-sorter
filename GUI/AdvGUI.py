from tkinter import Tk, Frame, Button, Label, NW, W, VERTICAL, X
from tkinter.ttk import Scale
from time import process_time
from copy import deepcopy as cp
from functools import partial
from PIL import Image, ImageTk
from ImageRecognizer import recognizer, gui_computer
from NonVolatile import defaultSet
from MoveControl import movecontrol


class AdvGUI:
    def __init__(self):

        self.settings = defaultSet.settings

        self.geometry = '1920x1080'
        self.title = "Visio"
        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback_exit)
        self.root.title(self.title)
        self.root.geometry(self.geometry)
        self.state_color = {True: "#4fc261", False: "#e04646"}
        self.camera_conn_textlist = {True: "KAMERA PŘIPOJENA", False: "KAMERA NEPŘIPOJENA"}

        self.views = {}
        images = {}
        self.scales = {}
        pictures = gui_computer.show_loop(self.settings)
        row = 0
        width = 0
        frames = [Frame(self.root) for _ in range(3)]
        for frame in frames:
            frame.pack(side="top", anchor=NW)
        for view, p in self.settings.items():
            images[view] = Image.fromarray(pictures[view])
            images[view] = ImageTk.PhotoImage(image=images[view])
            self.views[view] = (Label(frames[row], text=view, image=images[view], compound='bottom'))
            self.views[view].pack(side="left")
            width += 320
            for item, val in p["trackbars"].items():
                self.scales[item] = Scale(frames[row],
                                          value=val,
                                          orient=VERTICAL, from_=0, to=1020, length=300,
                                          command=partial(self.callback_scale, view, item))
                self.scales[item].pack(side="left")
                width += 20

            if width >= 1500:
                row += 1
                width = 0

        self.tools = Frame(frames[2], highlightbackground="black", highlightthickness=1)
        self.tools.pack(side="right")

        self.frame_counter = Frame(self.tools, highlightbackground="black", highlightthickness=1)
        self.frame_counter.pack()

        self.label_counter = Label(self.frame_counter, text="Počítadlo\nOK kusy", font=("Courier", 36), fg="green")
        self.label_counter.pack(side="left")
        self.counter = Label(self.frame_counter, text="", font=("Courier", 60), fg="blue")
        self.counter.pack(side="left")
        self.button_counter_reset = Button(self.frame_counter, text="Reset\npočítadla", font=("Courier", 36), command=partial(self.callback_button, "counter_reset"))
        self.button_counter_reset.pack(side="right", anchor=W)

        self.cam_conn_label = Label(self.tools, text="KAMERA NEPŘIPOJENA", font=("Courier", 20), bg="red")
        self.cam_conn_label.pack()
        self.evaluation_time = Label(self.tools, text="IMG time:: __ ms", font=("Courier", 20))
        self.evaluation_time.pack()
        self.plc_time = Label(self.tools, text="PLC time:__ ms", font=("Courier", 20))
        self.plc_time.pack()
        self.process_time = 0
        self.gui_time = Label(self.tools, text="GUI time: __ ms", font=("Courier", 20))
        self.gui_time.pack()

        self.button_save = Button(self.tools, text="Potvrdit a uložit", highlightbackground='green', font=("Courier", 30), command=partial(self.callback_button, "submit"))
        self.button_save.pack()

        self.run()

    def callback_scale(self, view, which, val):
        self.settings[view]["trackbars"][which] = int(float(val))
        self.button_save.configure(highlightbackground='red')

    def callback_button(self, name):
        if name == "submit":
            recognizer.settings = cp(self.settings)
            defaultSet.settings = cp(self.settings)
            self.button_save.configure(highlightbackground='green')
        if name == "counter_reset":
            defaultSet.counter = 0

    def run(self):
        while True:
            pictures = gui_computer.show_loop(self.settings)
            start = process_time()
            images = {}
            for view in self.settings:
                images[view] = Image.fromarray(pictures[view])
                ratio = images[view].size[1] / images[view].size[0]
                images[view] = images[view].resize((320, int(320*ratio)))
                images[view] = ImageTk.PhotoImage(image=images[view])
                self.views[view].configure(image=images[view])

            self.counter.configure(text=f"{defaultSet.counter:4}")
            self.cam_conn_label.configure(background=self.state_color[movecontrol.cam_connected], text=self.camera_conn_textlist[movecontrol.cam_connected])
            self.evaluation_time.configure(text=f"IMG time: {int(recognizer.process_time*1000):5} ms")
            self.plc_time.configure(text=f"PLC time: {int(movecontrol.process_time * 1000):5} ms")
            self.gui_time.configure(text=f"GUI time: {int(self.process_time * 1000):5} ms")

            self.root.update()
            self.process_time = process_time() - start

    def callback_exit(self):
        self.root.destroy()