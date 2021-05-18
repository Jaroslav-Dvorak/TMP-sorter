from tkinter import Tk, Frame, Button, Label, NW, W, VERTICAL, X, Scale
# from tkinter.ttk import Scale
from time import process_time
from copy import deepcopy as cp
from functools import partial
from PIL import Image, ImageTk
from ImageRecognizer import recognizer, gui_computer
from NonVolatile import defaultSet
from MoveControl import movecontrol
import cv2
import numpy as np


class SimpleGUI:
    def __init__(self):

        self.settings = defaultSet.settings
        self.state_color = {True: "#4fc261", False: "#e04646"}
        self.camera_conn_textlist = {True: "KAMERA PŘIPOJENA", False: "KAMERA NEPŘIPOJENA"}

        self.root = Tk()
        self.root.protocol("WM_DELETE_WINDOW", self.callback_exit)
        self.root.attributes("-fullscreen", True)
        self.root.configure(background='black')

        self.size_h = 750
        self.view1 = Label()
        self.view1.grid(row=0, column=0)
        self.view2 = Label()
        self.view2.grid(row=0, column=1)
        self.sensitivity = Scale(width=150, orient=VERTICAL, from_=0, to=1020, length=750, showvalue=False, command=partial(self.callback_scale, "default", "default"))
        self.sensitivity.set(value=self.settings["evaluated"]["trackbars"]["dscore"])
        self.sensitivity.grid(row=0, column=2)
        self.view3 = Label()
        self.view3.grid(row=1, column=1)

        self.tools = Frame(self.root, highlightbackground="black", highlightthickness=1)
        self.tools.grid(row=1, column=0)
        self.frame_counter = Frame(self.tools, highlightbackground="black", highlightthickness=1)
        self.frame_counter.pack()
        self.label_counter = Label(self.frame_counter, text="Počítadlo\nOK kusy", font=("Courier", 60), fg="green")
        self.label_counter.pack(side="left")
        self.counter = Label(self.frame_counter, text="", font=("Courier", 100), fg="blue")
        self.counter.pack(side="left")
        self.button_counter_reset = Button(self.frame_counter, text="Reset\npočítadla", font=("Courier", 60), command=partial(self.callback_button, "counter_reset"))
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

        self.button_save = Button(self.tools, text="Potvrdit a uložit", highlightbackground='green', font=("Courier", 40), command=partial(self.callback_button, "submit"))
        self.button_save.pack()

        # self.button_exit = Button(self.root, text="Exit", font=("Courier", 30), command=partial(self.callback_button, "exit"))
        # self.button_exit.pack()

        self.run()

    def callback_scale(self, view, which, val):
        self.settings["evaluated"]["trackbars"]["dscore"] = int(float(val))
        self.button_save.configure(highlightbackground='red')

    def callback_exit(self):
        self.root.destroy()

    def callback_button(self, name):
        if name == "submit":
            recognizer.settings = cp(self.settings)
            defaultSet.settings = cp(self.settings)
            self.button_save.configure(highlightbackground='green')
        if name == "counter_reset":
            defaultSet.counter = 0
        if name == "exit":
            self.root.destroy()

    def run(self):

        while True:
            pictures = gui_computer.show_loop(self.settings)
            start = process_time()

            indicator = self.draw_indicator(pictures["marked"])
            indicator = Image.fromarray(indicator)
            indicator = ImageTk.PhotoImage(image=indicator)
            self.view1.configure(image=indicator)

            result = self.merge_cropped_con_comp(pictures["cropped"], pictures["con_comp"])
            result = Image.fromarray(result)
            result = ImageTk.PhotoImage(image=result)
            self.view2.configure(image=result)

            stats = pictures["evaluated"]
            stats = cv2.resize(stats, (300, 300))
            stats = Image.fromarray(stats)
            stats = ImageTk.PhotoImage(image=stats)
            self.view3.configure(image=stats)

            self.counter.configure(text=f"{defaultSet.counter:4}")
            self.cam_conn_label.configure(background=self.state_color[movecontrol.cam_connected], text=self.camera_conn_textlist[movecontrol.cam_connected])
            self.evaluation_time.configure(text=f"IMG time: {int(recognizer.process_time*1000):5} ms")
            self.plc_time.configure(text=f"PLC time: {int(movecontrol.process_time * 1000):5} ms")
            self.gui_time.configure(text=f"GUI time: {int(self.process_time * 1000):5} ms")

            self.root.update()
            self.process_time = process_time() - start

    def draw_indicator(self, marked):
        sizeh = self.size_h
        h, w = marked.shape[0:2]
        center = w//2, h//2
        ratio = w / h

        big_circle = max(gui_computer.centertempl.sizes)
        cv2.circle(marked, center, big_circle // 2, (255, 255, 0), 1)
        marked = cv2.resize(marked, (int(sizeh * ratio), sizeh))
        return marked

    def merge_cropped_con_comp(self, cropped, con_comp):
        sizeh = self.size_h
        h, w = con_comp.shape[0:2]
        x, y, r = (w // 2, h // 2, h // 2)
        con_comp = cv2.flip(con_comp, 1)
        con_comp = cv2.linearPolar(con_comp, (x, y), r, cv2.INTER_AREA + cv2.WARP_FILL_OUTLIERS + cv2.WARP_INVERSE_MAP)
        ratio = w / h
        con_comp = cv2.resize(con_comp, (sizeh, int(sizeh * ratio)))

        cropped = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)
        h, w = cropped.shape[0:2]
        ratio = h / w
        cropped = cv2.resize(cropped, (sizeh, int(sizeh * ratio)))
        ret, mask = cv2.threshold(cv2.cvtColor(con_comp, cv2.COLOR_BGR2GRAY), 25, 255, cv2.THRESH_BINARY)

        mask = cv2.bitwise_not(mask)
        mask_inv = cv2.bitwise_not(mask)
        con_comp = cv2.bitwise_and(con_comp, con_comp, mask=mask_inv)
        cropped = cv2.bitwise_and(cropped, cropped, mask=mask)
        return cv2.add(con_comp, cropped)





# SimpleGUI()
