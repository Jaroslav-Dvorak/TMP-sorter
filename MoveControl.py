from time import perf_counter, time
from copy import copy
from gpiozero import LED, Button
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero.pins.local import LocalPiFactory
from NonVolatile import defaultSet
from is_rpi import is_rpi


class MoveControl:

    def __init__(self):

        pin_factory = LocalPiFactory() if is_rpi() else PiGPIOFactory(host='192.168.10.51')

        self.cam_connected = False

        self.switch = Button(21, pin_factory=pin_factory, pull_up=False)
        self.stul_0 = Button(26, pin_factory=pin_factory, pull_up=False)
        self.stul_1 = Button(20, pin_factory=pin_factory, pull_up=False)
        self.Pist1_0 = Button(16, pin_factory=pin_factory, pull_up=False)
        self.Pist1_1 = Button(19, pin_factory=pin_factory, pull_up=False)
        self.Pist2_0 = Button(13, pin_factory=pin_factory, pull_up=False)
        self.Pist2_1 = Button(12, pin_factory=pin_factory, pull_up=False)

        self.CamTrig = LED(6, pin_factory=pin_factory)
        self.StulStep = LED(5, pin_factory=pin_factory)
        self.StulArr = LED(22, pin_factory=pin_factory)
        self.Pist1 = LED(27, pin_factory=pin_factory)
        self.Pist2 = LED(17, pin_factory=pin_factory)

        self.badone = None

        self.process_time = 0

    def run(self):
        allow_run = False
        f_switch = False
        f_receivered = False
        cam_pulstimer = 0
        stul_command = False
        blocking_cyl = False
        counting_ok_piece = False
        waiting_photo = False

        while True:
            start = perf_counter()
            i_switch = copy(self.switch.is_pressed)
            i_stul_rot = copy(self.stul_0.is_pressed)
            i_stul_arr = copy(self.stul_1.is_pressed)
            i_cyl_1_0 = copy(self.Pist1_0.is_pressed)
            i_cyl_1_1 = copy(self.Pist1_1.is_pressed)
            i_cyl_2_0 = copy(self.Pist2_0.is_pressed)
            i_cyl_2_1 = copy(self.Pist2_1.is_pressed)
            o_capture = False
            o_aret = False
            o_step = False
            o_cyl_1 = False
            o_cyl_2 = False

            if not f_switch and i_switch and not waiting_photo:
                f_switch = True
                if self.cam_connected:
                    allow_run = True
            if not i_switch:
                f_switch = False
            if not self.cam_connected or not i_switch:
                allow_run = False

            # start kamera + válce
            if not f_receivered and i_stul_arr and allow_run:
                f_receivered = True
                cam_pulstimer = time()
                if not blocking_cyl:
                    o_cyl_1 = True
                o_cyl_2 = True

            elif not i_stul_arr or not allow_run:
                f_receivered = False
            if 0.05 < (time() - cam_pulstimer) < 0.2:
                o_capture = True
            if not i_switch and self.badone is None and not waiting_photo:
                cam_pulstimer = time()
                waiting_photo = True

            if self.Pist1.value:
                o_cyl_1 = True
            if self.Pist2.value:
                o_cyl_2 = True
            if i_cyl_1_1:
                o_cyl_1 = False
                if counting_ok_piece:
                    counting_ok_piece = False
                    defaultSet.counter += 1
            if i_cyl_2_1:
                o_cyl_2 = False

            # stůl
            if self.badone is not None:
                waiting_photo = False
                stul_command = True
                blocking_cyl = self.badone
                self.badone = None
            if stul_command and not i_stul_rot and i_stul_arr and i_cyl_1_0 and i_cyl_2_0 and allow_run and not self.Pist1.value and not self.Pist1.value:
                o_step = True
                counting_ok_piece = True
            if i_stul_rot and not i_stul_arr:
                o_aret = True
                stul_command = False
            if not allow_run:
                stul_command = False

            self.setter(self.CamTrig, o_capture)
            self.setter(self.StulArr, o_aret)
            self.setter(self.StulStep, o_step)
            self.setter(self.Pist1, o_cyl_1)
            self.setter(self.Pist2, o_cyl_2)

            self.process_time = perf_counter() - start

    @staticmethod
    def setter(output, state):
        if state:
            output.on()
        else:
            output.off()


movecontrol = MoveControl()
