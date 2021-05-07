import json
from copy import deepcopy as cp


class NonVolatile:
    def __init__(self, settings_filename, counter_filename):
        self.settings_filename = settings_filename + ".json"
        self.counter_filename = counter_filename + ".txt"
        self.init_settings()
        self.init_counter()

    def init_settings(self):
        try:
            with open(self.settings_filename, "r") as f:
                self.settings = json.loads(f.read())
        except FileNotFoundError:
            self.settings = {
                        "orig": {"trackbars": {}},
                        "marked": {"trackbars": {"offset": 0}},
                        "oppened": {"trackbars": {"kernel": 22}},
                        "blurred": {"trackbars": {"blur_val": 292}},
                        "differed": {"trackbars": {"bg_val": 1020}},
                        "cropped": {"trackbars": {}},
                        "unfolded": {"trackbars": {}},
                        "binarized": {"trackbars": {"thresh1": 60}},
                        "gabored": {"trackbars": {"kernel": 731, "uhel": 0, "sigma": 1020, "lambd": 114, "gamma": 0, "psi": 0}},
                        "con_comp": {"trackbars": {}},
                        "evaluated": {"trackbars": {"dscore": 220}}
                    }

    @property
    def settings(self):
        return cp(self._settings)

    @settings.setter
    def settings(self, settings):
        s = json.dumps(settings, indent=4)
        try:
            with open(self.settings_filename, "w") as f:
                f.write(s)
        except Exception as e:
            print(e)
        self._settings = settings
    #####################################################

    def init_counter(self):
        try:
            with open(self.counter_filename, "r") as f:
                self.counter = json.loads(f.read())
        except FileNotFoundError:
            self.counter = 0

    @property
    def counter(self):
        return self._counter

    @counter.setter
    def counter(self, counter):
        s = json.dumps(counter, indent=4)
        try:
            with open(self.counter_filename, "w") as f:
                f.write(s)
        except Exception as e:
            print(e)
        self._counter = counter


defaultSet = NonVolatile("default", "counter")
