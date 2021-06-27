from time import perf_counter
from Elektrolyt.VisioCore import *


class ImageRecognizer:
    def __init__(self, settings, resolution=(240, 320)):
        self.settings = settings

        self.centertempl = CenterByTemplate()
        self.blur = Blur()
        self.diff = DiffOrig()
        self.crop = CropCircle()
        self.unfold = Unfold()
        self.binarize = BinAdaptive1()
        self.gabor = Gabor()
        self.con_comp = ConComponents()
        self.eval = Evaluate()

        blank = np.zeros(resolution, np.uint8)
        self.input_img = blank

        self.process_time = 0

    def load_imge(self, file):
        self.input_img = cv2.imread(file, 0)

    def evaluation(self):
        start = perf_counter()
        input_img = self.input_img
        center, radius = self.centertempl.compute(input_img, self.settings["oppened"]["trackbars"]["kernel"])
        blurred = self.blur.compute(input_img, self.settings["blurred"]["trackbars"]["blur_val"])
        differed = self.diff.compute(input_img, blurred, self.settings["differed"]["trackbars"]["bg_val"])
        cropped = self.crop.compute(differed, (center, radius))
        unfolded = self.unfold.compute(cropped)
        binarized = self.binarize.compute(unfolded, self.settings["binarized"]["trackbars"]["thresh1"])
        gabored = self.gabor.compute(binarized, self.settings["gabored"]["trackbars"])
        connected = self.con_comp.compute(gabored)
        evaluated = self.eval.compute(connected, self.settings["evaluated"]["trackbars"]["dscore"])
        self.process_time = perf_counter() - start
        return evaluated


class Decider(ImageRecognizer):

    def __init__(self, settings):
        super().__init__(settings)

    def evaluator(self, file):
        self.load_imge(file)
        return self.evaluation()


class GuiProvider(ImageRecognizer):

    def __init__(self, settings):
        super().__init__(settings)

    def show_loop(self, settings):
        self.settings = settings
        self.evaluation()
        return {
            "orig": self.input_img,
            "marked": self.centertempl.marked_circle,
            "oppened": self.centertempl.opened_img,
            "blurred": self.blur.blurred,
            "differed": self.diff.differed,
            "cropped": self.crop.cropped,
            "unfolded": self.unfold.unfolded,
            "binarized": self.binarize.binarized,
            "gabored": self.gabor.gabored,
            "con_comp": self.con_comp.connected_comp,
            "evaluated": self.eval.eval_res
        }

