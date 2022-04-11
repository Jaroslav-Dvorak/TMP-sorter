from time import perf_counter
from Elektrolyt.VisioCore import *


class ImageRecognizer:
    def __init__(self, settings, resolution=(240, 320)):
        self.settings = settings

        self.open_for_center = Opening()
        self.centertempl = CenterByTemplate()
        self.mark = MarkCircle()
        self.open_for_proc = Opening()
        self.negative = Negative()
        self.crop = CropCircle()
        self.mask = Mask()
        self.binarize = Binarize()
        self.result = Result()
        self.eval = Evaluation()

        blank = np.zeros(resolution, np.uint8)
        self.input_img = blank

        self.process_time = 0

    def load_imge(self, file):
        self.input_img = cv2.imread(file, 0)

    def evaluation(self):
        start = perf_counter()
        input_img = self.input_img

        open_for_center = self.open_for_center.compute(input_img, self.settings["oppened_center"]["trackbars"]["kernel"])
        center, radius = self.centertempl.compute(open_for_center)
        marked = self.mark.compute(input_img, (center, radius))
        opened = self.open_for_proc.compute(input_img, self.settings["oppened_proc"]["trackbars"]["kernel"])
        cropped = self.crop.compute(opened, (center, radius), self.settings["cropped"]["trackbars"]["offset"])
        negatived = self.negative.compute(cropped)
        masked = self.mask.compute(negatived)
        binarized = self.binarize.compute(masked, self.settings["binariz"]["trackbars"]["thresh"])
        result = self.result.compute(masked, binarized)
        evaluation = self.eval.compute(binarized, center, self.settings["oppened_proc"]["trackbars"]["kernel"], self.settings["evaluation"]["trackbars"]["thresh"])

        self.process_time = perf_counter() - start
        return evaluation


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
            "oppened_center": self.open_for_center.opened_img,
            "marked": self.mark.marked_circle,
            "cropped": self.crop.cropped,
            "oppened_proc": self.open_for_proc.opened_img,
            "negatived": self.negative.negatived,
            "masked": self.mask.masked,
            "binariz": self.binarize.binarized,
            "result": self.result.result,
            "evaluation": self.eval.eval_res
        }

