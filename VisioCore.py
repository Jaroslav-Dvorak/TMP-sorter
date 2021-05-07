import cv2
import numpy as np
from copy import deepcopy as cp


class CenterByTemplate:
    def __init__(self):
        sizes = [n for n in range(140, 220, 5)]
        self.templates = [np.zeros((size, size), np.uint8) for size in sizes]
        for template in self.templates:
            cv2.circle(img=template,
                       center=((template.shape[0]) // 2, (template.shape[1]) // 2),
                       radius=template.shape[0] // 2,
                       color=255,
                       thickness=-1)

        self.opened_img = np.zeros((100, 100), np.uint8)
        self.marked_circle = np.zeros((100, 100), np.uint8)

    def compute(self, img, open_kernel):
        orig_img = cp(img)
        img = cv2.bitwise_not(img)
        opening = (open_kernel, open_kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones(opening, np.uint8))
        img = cv2.bitwise_not(img)

        max_val_best = -100

        # methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
        #            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
        min_loc, max_loc, w, h = 0, 0, 1, 1
        method = cv2.TM_CCOEFF_NORMED
        for template in self.templates:
            res = cv2.matchTemplate(img, template, method)
            temp_vals = cv2.minMaxLoc(res)
            if temp_vals[1] >= max_val_best:
                w, h = template.shape[::-1]
                max_val_best = temp_vals[1]
                min_val, max_val, min_loc, max_loc = temp_vals

        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        center = (
            top_left[0] + (bottom_right[0] - top_left[0]) // 2, top_left[1] + (bottom_right[1] - top_left[1]) // 2)
        radius = w // 2

        self.opened_img = img
        orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)
        cv2.circle(orig_img, center, radius, (0, 255, 0), 3)
        self.marked_circle = orig_img
        return center, radius


class Blur:
    def __init__(self):
        self.blurred = np.zeros((100, 100), np.uint8)

    def compute(self, img, blur_val):
        img = cp(img)
        blur_val = blur_val//4
        if blur_val % 2 == 0:
            blur_val += 1
        img = cv2.medianBlur(img, blur_val)
        self.blurred = img
        return img


class DiffOrig:
    def __init__(self):
        self.differed = np.zeros((100, 100), np.uint8)

    def compute(self, orig_img, diff_img, bg_val):
        orig_img = cp(orig_img)
        diff_img = cp(diff_img)
        bg_val = bg_val//4
        if bg_val > 255:
            bg_val = 255
        img = bg_val - cv2.absdiff(orig_img, diff_img)
        self.differed = img
        return img


class CropCircle:
    def __init__(self):
        self.cropped = np.zeros((100, 100), np.uint8)

    def compute(self, img, coordinations):
        img = cp(img)
        center, radius = coordinations
        crop = img[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
        crop = cv2.resize(crop, (100, 100))
        self.cropped = crop
        return crop


class Unfold:
    def __init__(self):
        self.unfolded = np.zeros((100, 100), np.uint8)

    def compute(self, img):
        img = cp(img)
        h, w = img.shape
        unfolded = cv2.linearPolar(img, (h // 2, w // 2), h // 2, cv2.INTER_CUBIC)
        unfolded = cv2.flip(unfolded, 1)

        self.unfolded = unfolded

        return unfolded


class BinAdaptive1:
    def __init__(self):
        self.binarized = np.zeros((100, 100), np.uint8)

    def compute(self, img, thresh1):
        img = cp(img)
        thresh1 = thresh1 // 4
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, thresh1)
        self.binarized = img
        return img


class Gabor:
    def __init__(self):
        self.gabored = np.zeros((100, 100), np.uint8)

    def compute(self, img, kwargs):
        img = cp(img)
        # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        uhel = kwargs["uhel"]
        sigma = kwargs["sigma"]
        lambd = kwargs["lambd"]
        gamma = kwargs["gamma"]
        psi = kwargs["psi"]
        kernel = kwargs["kernel"]

        uhel = uhel/1020*3.14
        sigma = sigma/1020*20
        lambd = lambd/1020*10
        gamma = gamma/1020*3
        psi = psi/1020*3
        kernel = int(kernel/1020*30)
        kernel = (8, kernel)

        g_kernel = cv2.getGaborKernel(kernel, sigma, uhel, lambd, gamma, psi, ktype=cv2.CV_32F)
        gabor = cv2.filter2D(img, -1, g_kernel)

        # f = np.fft.fft2(img)
        # fshift = np.fft.fftshift(f)
        # magnitude_spectrum = 20 * np.log(np.abs(fshift))
        # magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
        img = gabor

        img = cv2.bitwise_not(img)

        _, img = cv2.threshold(img, 1, 255, 0)

        # img = cv2.ximgproc.thinning(img)
        img = cv2.bitwise_not(img)
        self.gabored = img
        return img


class ConComponents:
    def __init__(self):
        self.connected_comp = np.zeros((100, 100), np.uint8)

    def compute(self, img, bgr=False):
        img = cp(img)
        img = cv2.bitwise_not(img)
        if bgr:
            green = [0, 255, 0]
            orange = [0, 127, 255]
            red = [0, 0, 255]
            blue = [255, 0, 0]
        else:
            green = [0, 255, 0]
            orange = [255, 127, 0]
            red = [255, 0, 0]
            blue = [0, 0, 255]

        img[:, 60:] = 0

        nb_components, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
        heights = stats[1:, cv2.CC_STAT_HEIGHT]
        # widths = stats[1:, cv2.CC_STAT_WIDTH]
        # tops = stats[1:, cv2.CC_STAT_TOP]
        # lefts = stats[1:, cv2.CC_STAT_LEFT]
        # centroids = centroids[1:]
        # nb_components = nb_components - 1

        big = 30
        middle = 20

        big_groups = np.where(heights >= big)
        big_groups = np.add(big_groups, 1)[0]
        middle_groups = np.where(np.logical_and(heights >= middle, heights < big))
        middle_groups = np.add(middle_groups, 1)[0]
        small_groups = np.where(heights < middle)
        small_groups = np.add(small_groups, 1)[0]

        loc_big = np.where(np.isin(labels, big_groups))
        loc_middle = np.where(np.isin(labels, middle_groups))
        loc_small = np.where(np.isin(labels, small_groups))

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        img[loc_big] = green
        img[loc_middle] = orange
        img[loc_small] = red

        img[:, 30:31] = blue
        img[50:51, :60] = blue

        sectors = {
            "tl": {"img": img[:50, :30]},
            "tr": {"img": img[:50, 30:60]},
            "bl": {"img": img[50:100, :30]},
            "br": {"img": img[50:100, 30:60]}
                  }

        # imgstats = np.zeros((250, 250, 3), dtype=np.uint8)
        score = {}
        for sector, cont in sectors.items():
            greens = (cont["img"] == green).all(axis=2).sum()
            oranges = (cont["img"] == orange).all(axis=2).sum()
            reds = (cont["img"] == red).all(axis=2).sum()

            greens *= 1
            oranges *= 1.3
            reds *= 1.5

            score[sector] = (greens + oranges + reds)

            # sectors[sector]["score"] = score
            # Commons["score"][sector] = score
            # print(sector, score)

        self.connected_comp = img
        return score


class Evaluate:
    def __init__(self):
        self.blue = [0, 0, 255]
        self.green = [0, 255, 0]
        self.red = [255, 0, 0]

        self.blank = np.zeros((100, 100, 3), np.uint8)
        self.blank[:, 32:33] = self.blue
        self.blank[50:51, :66] = self.blue
        self.blank[:, 65:66] = self.blue

        self.eval_res = np.zeros((100, 100), np.uint8)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5
        self.lineType = 1
        self.scores_pos = {"tl": (1, 28), "tr": (34, 28), "bl": (1, 78), "br": (34, 78), "dscore": (67, 28)}

    def compute(self, score, dscore):

        OK = True
        self.eval_res = cp(self.blank)
        for sector, score in score.items():
            if score < dscore:
                cv2.putText(self.eval_res, str(int(score)), self.scores_pos[sector], self.font, self.fontScale, self.red, self.lineType)
                OK = False
            else:
                cv2.putText(self.eval_res, str(int(score)), self.scores_pos[sector], self.font, self.fontScale,  self.green, self.lineType)

        if OK:
            cv2.putText(self.eval_res, str(dscore), self.scores_pos["dscore"], self.font, self.fontScale, self.green, self.lineType)
        else:
            cv2.putText(self.eval_res, str(dscore), self.scores_pos["dscore"], self.font, self.fontScale, self.red, self.lineType)

        return OK
