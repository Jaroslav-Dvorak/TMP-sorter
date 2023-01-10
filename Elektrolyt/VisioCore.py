import cv2
import numpy as np
from copy import deepcopy as cp


class Negative:
    def __init__(self):
        self.negatived = np.zeros((100, 100), np.uint8)

    def compute(self, img):
        img = cp(img)
        img = cv2.bitwise_not(img)
        self.negatived = img
        return img


class Opening:
    def __init__(self):
        self.opened_img = np.zeros((100, 100), np.uint8)

    def compute(self, img, open_kernel):
        open_kernel = open_kernel // 40
        img = cp(img)
        img = cv2.bitwise_not(img)
        opening = (open_kernel, open_kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, np.ones(opening, np.uint8))
        img = cv2.bitwise_not(img)
        self.opened_img = img
        return img


class CenterByTemplate:
    def __init__(self):
        self.sizes = [n for n in range(100, 150, 5)]
        self.templates = [np.zeros((size, size), np.uint8) for size in self.sizes]
        for template in self.templates:
            cv2.circle(img=template,
                       center=((template.shape[0]) // 2, (template.shape[1]) // 2),
                       radius=template.shape[0] // 2,
                       color=255,
                       thickness=-1)

    def compute(self, img):
        img = cp(img)

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

        return center, radius


class MarkCircle:
    def __init__(self):
        self.marked_circle = np.zeros((100, 100), np.uint8)

    def compute(self, img, coordinations):
        img = cp(img)
        center, radius = coordinations
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(img, center, radius, (0, 255, 0), 3)
        self.marked_circle = img
        return img

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

    def compute(self, img, coordinations, offset):
        img = cp(img)
        center, radius = coordinations
        offset = offset - 1020//2
        offset = offset // 50
        try:
            offset_radius = radius + offset
            crop = img[center[1]-offset_radius:center[1]+offset_radius, center[0]-offset_radius:center[0]+offset_radius]
        except cv2.error:
            crop = img[center[1]-radius:center[1]+radius, center[0]-radius:center[0]+radius]
        try:
            crop = cv2.resize(crop, (100, 100))
        except cv2.error:
            return self.cropped
        self.cropped = crop
        return crop


class Mask:
    def __init__(self):
        self.masked = np.zeros((100, 100), np.uint8)

    def compute(self, img):
        img = cp(img)
        mask = np.zeros_like(img)
        mask = cv2.circle(mask, (50, 50), 50, (255), -1)
        w = np.where(mask == 0)
        img[w] = mask[w]
        self.masked = img
        return img


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


class Binarize:
    def __init__(self):
        self.binarized = np.zeros((100, 100), np.uint8)

    def compute(self, img, thresh1):
        img = cp(img)
        thresh1 = thresh1 // 4
        ret, img = cv2.threshold(img, thresh1, 255, cv2.THRESH_BINARY)
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
        widths = stats[1:, cv2.CC_STAT_WIDTH]
        tops = stats[1:, cv2.CC_STAT_TOP]
        bottoms = tops + heights
        lefts = stats[1:, cv2.CC_STAT_LEFT]
        rights = lefts + widths
        x_axises = lefts + widths//2
        centroids = centroids[1:]
        areas = stats[1: cv2.CC_STAT_AREA]

        big = 30
        middle = 20

        big_groups_index = np.where(heights >= big)
        big_groups = np.add(big_groups_index, 1)[0]
        middle_groups_index = np.where(np.logical_and(heights >= middle, heights < big))
        middle_groups = np.add(middle_groups_index, 1)[0]
        small_groups_index = np.where(heights < middle)
        small_groups = np.add(small_groups_index, 1)[0]

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        sectors_l = np.zeros(img.shape[0])
        sectors_r = np.zeros(img.shape[0])
        if len(rights[big_groups_index]) > 0:

            right = max(rights[big_groups_index])
            half = right//2
            # img[:, half:half+1] = blue
            # img[:, right:right+1] = blue

            left_cons = np.where(x_axises < half)
            right_cons = np.where((half <= x_axises) & (x_axises <= right))
            groups = dict(
                left_bigs=np.intersect1d(left_cons, big_groups_index),
                left_middles=np.intersect1d(left_cons, middle_groups_index),
                left_smalls=np.intersect1d(left_cons, small_groups_index),
                right_bigs=np.intersect1d(right_cons, big_groups_index),
                right_middles=np.intersect1d(right_cons, middle_groups_index),
                right_smalls=np.intersect1d(right_cons, small_groups_index)
            )

            big_score = 1
            middle_score = 0.7
            small_score = 0.1

            for group, indexes in groups.items():
                for index in indexes:
                    if group == "left_bigs":
                        sectors_l[tops[index]:bottoms[index]] += big_score
                    if group == "left_middles":
                        sectors_l[tops[index]:bottoms[index]] += middle_score
                    if group == "left_smalls":
                        sectors_l[tops[index]:bottoms[index]] += small_score

                    if group == "right_bigs":
                        sectors_r[tops[index]:bottoms[index]] += big_score
                    if group == "right_middles":
                        sectors_r[tops[index]:bottoms[index]] += middle_score
                    if group == "right_smalls":
                        sectors_r[tops[index]:bottoms[index]] += small_score

        score = (sectors_l, sectors_r)
        loc_big = np.where(np.isin(labels, big_groups))
        loc_middle = np.where(np.isin(labels, middle_groups))
        loc_small = np.where(np.isin(labels, small_groups))

        img[loc_big] = green
        img[loc_middle] = orange
        img[loc_small] = red
        self.connected_comp = img
        return score


class Result:
    def __init__(self):
        self.result = np.zeros((100, 100, 3), np.uint8)

    def compute(self, orig_img, defects_img):
        img = cp(orig_img)
        defects_img = cp(defects_img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        w = np.where(defects_img == 255)
        img[w] = (255, 0, 0)
        self.result = img
        return img


class Evaluation:
    def __init__(self):
        self.blue = [51, 153, 255]
        self.green = [0, 255, 0]
        self.red = [255, 0, 0]

        self.blank = np.zeros((100, 100, 3), np.uint8)
        self.eval_res = np.zeros((100, 100), np.uint8)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5
        self.lineType = 1
        self.scores_pos = {"zalis": (35, 60), "uzaver": (35, 90), "dscore": (35, 30)}

        cv2.putText(self.blank, "citlivost:", (1, 15), self.font, self.fontScale, self.blue, self.lineType)
        cv2.putText(self.blank, "hranice:", (1, 45), self.font, self.fontScale, self.blue, self.lineType)
        cv2.putText(self.blank, "stred:", (1, 75), self.font, self.fontScale, self.blue, self.lineType)

    def compute(self, defect_img, center, sensitivity, thresh):
        sensitivity //= 40
        thresh = int(thresh * 1)
        self.eval_res = cp(self.blank)
        defects = cv2.countNonZero(defect_img)
        ok_defects = defects < thresh
        ok_center = (50 < center[0] < 250) and (80 < center[1] < 170)
        cv2.putText(self.eval_res, f"{sensitivity}", (35, 30), self.font, self.fontScale, self.blue, self.lineType)
        cv2.putText(self.eval_res, f"{thresh}<", (10, 60), self.font, self.fontScale, self.blue, self.lineType)

        if ok_defects:
            cv2.putText(self.eval_res, f"{defects}", (60, 60), self.font, self.fontScale, self.green, self.lineType)
        else:
            cv2.putText(self.eval_res, f"{defects}", (60, 60), self.font, self.fontScale, self.red, self.lineType)

        if ok_center:
            cv2.putText(self.eval_res, "OK", (40, 90), self.font, self.fontScale, self.green, self.lineType)
        else:
            cv2.putText(self.eval_res, "MIMO", (30, 90), self.font, self.fontScale, self.red, self.lineType)
        return (ok_defects and ok_center)


class Evaluate:
    def __init__(self):
        self.blue = [51, 153, 255]
        self.green = [0, 255, 0]
        self.red = [255, 0, 0]

        self.blank = np.zeros((100, 100, 3), np.uint8)

        # self.blank[:, 32:33] = self.blue
        # self.blank[50:51, :66] = self.blue
        # self.blank[:, 65:66] = self.blue

        self.eval_res = np.zeros((100, 100), np.uint8)

        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.fontScale = 0.5
        self.lineType = 1
        self.scores_pos = {"zalis": (35, 60), "uzaver": (35, 90), "dscore": (35, 30)}

        cv2.putText(self.blank, "hranice vad:", (1, 15), self.font, self.fontScale, self.blue, self.lineType)
        cv2.putText(self.blank, "vady zalis:", (1, 45), self.font, self.fontScale, self.blue, self.lineType)
        cv2.putText(self.blank, "vady uzaver:", (1, 75), self.font, self.fontScale, self.blue, self.lineType)

    def compute(self, score, dscore):
        dscore = int(dscore / 1020 * 100)
        self.eval_res = cp(self.blank)
        vady_zalis = len(score[0][score[0] <= 0.7])
        vady_uzaver = len(score[1][score[1] <= 0.5])
        ok_zalis = not(vady_zalis > dscore)
        ok_uzaver = not(vady_uzaver > dscore)
        OK = ok_zalis and ok_uzaver

        if ok_zalis:
            cv2.putText(self.eval_res, str(vady_zalis), self.scores_pos["zalis"], self.font, self.fontScale, self.green, self.lineType)
        else:
            cv2.putText(self.eval_res, str(vady_zalis), self.scores_pos["zalis"], self.font, self.fontScale, self.red, self.lineType)

        if ok_uzaver:
            cv2.putText(self.eval_res, str(vady_uzaver), self.scores_pos["uzaver"], self.font, self.fontScale, self.green, self.lineType)
        else:
            cv2.putText(self.eval_res, str(vady_uzaver), self.scores_pos["uzaver"], self.font, self.fontScale, self.red, self.lineType)

        if OK:
            cv2.putText(self.eval_res, str(dscore), self.scores_pos["dscore"], self.font, self.fontScale, self.green, self.lineType)
        else:
            cv2.putText(self.eval_res, str(dscore), self.scores_pos["dscore"], self.font, self.fontScale, self.red, self.lineType)

        return OK
