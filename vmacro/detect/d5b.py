import os
import cv2
import numpy as np
from gtuner import *
import gtuner
import time
from random import randrange
import queue
import threading
import time

'''
<version>1.0</version>

<shortdesc>
DJMax Trilogy: Auto Play
Computer Vision : Detects notes and Auto play it
<i>Tested @ 1920x1080 input using PS Remote Play</i>
</shortdesc>

<keywords>Djmax Respect, Djmax, Computervision, CV, Auto, </keywords>

<donate>N/A</donate>
<docurl>N/A</docurl>
'''

class GCVWorker:
    def __init__(self, width, height):
        self.gcvdata = bytearray([0xFF]*10)
        self.keyTiming = [0,0,0,0,0,0,0,0,0,0]
        self.width = width
        self.height = height
        self.white5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/whiteNote5B.jpg', dtype=np.uint8), -1)
        self.color5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/colorNote5B.jpg', dtype=np.uint8), -1)
        self.whiteHoldS5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/whiteHoldS5B.jpg', dtype=np.uint8), -1)
        self.whiteHoldM5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/whiteHoldM5B.jpg', dtype=np.uint8), -1)
        self.whiteHoldE5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/whiteHoldE5B.jpg', dtype=np.uint8), -1)
        self.colorHoldS5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/colorHoldS5B.jpg', dtype=np.uint8), -1)
        self.colorHoldM5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/colorHoldM5B.jpg', dtype=np.uint8), -1)
        self.colorHoldE5b = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/colorHoldE5B.jpg', dtype=np.uint8), -1)
        self.sideHoldS = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/sideHoldS.jpg', dtype=np.uint8), -1)
        self.sideHoldM = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/sideHoldM.jpg', dtype=np.uint8), -1)
        self.sideHoldE = cv2.imdecode(np.fromfile('C:/Users/mdnpm/Desktop/projects/GTuner/gTuner-configs-master/CV/djmax_respect/template/sideHoldE.jpg', dtype=np.uint8), -1)
        self.sideColorBoundaryV1 = [ #RGB Boundary
            [177,169,25], [215,220,95]
        ]
        self.threshold = 0.85 # 0.85 default, 0.63 pixelated
        self.isDebug = False
        self.bak = None
        self.lastRes = -1

    def process(self, frame):
        clean = frame.copy()
        # frame = cv2.drawContours(frame, [self.contour1], 0, (255,255,255), 1)
        # frame = cv2.imread('test/waitFail.png')

        # threshold = 0.9
        # loc = np.where(result >= threshold)
        # print(len(loc))
        # for pt in zip(*loc[::-1]):
        #     cv2.rectangle(frame, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        q = queue.Queue()
        worker = [None] * 7
        results = [0.00] * 10

        for i in range(7 if not self.isDebug else 0):
            if(i == 0):
                worker[i] = threading.Thread(target=self.detectFirstWhite5b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 1):
                worker[i] = threading.Thread(target=self.detectSecondWhite5b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 2):
                worker[i] = threading.Thread(target=self.detectThirdWhite5b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 3):
                worker[i] = threading.Thread(target=self.detectFirstColor5b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 4):
                worker[i] = threading.Thread(target=self.detectSecondColor5b, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 5):
                worker[i] = threading.Thread(target=self.detectLeftSide, args=(q, frame, clean, results, i,), daemon=True)
            elif(i == 6):
                worker[i] = threading.Thread(target=self.detectRightSide, args=(q, frame, clean, results, i,), daemon=True)
            worker[i].start()
            q.put(i)

        if not self.isDebug:
            q.join()
        # # RESULTS HERE
        # for i in range(len(results)):
        #     print(" ".join(str(results)))
        # print(self.processData5B(results))
        return (frame, self.processData5B(results)) if not self.isDebug else (frame, None)

    def processData5B(self, resultArr):
        # for i in range(len(resultArr)):
        #     if(self.keyTiming[i] == 0 and resultArr[i] >= self.threshold):
        #         self.keyTiming[i] = 1
        #     elif(self.keyTiming[i] == 1 and resultArr[i] < self.threshold):
        #         self.keyTiming[i] = 2

        # buildArray = [0x00]*len(resultArr)
        # for i in range(len(resultArr)):
        #     if(self.keyTiming[i] == 2):
        #         buildArray[i] = 0x01
        #         self.keyTiming[i] = 0

        buildArray = [0x00]*len(resultArr)
        buildArray[0] = 0x01 if resultArr[0] >= self.threshold else 0x00
        buildArray[1] = 0x01 if resultArr[1] >= self.threshold else 0x00
        buildArray[3] = 0x01 if resultArr[2] >= self.threshold else 0x00
        buildArray[4] = 0x01 if resultArr[3] >= self.threshold else 0x00
        buildArray[5] = 0x01 if resultArr[4] >= self.threshold else 0x00
        buildArray[6] = 0x01 if resultArr[5] >= self.threshold else 0x00
        buildArray[7] = 0x01 if resultArr[6] >= self.threshold else 0x00

        return bytearray(buildArray)

    def genericDetect(self, frame, clean, template, xStart, xEnd, debugName, debugX, debugY, activeColor, activeOnly, q):
        t0 = time.perf_counter()
        cleanCropped = clean[0:self.height, xStart:xEnd]
        chan,w,h = template.shape[::-1]
        result = cv2.matchTemplate(cleanCropped, template, 5)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        if(self.threshold < max_val):
            frame = cv2.rectangle(frame,(top_left[0] + xStart, top_left[1]), (bottom_right[0] + xStart, bottom_right[1]), 255, 2)
        t1 = time.perf_counter()
        if(self.threshold < max_val or activeOnly):
            cv2.putText(frame, debugName + str('{0:.2f}'.format(max_val)) + str(top_left) + " " + '{0:.5f}'.format(1000*(t1-t0)) + "ms", (debugX, debugY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, activeColor if self.threshold < max_val else (255, 255, 255), 1, cv2.LINE_AA)
        return max_val if not self.isDebug else max_val
        # return cleanCropped if self.isDebug else frame

    def genericSideDetect(self, frame, clean, xStart, xEnd, debugName, debugX, debugY, q):
        if not self.isDebug:
            q.get()

        t0 = time.perf_counter()
        result = self.isColorInRangeOfSideV1(cv2.mean(clean[60:80, xStart:xEnd]))
        t1 = time.perf_counter()
        cv2.putText(frame, debugName + str('{0:.2f}'.format(result)) + " " + '{0:.5f}'.format(1000*(t1-t0)) + "ms", (debugX, debugY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255) if result == 1.0 else (255, 255, 255), 1, cv2.LINE_AA)

        if self.isDebug:
            frame = clean[40:60, xStart:xEnd]
            breh = cv2.mean(clean[40:60, xStart:xEnd])
            if self.bak == None:
                self.bak = breh
                print(breh)
            if (breh != self.bak):
                self.bak = breh
                print(breh)

        if not self.isDebug:
            q.task_done()

        return result if not self.isDebug else frame

    def detectFirstWhite5b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.whiteHoldS5b, 22, 214, "5B1WHS: ", 2, 13, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.whiteHoldM5b, 22, 214, "5B1WHM: ", 2, 13, 	(0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.whiteHoldE5b, 22, 214, "5B1WHE: ", 2, 13, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.white5b, 22, 214, "5B1W: ", 2, 13, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()
        # self.genericDetect(frame, clean, self.whiteHoldS5b, 22, 214, "5B1HS: ", 2, 13, (0, 255, 0), True, q)
        # self.genericDetect(frame, clean, self.whiteHoldM5b, 22, 214, "5B1HM: ", 2, 33, 	(0, 215, 255), True, q)
        # self.genericDetect(frame, clean, self.whiteHoldE5b, 22, 214, "5B1HE: ", 2, 53, (0, 255, 255), True, q)
        # self.genericDetect(frame, clean, self.white5b, 22, 214, "5B1W: ", 2, 73, (0, 255, 0), True, q)

    def detectSecondWhite5b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.whiteHoldS5b, 407, 599, "5B2WHS: ", 2, 53, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.whiteHoldM5b, 407, 599, "5B2WHM: ", 2, 53, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.whiteHoldE5b, 407, 599, "5B2WHE: ", 2, 53, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.white5b, 407, 599, "5B2W: ", 2, 53, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.white5b, 338, 501, "5B2W: ", 2, 53, q)

    def detectThirdWhite5b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.whiteHoldS5b, 791, 983, "5B3WHS: ", 2, 73, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.whiteHoldM5b, 791, 983, "5B3WHM: ", 2, 73, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.whiteHoldE5b, 791, 983, "5B3WHE: ", 2, 73, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.white5b, 791, 983, "5B3W: ", 2, 73, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.white5b, 791, 983, "5B3W: ", 2, 73, q)

    def detectFirstColor5b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.colorHoldS5b, 215, 407, "5B1CHS: ", 2, 33, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.colorHoldM5b, 215, 407, "5B1CHM: ", 2, 33, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.colorHoldE5b, 215, 407, "5B1CHE: ", 2, 33, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.color5b, 215, 407, "5B1C: ", 2, 33, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.color5b, 215, 407, "5B1C: ", 2, 33, q)

    def detectSecondColor5b(self, q, frame, clean, results, i):
        if not self.isDebug:
            q.get()
        temp = self.genericDetect(frame, clean, self.colorHoldS5b, 599, 791, "5B2CHS: ", 2, 93, (0, 255, 0), False, q)
        if temp >= self.threshold:
            results[i] = temp
            if not self.isDebug:
                q.task_done()
        else:
            temp = self.genericDetect(frame, clean, self.colorHoldM5b, 599, 791, "5B2CHM: ", 2, 93, (0, 215, 255), False, q)
            if temp >= self.threshold:
                results[i] = temp
                if not self.isDebug:
                    q.task_done()
            else:
                temp = results[i] = self.genericDetect(frame, clean, self.colorHoldE5b, 599, 791, "5B2CHE: ", 2, 93, (0, 255, 255), False, q)
                if temp >= self.threshold:
                    results[i] = temp
                    if not self.isDebug:
                        q.task_done()
                else:
                    results[i] = self.genericDetect(frame, clean, self.color5b, 599, 791, "5B2C: ", 2, 93, (0, 255, 0), True, q)
                    if not self.isDebug:
                        q.task_done()

        # results[i] = self.genericDetect(frame, clean, self.color5b, 599, 791, "5B2C: ", 2, 93, q)

    def detectLeftSide(self, q, frame, clean, results, i):
        results[i] = self.genericSideDetect(frame, clean, 23, 236, "SideL: ", 2, 133, q)

    def detectRightSide(self, q, frame, clean, results, i):
        results[i] = self.genericSideDetect(frame, clean, 503, 716, "SideR: ", 2, 153, q)

    def isColorInRangeOfSideV1(self, rgb):
        result = 0.00
        if len(rgb) == 4:
            r,g,b,a = (rgb)
        else:
            r,g,b = (rgb)

        if(b >  self.sideColorBoundaryV1[0][2] and b <  self.sideColorBoundaryV1[1][2]):#bluecheck
            if(g >  self.sideColorBoundaryV1[0][1] and g <  self.sideColorBoundaryV1[1][1]):#greencheck
                if(r >  self.sideColorBoundaryV1[0][0] and r <  self.sideColorBoundaryV1[1][0]):#redcheck
                    return 1.00

        return result

    def samplePerformance(self, frame):
        t0 = time.perf_counter()
        frame = frame[0:self.height, 500:1000]
        t1 = time.perf_counter()
        print("Function=%s, Time=%s" % ('crop', t1 - t0))
        print("Function=%s, Time=%s" % ('Total', t1 - t0))

    def __del__(self):
        del self.gcvdata
        del self.keyTiming
        del self.width
        del self.height
        del self.white5b
        del self.color5b
        del self.whiteHoldS5b
        del self.whiteHoldM5b
        del self.whiteHoldE5b
        del self.colorHoldS5b
        del self.colorHoldM5b
        del self.colorHoldE5b
        del self.sideHoldS
        del self.sideHoldM
        del self.sideHoldE
        del self.sideColorBoundaryV1
        del self.threshold
        del self.isDebug
        del self.bak
        del self.lastRes
