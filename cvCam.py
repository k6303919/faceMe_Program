import threading
import time
import cv2

# used to list usb devices
import os

from language import getText

import json


class CvCam:
    windowName = getText("title-of-usb-cam")
    capturing = False
    alive = False

    readSuccess = False
    frame = []
    faces = []

    _outputOriginalOnly = False

    _rebind = True

    _minSize = (60, 60)
    _maxSize = (600, 600)

    _cropRect = None

    _subArea = []
    _subRect = {}

    def __init__(self, camIndex, control=None, camWidth=1280, camHeight=720):
        self.camIndex = camIndex
        self.camInstance = cv2.VideoCapture(self.camIndex)
        print('open cvCam on camera %d' % camIndex)

        # use camera default resolution
        self.camWidth = self.camInstance.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.camHeight = self.camInstance.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print('cam resolution: (%d x %d)' % (self.camWidth, self.camHeight))

        try:
            # self.camWidth = camWidth
            # self.camInstance.set(cv2.CAP_PROP_FRAME_WIDTH, self.camWidth)
            # self.camHeight = camHeight
            # self.camInstance.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camHeight)
            print('cam resolution: (%d x %d)' %
                  (self.camWidth, self.camHeight))
        except Exception as e:
            print(e)

        self.control = control

        self.drawer = self.Drawer()

        self.faceCascade = cv2.CascadeClassifier('cascade.xml')

        self._subArea = self.loadSubArea()
        if self._subArea:
            print("casecade on %d sub area: " % len(self._subArea))
        else:
            print("cascade on main.")

    # Only load detection area if exist any
    def loadSubArea(self):
        area = []
        try:
            with (open("detections.config", "r", encoding='UTF-8')) as f:
                subAreaJsonArray = json.load(f)
                for subArea in subAreaJsonArray:
                    area.append(subArea)
                return area
        except Exception:
            print("detections load error")
            return area

    def getCamWidth(self):
        if self._cropRect != None:
            return self._cropRect[1]
        else:
            return self.camWidth

    def getCamHeight(self):
        if self._cropRect != None:
            return self._cropRect[3]
        else:
            return self.camHeight

    def outputOriginal(self):
        self._outputOriginalOnly = True

    def getFrame(self):
        return self.frame.copy()

    def getSubArea(self):
        return self._subArea

    def setCropArea(self, x=0, w=-1, y=0, h=-1):
        if w == -1:
            w = int(self.camWidth) - x

        if h == -1:
            h = int(self.camHeight) - y

        self._cropRect = (x, w, y, h)

    def getFaces(self):
        return self.faces

    def start(self):
        self.capturing = True
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        while (self.capturing and self.camInstance.isOpened):
            self.readSuccess, frame = self.camInstance.read()

            if self.readSuccess <= 0:
                continue

            self.frame = cv2.flip(frame, 1)

            if self._cropRect != None:
                self.frame = self.frame[
                    self._cropRect[2]: self._cropRect[2] + self._cropRect[3], self._cropRect[0]: self._cropRect[0] + self._cropRect[1]
                ]

            if not self.capturing:
                break

            if self.readSuccess <= 0:
                continue

            if not self.frame.any():
                continue

            if self._subArea:
                tempFaces = []

                width = self.getCamWidth()
                height = self.getCamHeight()
                for subArea in self._subArea:
                    key = str(subArea)
                    rect = self._subRect.get(key)
                    if rect == None:
                        rect = (
                            int(float(subArea["x"]) * width),
                            int(float(subArea["w"]) * width),
                            int(float(subArea["y"]) * height),
                            int(float(subArea["h"]) * height)
                        )
                        self._subRect[key] = rect
                        print("rect: ", rect)

                    subFrame = self.frame[
                        rect[2]: rect[2] + rect[3], rect[0]: rect[0] + rect[1]
                    ]
                    gray = cv2.cvtColor(subFrame, cv2.COLOR_BGR2GRAY)

                    faces = self.faceCascade.detectMultiScale(
                        gray,
                        scaleFactor=1.1,
                        minNeighbors=3,
                        minSize=self._minSize,
                        maxSize=self._maxSize,
                        flags=cv2.CASCADE_SCALE_IMAGE
                    )

                    # fixFaces = []
                    for (x, y, w, h) in faces:
                        tempFaces.append((x + rect[0], y + rect[2], w, h))

                self.faces = tempFaces
            else:
                gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

                self.faces = self.faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=self._minSize,
                    maxSize=self._maxSize,
                    flags=cv2.CASCADE_SCALE_IMAGE
                )

            if self._outputOriginalOnly:
                continue

            if not self.capturing:
                break

            for (x, y, w, h) in self.faces:
                self.drawer.drawRect(
                    self.frame,
                    x,
                    y,
                    w,
                    h)

            if self.control != None:
                self.control.showImg(self.windowName, self.frame)

        if not self._outputOriginalOnly and self.control != None:
            self.control.destroyImg(self.windowName)

        # release cam when stop
        self.camInstance.release()
        if self._rebind:
            self.camInstance = cv2.VideoCapture(self.camIndex)

    def stop(self):
        self.capturing = False

    def release(self):
        self._rebind = False
        if (self.capturing):
            self.stop()
        else:
            self.camInstance.release()

    def test(self):
        while (True):
            _, self.frame = self.camInstance.read()
            cv2.imshow(self.windowName, self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if cv2.waitKey(1) & 0xFF == ord('c'):
                cv2.imwrite("capture.jpg", self.frame)
                break

    class Drawer:
        # color (B, G, R)
        def drawRect(self, img, x, y, w, h, color=(0, 255, 0), thickness=2, showSize=False):
            cv2.rectangle(
                img,
                (x, y),
                (x + w, y + h),
                color,
                thickness
            )
