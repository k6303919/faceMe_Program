import FaceMe
import os
import platform
import inspect
from time import sleep
from FaceMe.FaceMePython3SDK import WebcamInfo
from FaceMe.Recognizer import Recognizer
from FaceMe.LicenseManager import LicenseManager
import FaceMe.FaceMePython3SDK as FaceMe
#from FaceMe.FaceMeSDK import  FR_FAILED, FaceMeSDK
import FaceMe.FaceMeSDK as FC
import cv2
f = FC.FaceMeSDK()
class FaceMeSDK():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Prepare the camera...
        print("Camera warming up ...")


    def get_frame(self):

        s, img = self.cap.read()
        if s:  # frame captures without errors...

            pass

        return img

    def release_camera(self):
        self.cap.release()


def main():
        while True: #True
            cam1 = FaceMeSDK().get_frame()
            frame = cv2.resize(cam1, (0, 0), fx = 0.75, fy = 0.75)
            frame2 = f.convert_opencvMat__to_faceMe_image(frame)
            cv2.imshow("Frame", frame2)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            FaceMeSDK().release_camera()
        return ()
if __name__ == '__main__':
    main()
    cv2.destroyAllWindows()
