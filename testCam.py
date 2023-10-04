# -*- coding: utf-8 -*-

import cvCam
import cv2
import time
import platform
system = platform.system()


def queryCamInfosOnLinux(includeGethermal=False):
    import glob
    import os
    deviceInfo = []
    filePathList = glob.glob('/sys/bus/usb/drivers/uvcvideo/*')
    for filePath in filePathList:
        v4l_path = '%s/video4linux' % filePath
        if not os.path.isdir(v4l_path):
            continue

        _ = glob.glob('%s/video*' % v4l_path)
        dev_name = os.path.basename(_[0])
        dev_path = '/dev/%s' % dev_name
        dev_index = dev_name[dev_name.rindex(
            'video') + len('video'): len(dev_name)]

        _ = '%s/%s/name' % (v4l_path, dev_name)
        product_name = open(_).readline().strip()

        if "PureThermal" in product_name and not includeGethermal:
            continue

        deviceInfo.append((int(dev_index), dev_path, product_name))

    return deviceInfo


class TestCam:
    def __init__(self):

        if not 'Windows' in system:
            camInfos = queryCamInfosOnLinux(
                False)  # False �j�M���]�t�������ṳ��
            try:
                (index, _, _) = camInfos[0]
            except IndexError:
                raise 'can not find any usb cam'
        else:
            index = 0

        self.cvCam = cvCam.CvCam(index, self)

    def run(self):
        self.cvCam.start()

    def showImg(self, windowName, frame):
        cv2.imshow(windowName, frame)
        cv2.waitKey(1)

    def test(self):
        self.cvCam.test()

    def destroy(self):
        self.cvCam.release()


if __name__ == '__main__':
    testCam = TestCam()
    testCam.test()
    testCam.destroy()
