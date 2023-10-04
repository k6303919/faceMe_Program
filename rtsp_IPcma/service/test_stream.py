
import threading
import cv2

import time

# mjpeg 影像串流 server 測試

cap = cv2.VideoCapture(0)

def startCap():
    while True:
        try:
            success, frame = cap.read()
            if success:
                if 'appServ' in globals():
                    appServ.stream.feed(frame)
                cv2.imshow('frame', frame)
        except Exception as e:
            print(e)
            pass
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

    import sys
    sys.exit()


if __name__ == '__main__':

    camThread = threading.Thread(target=startCap, daemon=True)
    camThread.start()

    global appServ
    import app_server as appServ
    appServ.start()
