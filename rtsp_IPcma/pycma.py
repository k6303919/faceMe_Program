import cv2

cam=cv2.VideoCapture('http://192.168.0.158:8080?user=admin&pwd=test&action=stream')

while True:
    ret,img=cam.read()

    vis=img.copy()

    cv2.imshow('getCamera',vis)

    if 0xFF & cv2.waitKey(1) == 27:
        break
cv2.destroyAllWindows()

