import cv2
import numpy as np
import FaceMe.FaceMePython3SDK as FaceMe
from FaceMe.FaceMeSDK import FaceMeSDK, FR_FAILED, FR_SUCC
f = FaceMeSDK()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
while(cap.isOpened()):
    ret, frame = cap.read()
    image = ["gray.jpg"]
    ret, frame = f.convert_to_faceMe_image(image)
    result = f.recognize_faces(image.shape)
    print(result)
    cv2.imshow("result_img",image)
    cv2.imwrite("result.jpg",image)
    cv2.destroyAllWindows()
cap.release()
"""""
imgMat=cv2.imread(image_path)
cv2.imshow("test",imgMat)
"""""
#ret, detect_result = faceMe_sdk.detect_face(image,recognize_config)
#detect_config = {'detectOptions':FaceMe.FR_FEATURE_OPTION_BOUNDING_BOX | FaceMe.FR_FEATURE_OPTION_FEATURE_LANDMARK | FaceMe.FR_FEATURE_OPTION_ALL }
#recognize_config = {'extractOptions':FaceMe.face_attribute_option}
"""""
if FR_SUCC(ret):
    print ("Convert to FaceMe image SUCC")

if FR_FAILED(ret):
    print("Convert to FaceMe image failed")
"""""
"""""
cv2.imshow("output",frame)
if cv2.waitKey(1) & 0xFF == 27:
    cap.release()
cv2.destroyAllWindows()
"""""
"""""

cv2.waitKey(0)
"""""


cap.release()
cv2.destroyAllWindows()

"""""
cv2.imshow('test', ret)
"""""

#ret, detect_result = faceMe_sdk.detect_face(image,recognize_config)
"""
if FR_FAILED(ret):
    print("Detect face failed, return: ", ret)
"""
"""
if FR_FAILED(ret):
        print("Recognize face failed, return: ", ret)

#print("detect_result: ", detect_result)

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
"""







    


