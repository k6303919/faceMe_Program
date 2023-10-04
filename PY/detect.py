import FaceMe.FaceMePython3SDK as FaceMe
from FaceMe.FaceMeSDK import FaceMeSDK, FR_FAILED, FR_SUCC
import os
import cv2

faceMe_sdk = FaceMeSDK()
image_path1 = "123.jpg"
ret, image1 = faceMe_sdk.convert_to_faceMe_image(image_path1)
if FR_FAILED(ret):
    print("convert to Faceme Image failed")

image_path2 = "0820_KEVIN.jpg"
ret, image2 = faceMe_sdk.convert_to_faceMe_image(image_path2)
if FR_FAILED(ret):
    print("convert to Faceme Image failed")

ret, recognize_results1 = faceMe_sdk.recognize_faces([image1])
if FR_FAILED(ret):
    print("Recognize image1 failed, return: ", ret)

if len(recognize_results1)!=1:
    print("No face or mutiple face found in image1")

ret, recognize_results2 = faceMe_sdk.recognize_faces([image2])
if FR_FAILED(ret):
    print("Recognize image2 failed, return: ", ret)

if len(recognize_results2)!=1:
    print("No face or mutiple face found in image2")

ret, compare_result = faceMe_sdk.compare_face_feature(recognize_results1[0]
['faceFeatureStruct'], recognize_results2[0]['faceFeatureStruct'])
if FR_FAILED(ret):
    print("Compare face feature failed, return: ", ret)

print("Compare confidence: ", compare_result['confidence'], ", same Person: ",compare_result['isSamePerson'])