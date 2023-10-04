from os import wait
import cv2
import FaceMe.FaceMePython3SDK as FaceMe
from FaceMe.FaceMeSDK import FaceMeSDK, FR_FAILED,FR_SUCC
from FaceMe.Recognizer import Recognizer

faceMe_images = ""
faceMe_images = '0820_KEVIN.jpg'
global faceMe_sdk
ret, img = faceMe_sdk.convert_to_faceMe_image(faceMe_images)
if FR_FAILED(ret):
    print("Recognizer initialize failed")

'''''''''''''''''
recognizer = Recognizer()
ret = recognizer.Initialize(FaceMeSDK.__recognizer)
if FR_SUCC(ret):
    FaceMeSDK.__recognizer = recognizer
    if FaceMeSDK.__feature_scheme is None:
        ret, FaceMeSDK.__feature_scheme = FaceMeSDK.__recognizer.GetFeatureScheme()
        if FR_FAILED(ret):
            FaceMeSDK.__feature_scheme = None
        print (type (ret))

''''''''''''''''''''
ret = cv2.imshow() 
        print (ret, 0)
        cv2.waitKey(1)
ret, img = cv2.imshow()
print("recognizer_image",ret)     
cv2.destroyAllWindows()        
'''''''''''''''''''''

"""""
def recognize_faces(self, faceMe_images , options=None):
        print(type(faceMe_images ))
        ret = FaceMe.FR_RETURN_OK
        ret = self.__create_recognzier_and_initialize()
        if FR_FAILED(ret):
            print("Recognizer initialize failed")
            return ret, []
        
        if options and type(options) is dict:
            argd = options.copy()
        else:
            argd = {}

        ret, face_counts = self.__recognizer.RecognizeFace(faceMe_images, argd)
        image_size = len(faceMe_images)
        face_results = []
        for image_index in range(0, image_size):
            for face_index in range(0, face_counts[image_index]):
                __, face_info_dict = self.__recognizer.GetFaceInfo(image_index, face_index)
                __, face_landmark_dict = self.__recognizer.GetFaceLandmark(image_index, face_index)
                __, face_attribute_dict = self.__recognizer.GetFaceAttribute(image_index, face_index)
                __, face_feature_dict = self.__recognizer.GetFaceFeature(image_index, face_index)
                filter_result = FilterResult(face_info_dict, face_landmark_dict, argd.get('extractOptions', FaceMe.FR_FEATURE_OPTION_ALL), face_attribute_dict, face_feature_dict)
                face_result = {'imageIndex': image_index, 'faceIndex': face_index, **filter_result}
                face_results.append(face_result)
                #print(type(face_results))
        return ret, face_results
        
""""""""
ret = FaceMe.FR_RETURN_OK
ret = image.__create_recognzier_and_initialize()
if FR_FAILED(ret):
    print("Recognizer initialize failed")
    print (ret, {})
"""""""""      

"""""
global faceMe_sdk
faceMe_sdk = FaceMeSDK()
img = "0820_KEVIN.jpg"
ret, image =faceMe_sdk.convert_to_faceMe_image(img)
"""""