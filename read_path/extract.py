import os
import platform
import inspect
import FaceMe.FaceMePython3SDK as FaceMe
from FaceMe.LicenseManager import LicenseManager
from FaceMe.DataManager import DataManager
from FaceMe.Recognizer import Recognizer
from FaceMe.CameraRecognizer import CameraRecognizer
from FaceMe.QualityDetector import QualityDetector
from FaceMe.SpoofingManager import SpoofingManager
import  cv2
from FaceMe.FaceMeSDK import FR_FAILED

image_path = "0820_KEVIN.jpg"
ret = FaceMe.FR_RETURN_OK
ret = FaceMe.FR_RETURN_OK
recognizer = Recognizer()


class FaceMeSDK():
    def __init__(self):
        self.__license_manager = None

        self.__data_manager = None
        self.__feature_scheme = None

        self.__camera_reocognizer_data_manager = None
        self.__camera_recognizer_feature_scheme = None

        self.__camera_reocognizer_adv_data_manager = None
        self.__camera_recognizer_adv_feature_scheme = None

        self.__recognizer = None
        self.__camera_capturers = {}
        self.__camera_recognizer = None
        self.__quality_detector = None
        self.__spoofing_manager = None
        self.__utils = None

        self.__license_key = ""
        self.__app_bundle_path = ""
        self.__app_data_path = ""
        self.__app_cache_path = ""

        self.__license_manager_config = {}
        self.__recognizer_config = {}
        self.__data_manager_config = {}
        self.__camera_recognizer_config = {}
        self.__camera_recognizer_callback = None

        self.__quality_detector_config = {}

        self.__spoofing_manager_config = {}
        self.__spoofing_result_handler = None
        if self.__recognizer:
            ret = self.__recognizer.Finalize()
            if FR_FAILED(ret):
                return ret
            print(type(ret))

"""""""""""
        ret = recognizer.Initialize(self.__recognizer_config)
if FR_SUCC(ret):
    print("faceMe_image",image_path)
print("faceMe_image_type",type(image_path))

def create__initialize(self):
    ret = self.__create_recognzier_and_initialize()
    if FR_FAILED(ret):
        print("Recognizer initialize failed")
"""""""""""""""            
#ret, faceMe_image = FaceMeSDK.__utils.ConvertToFaceMeImage(image_path)

"""""
global faceMe_sdk
faceMe_sdk = FaceMeSDK()
img = "0820_KEVIN.jpg"
ret, image =faceMe_sdk.convert_to_faceMe_image(img)
"""""


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

