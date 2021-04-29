import dlib
from keras_vggface.vggface import VGGFace
import pickle
import cv2
import tensorflow as tf

def set_gpu_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            # print(len(gpus), "Physical GPUs,", len(logical_gpus),
            #       "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

class BMIModels:
    cnn_face_detector = None
    hog_face_detector = None
    vggface_embeddings = None
    xgb_regr = None
    age_net = None

    @staticmethod
    def get_cnn_face_detector():
        if BMIModels.cnn_face_detector is None:
            BMIModels()
        return BMIModels.cnn_face_detector

    @staticmethod
    def get_age_net():
        if BMIModels.age_net is None:
            BMIModels()
        return BMIModels.age_net

    @staticmethod
    def get_hog_face_detector():
        if BMIModels.hog_face_detector is None:
            BMIModels()
        return BMIModels.hog_face_detector

    @staticmethod
    def get_vggface_embeddings():
        if BMIModels.vggface_embeddings is None:
            BMIModels()
        return BMIModels.vggface_embeddings

    @staticmethod
    def get_xgb_regr():
        if BMIModels.xgb_regr is None:
            BMIModels()
        return BMIModels.xgb_regr

    def __init__(self):
        if (BMIModels.cnn_face_detector is not None) and \
        (BMIModels.hog_face_detector is not None) and \
        (BMIModels.vggface_embeddings is not None) and \
        (BMIModels.xgb_regr is not None) and \
        (BMIModels.age_net is not None):
            raise Exception("This class is a singleton!")

        set_gpu_growth()
        BMIModels.cnn_face_detector = dlib.cnn_face_detection_model_v1(
            'models/mmod_human_face_detector.dat')
        BMIModels.hog_face_detector = dlib.get_frontal_face_detector()
        BMIModels.vggface_embeddings = VGGFace(
            model='senet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        with open('models/regressor.pkl', 'rb') as f:
            BMIModels.xgb_regr = pickle.load(f)
        BMIModels.age_net = cv2.dnn.readNetFromCaffe(
            'models/DEX_real_age_testing.prototxt',
            'models/DEX_real_age.caffemodel')
