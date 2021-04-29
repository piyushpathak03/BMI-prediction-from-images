import cv2
import numpy as np
import base64
import cv2
from model_loader import BMIModels
import tensorflow as tf


MARGIN = 0.2
AGE_LIST = list(range(101))
INPUT_SIZE = (224, 224)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

def base64_to_cv2image(b64_string):
    image_bytes = base64.b64decode(b64_string)
    cv_image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)
    return cv_image


def crop_face(img, face_detector, face_detection):
    f = face_detector(img, 1)
    h, w = img.shape[:2]
    if face_detection == 'cnn':
        x1 = int(f[0].rect.left() - MARGIN * w)
        y1 = int(f[0].rect.top() - MARGIN * h)
        x2 = int(f[0].rect.right() + MARGIN * w)
        y2 = int(f[0].rect.bottom() + MARGIN * h)
    elif face_detection == 'hog':
        x1 = int(f[0].left() - MARGIN * w)
        y1 = int(f[0].top() - MARGIN * h)
        x2 = int(f[0].right() + MARGIN * w)
        y2 = int(f[0].bottom() + MARGIN * h)
    crop_img = crop_image(img, x1, y1, x2, y2)
    return crop_img


def crop_image(img, x1, y1, x2, y2):
    if x1 < 0 or y1 < 0 or x2 > img.shape[1] or y2 > img.shape[0]:
        img, x1, x2, y1, y2 = pad_img_to_fit_bbox(img, x1, x2, y1, y2)
    return img[y1:y2, x1:x2, :]


def pad_img_to_fit_bbox(img, x1, x2, y1, y2):
    img = cv2.copyMakeBorder(img, - min(0, y1), max(y2 - img.shape[0], 0),
                             -min(0, x1), max(x2 - img.shape[1], 0), cv2.BORDER_REPLICATE)
    y2 += -min(0, y1)
    y1 += -min(0, y1)
    x2 += -min(0, x1)
    x1 += -min(0, x1)
    return img, x1, x2, y1, y2


def get_blob(face_image):
    return cv2.dnn.blobFromImage(face_image,
                                 1.0,
                                 INPUT_SIZE,
                                 MODEL_MEAN_VALUES,
                                 swapRB=False)


def model_load_bmi(face_detection):
    if face_detection == 'cnn':
        face_detector = BMIModels.get_cnn_face_detector()
    elif face_detection == 'hog':
        face_detector = BMIModels.get_hog_face_detector()
    vggface_embeddings_senet = BMIModels.get_vggface_embeddings()
    xgb_regr = BMIModels.get_xgb_regr()
    return face_detector, vggface_embeddings_senet, xgb_regr


def model_load_age(face_detection):
    if face_detection == 'cnn':
        face_detector = BMIModels.get_cnn_face_detector()
    elif face_detection == 'hog':
        face_detector = BMIModels.get_hog_face_detector()
    age_net = BMIModels.get_age_net()
    return face_detector, age_net


def predict_bmi(img, face_detector, vggface_embeddings_senet, xgb_regr, crop, face_detection):
    # face crop
    if crop:
        try:
            img = crop_face(img, face_detector, face_detection)
        except RuntimeError as re:
            print(str(re))
            return 0, 0
    face = cv2.resize(img, (224, 224))

    # BMI
    face = np.array([face])
    emb = vggface_embeddings_senet.predict([face])
    pred = xgb_regr.predict(emb)

    return pred[0]


def predict_bmi_from_image_file(file_path, crop=True, face_detection='hog'):
    face_detector, vggface_embeddings_senet, xgb_regr = model_load_bmi(
        face_detection)
    # image load
    img = cv2.imread(file_path)
    return predict_bmi(img, face_detector, vggface_embeddings_senet, xgb_regr, crop, face_detection)


def predict_bmi_from_base64_file(filepath, crop=True, face_detection='hog'):
    face_detector, vggface_embeddings_senet, xgb_regr = model_load_bmi(
        face_detection)
    with open(filepath, 'r') as f:
        b64_string = f.read()
    img = base64_to_cv2image(b64_string)
    return predict_bmi(img, face_detector, vggface_embeddings_senet, xgb_regr, crop, face_detection)


def predict_bmi_from_base64_string(b64_string, crop=True, face_detection='hog'):
    face_detector, vggface_embeddings_senet, xgb_regr = model_load_bmi(
        face_detection)
    img = base64_to_cv2image(b64_string)
    return predict_bmi(img, face_detector, vggface_embeddings_senet, xgb_regr, crop, face_detection)


def predict_age(img, face_detector, age_net, crop=True, face_detection='hog'):
    # face crop
    if crop:
        try:
            img = crop_face(img, face_detector, face_detection)
        except RuntimeError as re:
            print(str(re))
            return 0, 0
    face = cv2.resize(img, (224, 224))

    # AGE
    blob = get_blob(face)
    age_net.setInput(blob)
    predictions = age_net.forward()
    age = AGE_LIST[predictions[0].argmax()]

    return age


def predict_age_from_image_file(file_path, crop=True, face_detection='hog'):
    face_detector, age_net = model_load_age(face_detection)
    # image load
    img = cv2.imread(file_path)
    return predict_age(img, face_detector, age_net, crop, face_detection)


def predict_age_from_base64_file(filepath, crop=True, face_detection='hog'):
    face_detector, age_net = model_load_age(face_detection)
    with open(filepath, 'r') as f:
        b64_string = f.read()
    img = base64_to_cv2image(b64_string)
    return predict_age(img, face_detector, age_net, crop, face_detection)


def predict_age_from_base64_string(b64_string, crop=True, face_detection='hog'):
    face_detector, age_net = model_load_age(face_detection)
    img = base64_to_cv2image(b64_string)
    return predict_age(img, face_detector, age_net, crop, face_detection)
