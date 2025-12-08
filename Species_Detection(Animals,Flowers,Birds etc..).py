# ensemble_tta_species.py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as prep_resnet, decode_predictions
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as prep_eff
from PIL import Image
import cv2

MODEL_INPUT = (224,224)

resnet = ResNet50(weights="imagenet", input_shape=(224,224,3))
effnet = EfficientNetB0(weights="imagenet", input_shape=(224,224,3))

def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def get_tta_crops(pil_img, size=(224,224)):
    w, h = pil_img.size
    tw, th = size
    # if image smaller, upscale
    if w < tw or h < th:
        pil_img = pil_img.resize((max(w,tw), max(h,th)))
        w, h = pil_img.size
    crops = []
    # center crop
    cx = (w - tw)//2
    cy = (h - th)//2
    crops.append(pil_img.crop((cx, cy, cx+tw, cy+th)))
    # four corner crops
    corners = [(0,0), (w-tw,0), (0,h-th), (w-tw, h-th)]
    for (x,y) in corners:
        crops.append(pil_img.crop((x, y, x+tw, y+th)))
    # horizontal flips of each
    flips = [c.transpose(Image.FLIP_LEFT_RIGHT) for c in crops]
    crops.extend(flips)
    return crops

def preprocess_for_model(pil_img, model_name):
    arr = np.array(pil_img).astype("float32")
    if model_name == "resnet":
        arr = cv2.resize(arr, MODEL_INPUT)
        arr = prep_resnet(arr)
    else:
        arr = cv2.resize(arr, MODEL_INPUT)
        arr = prep_eff(arr)
    return arr

def predict_ensemble_tta(img_path, topk=5):
    pil = load_image(img_path)
    crops = get_tta_crops(pil, size=MODEL_INPUT)
    preds_res = []
    preds_eff = []
    for c in crops:
        a_res = preprocess_for_model(c, "resnet")
        a_eff = preprocess_for_model(c, "effnet")
        preds_res.append(resnet.predict(np.expand_dims(a_res,0), verbose=0)[0])
        preds_eff.append(effnet.predict(np.expand_dims(a_eff,0), verbose=0)[0])
    preds_res = np.array(preds_res)
    preds_eff = np.array(preds_eff)
    # average over crops
    avg_res = preds_res.mean(axis=0)
    avg_eff = preds_eff.mean(axis=0)
    # ensemble average (simple mean)
    avg = (avg_res + avg_eff) / 2.0
    # get topk indices
    top_idx = avg.argsort()[-topk:][::-1]
    # decode labels using ResNet decode (both use same imagenet ids)
    decoded = []
    for i in top_idx:
        # we need the human label; use ResNet decode by creating a 1x1000 array placeholder
        dummy = np.zeros((1,1000))
        dummy[0,i] = 1.0
        label = decode_predictions(dummy, top=1)[0][0][1]
        score = avg[i]
        decoded.append((i, label, float(score)))
    return decoded

if __name__ == "__main__":
    img_path = input("Enter image path: ")
    res = predict_ensemble_tta(img_path, topk=5)
    print("\nTop predictions (ensemble + TTA):")
    for idx, label, score in res:
        print(f"{label}: {score:.4f}")
