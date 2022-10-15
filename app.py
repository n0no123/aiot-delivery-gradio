import cv2
import gradio as gr
import math
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

bgSize = 96
classifier = Classifier("keras_model.h5", "labels.txt")
detector = HandDetector(maxHands=1)
labels = ["Look", "Drink", "Eat", "Ok"]
offset = 20

def segment(image):
    result = "Abricot"
    hands, frame = detector.findHands(image)
    try:
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            croppedHand = np.ones((bgSize, bgSize, 3), np.uint8) * 12
            imgCrop = frame[y - offset:y + h +
                            offset, x - offset:x + w + offset]
            aspectRatio = h / w
            if aspectRatio > 1:
                constant = bgSize / h
                wComputed = math.floor(constant * w)
                bgResize = cv2.resize(imgCrop, (wComputed, bgSize))
                bgResizeShape = bgResize.shape
                wGap = math.floor((bgSize-wComputed)/2)
                croppedHand[:bgResizeShape[0],
                            wGap:wGap + wComputed] = bgResize
            else:
                constant = bgSize / w
                hComputed = math.floor(constant * h)
                bgResize = cv2.resize(imgCrop, (bgSize, hComputed))
                bgResizeShape = bgResize.shape
                hGap = math.floor((bgSize - hComputed) / 2)
                croppedHand[hGap: hComputed + hGap, :] = bgResize
            _, index = classifier.getPrediction(croppedHand, draw=False)
            return labels[index]
    except Exception as e:
        print(e)
    return 'No sign detected'

gr.interface.Interface(fn=segment, live=True, inputs=gr.Image(source='webcam', streaming=True), outputs="text").launch()
