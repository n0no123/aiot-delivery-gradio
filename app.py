import av
import cv2
import math
import numpy as np
from streamlit_webrtc import webrtc_streamer
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

bgSize = 96
classifier = Classifier("keras_model.h5", "labels.txt")
detector = HandDetector(maxHands=1)
labels = ["Look", "Drink", "Eat", "Ok"]
offset = 20

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    hands, frame = detector.findHands(img)
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
            cv2.putText(frame, labels[index], (30, 30),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
    except Exception as e:
        print(e)
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="Hand Sign Detection",
    media_stream_constraints={"video": True, "audio": False},
    video_frame_callback=video_frame_callback,
)
