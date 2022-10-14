import cv2
import math
import numpy as np
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier

class HandSignDetection():
    def build(self):
        self.capture = cv2.VideoCapture(0)
        self.classifier = Classifier("keras_model.h5", "labels.txt")
        self.detector = HandDetector(maxHands=1)
        self.offset = 20
        self.bgSize = 96
        self.labels = ["Look", "Drink", "Eat", "Ok"]
        return self.run()
    
    def run(self):
        while True:
            _, frame = self.capture.read()
            hands, frame = self.detector.findHands(frame)
            try:
                if hands:
                    hand = hands[0]
                    x, y, w, h = hand['bbox']
                    croppedHand = np.ones((self.bgSize, self.bgSize, 3), np.uint8) * 12
                    imgCrop = frame[y - self.offset:y + h + self.offset, x - self.offset:x + w + self.offset]
                    aspectRatio = h / w
                    if aspectRatio > 1:
                        constant = self.bgSize / h
                        wComputed = math.floor(constant * w)
                        bgResize = cv2.resize(imgCrop, (wComputed, self.bgSize))
                        bgResizeShape = bgResize.shape
                        wGap = math.floor((self.bgSize-wComputed)/2)
                        croppedHand[:bgResizeShape[0], wGap:wGap + wComputed] = bgResize
                    else:
                        constant = self.bgSize / w
                        hComputed = math.floor(constant * h)
                        bgResize = cv2.resize(imgCrop, (self.bgSize, hComputed))
                        bgResizeShape = bgResize.shape
                        hGap = math.floor((self.bgSize - hComputed) / 2)
                        croppedHand[hGap: hComputed + hGap, :] = bgResize
                    _, index = self.classifier.getPrediction(croppedHand, draw= False)
                    cv2.putText(frame, self.labels[index], (30, 30), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            except:
                continue
            cv2.imshow("Image", frame)
            if (cv2.waitKey(1) == ord('q')):
                break
        cv2.destroyAllWindows()

if __name__ == '__main__':
    HandSignDetection().build()
