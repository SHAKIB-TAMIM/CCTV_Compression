import cv2
import os

class Detector:
    def __init__(self, model_type='face'):
        self.model_type = model_type
        if self.model_type == 'face':
            casc_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            if not os.path.exists(casc_path):
                raise FileNotFoundError("Haarcascade not found.")
            self.detector = cv2.CascadeClassifier(casc_path)
        else:
            raise ValueError("Unsupported model_type")

    def detect(self, frame, scaleFactor=1.1, minNeighbors=5, minSize=(30,30)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = self.detector.detectMultiScale(gray, scaleFactor=scaleFactor,
                                               minNeighbors=minNeighbors,
                                               minSize=minSize)
        return rects.tolist() if len(rects) else []
