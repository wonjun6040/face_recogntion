import ctypes
import time

import cv2
import numpy as np
import dlib
from scipy.spatial.distance import euclidean, cosine


import facenet


class face_recognition:
    def __init__(self):
        self.model = facenet.InceptionResNetV2()
        self.detector = dlib.get_frontal_face_detector()
        self.landmark = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
        self.cap = cv2.VideoCapture(0)
        self.save_face = None
        self.threshold = 0.4

    def recognition(self):
        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dets = self.detector(gray, 1)

            num_faces = len(dets)


            faces = dlib.full_object_detections()
            for detection in dets:
                faces.append(self.landmark(img, detection))



            try:
                images = dlib.get_face_chips(img, faces, size=160)
                image = images[0]
                image = (image / 255.).astype(np.float32)
                cv2.imshow('image', image)
                if self.save_face is None:
                    if cv2.waitKey(1) == ord('c'):
                        self.save_face = self.model.predict(np.expand_dims(image, axis=0))[0]
                        cv2.destroyAllWindows()
                        time.sleep(3)

                else:
                    embedding_vector = self.model.predict(np.expand_dims(image, axis=0))[0]
                    print(cosine(self.save_face, embedding_vector))
                    if cosine(self.save_face, embedding_vector) > self.threshold:
                        print('인증 실패')
                        ctypes.windll.user32.LockWorkStation()
                        exit()
                if cv2.waitKey(1) == ord('q'):
                    exit()


            except RuntimeError:
                continue



a = face_recognition()
a.recognition()



