import ctypes
import time

import cv2
import numpy as np
import dlib

import facenet


class face_recognition:
    def __init__(self):
        self.model = facenet.InceptionResNetV2()
        self.detector = dlib.get_frontal_face_detector() 
        self.landmark = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
        self.cap = cv2.VideoCapture(0)
        self.save_face = None
        self.same = True
        self.threshold = 8.5

    def detection(self):
        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            dets = self.detector(img, 1)

            num_faces = len(dets)

            
            faces = dlib.full_object_detections()
            for detection in dets:
                faces.append(self.landmark(img, detection))

            try:
                images = dlib.get_face_chips(img, faces, size=160)
                image = images[0]
                image = (image / 255.).astype(np.float32)
                cv2.imshow('image', image)
                if cv2.waitKey(1) == ord('c'):
                    self.save_face = self.model.predict(np.expand_dims(image, axis=0))[0]
                    cv2.destroyAllWindows()
                    time.sleep(3)
                    break
                elif cv2.waitKey(1) == ord('q'):
                    exit()


            except RuntimeError:
                continue

    def recogniton(self):

        while True:
            ret, img = self.cap.read()
            if not ret:
                break

            dets = self.detector(img, 1)

            num_faces = len(dets)


            faces = dlib.full_object_detections()
            for detection in dets:
                faces.append(self.landmark(img, detection))



            try:
                images = dlib.get_face_chips(img, faces, size=160)
                image = images[0]
                image = (image / 255.).astype(np.float32)
                embedding_vector = self.model.predict(np.expand_dims(image, axis=0))[0]
                cv2.imshow('a', image)
                print(self.findEuclideanDistance(self.save_face, embedding_vector))
                if cv2.waitKey(1) == ord('c'):
                    print(self.findEuclideanDistance(self.save_face, embedding_vector))
                    if self.findEuclideanDistance(self.save_face, embedding_vector) > self.threshold:
                        self.same = False
                        break
                    else:
                        break
                elif cv2.waitKey(1) == ord('q'):
                    exit()

            except RuntimeError:
                continue

    def findEuclideanDistance(self, source_representation, test_representation):
        if type(source_representation) == list:
            source_representation = np.array(source_representation)

        if type(test_representation) == list:
            test_representation = np.array(test_representation)

        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance

    def verification(self):

        if self.same:
            print('인증 완료')
        else:
           ctypes.windll.user32.LockWorkStation()



a = face_recognition()
a.detection()
a.recogniton()
a.verification()

