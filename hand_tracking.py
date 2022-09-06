from lib2to3.pytree import Base
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class HandTracking:
    def __init__(self) -> None:
        self.labels = ['up', 'down', 'palm']

    def fetch_hand_landmarks(self, model=None):
        try:
            capture = cv2.VideoCapture(0)
            hands = mp.solutions.hands.Hands()
            img_counter = 105
            while True:
                cropped_img = []
                success, img = capture.read()
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                if results.multi_hand_landmarks:
                    for landmarks in results.multi_hand_landmarks:
                        minx = 9999
                        maxy = -9999
                        maxx = -9999
                        miny = 9999
                        for id, lm in enumerate(landmarks.landmark):
                            h, w, c = img.shape
                            cx, cy = int(lm.x *w), int(lm.y*h)

                            # need to find max and min for x and y to draw rectangle around landmarks
                            if cx > maxx:
                                maxx = cx
                            if cx < minx:
                                minx = cx
                            if cy > maxy:
                                maxy = cy
                            if cy < miny:
                                miny = cy

                        # pad the rectangle 20 pixels from landmarks
                        x1 = minx - 20
                        y1 = maxy + 20
                        x2 = maxx + 20
                        y2 = miny - 20

                        # pad rectangle to make 300x300 square
                        if (x2 - x1) > (y1 - y2):
                            square = (x2 - x1) + 5
                        if (y1 - y2) > (x2 - x1):
                            square = (y1 - y2) + 5
                        padx = square - (x2 - x1)
                        pady = square - (y1 - y2)

                        x2 = x2 + padx
                        y1 = y1 + pady

                        cv2.rectangle(img,(x1, y1),(x2, y2),(0,255,0),3)
                        mp.solutions.drawing_utils.draw_landmarks(img, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                        
                        cropped_img = img[y2:y1,x1:x2]
                        if not model == None: 
                            try:
                                resized = cv2.resize(cropped_img, dsize=(227, 227), interpolation=cv2.INTER_NEAREST)
                                gesture_pred = model.predict(np.expand_dims(resized, 0))
                                print(self.labels[np.argmax(np.round(gesture_pred))])
                            except BaseException as e:
                                cropped_img = []

                        # plt.imshow(cropped_img)
                        # plt.show()

                cv2.imshow("Image", img)
                cv2.waitKey(1)
                k = cv2.waitKey(1)
                
                if k%256 == 32 and not cropped_img == []: # spacebar for screenshots
                    img_name = "down{}.png".format(img_counter)
                    cv2.imwrite(img_name, cropped_img)
                    print("{} written!".format(img_name))
                    img_counter += 1

        except KeyboardInterrupt:
            capture.release()
            cv2.destroyAllWindows()

        except BaseException as e:
            print('error:')
            print(e)

if __name__ == "__main__":
    hand_tracking = HandTracking()
    hand_tracking.fetch_hand_landmarks()