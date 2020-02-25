"""

This code was created as part of the ThyssenKrupp hack4tk hackathon on the 03./04.07.2019. Commercial use is not permitted unless agreed upon between the team Investeel and the using party. The term "Investeel" describes a team comprising Stefan Bassler, Jonas Kaendler, Victoria Faber, Andrey Bogomolov and Tim Treis.

Author: Tim Treis
Date: 04.07.2019

"""


# import the necessary packages
import skimage
from skimage.measure import compare_ssim
import imutils
import numpy as np
import datetime
import matplotlib.pyplot as plt
import tkinter
#from imageai.Detection import VideoObjectDetection
import numpy as np
from PIL import Image
from keras import models

import cv2
#from imageai.Prediction import ImagePrediction
import os

top = tkinter.Tk()
top.title('Car door simulator!')
top.geometry('250x70')

def openDoor():

  vs = cv2.VideoCapture("white_to_black.avi")
  #vs = cv2.VideoCapture(0)

  fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

  i = 0
  firstFrame = None

  vals = []

  while vs.isOpened():

    frame = vs.read()
    frame = frame[1]

    if frame is None:

      break

    text = "No issues"

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if firstFrame is None:

      firstFrame = gray

    frameDelta = cv2.absdiff(firstFrame, gray)

    ret, thresh1 = cv2.threshold(frameDelta, 50, 255, cv2.THRESH_BINARY)

    thresh = cv2.dilate(thresh1, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # Adjust for spacing of bounding rectangle
    a = 1
    b = 0

    for c in cnts:

      for d in cnts:

        (x1, y1, w1, h1) = cv2.boundingRect(c)
        (x2, y2, w2, h2) = cv2.boundingRect(d)
        (x1, y1, w1, h1) = (
        x1 - (w1 * b), y1 - (h1 * b), w1 * (a + b), h1 * (a + b))
        (x2, y2, w2, h2) = (
        x2 - (w2 * b), y2 - (h2 * b), w2 * (a + b), h2 * (a + b))

        for point in [(x1, y1), (x1 + w1, y1), (x1, y1 + h1),
                      (x1 + w1, y1 + h1)]:

          if (point[0] >= x2) and (point[0] <= x2 + w2) and (
              point[1] >= y2) and (point[1] <= y2 + h2):
            x_min = int(min(x1, x2))
            x_max = int(max(x1, x2))
            y_min = int(min(y1, y2))
            y_max = int(max(y1, y2))

            if x_min < 0:
              x_min = 0

            if y_min < 0:
              y_min = 0

            if y_max > frame.shape[0]:
              y_max = 449

            if y_max < 0:
              y_max = 0

            if x_max < 0:
              x_max = 0

            if x_max > frame.shape[1]:
              x_max = 799

            cv2.rectangle(frame,
                          (min(x_min, x_max),
                           min(y_min, y_max)),
                          (max(x_min, x_max),
                           max(y_min, y_max)), (0, 255, 0), 2)

            #cv2.imwrite("cropped"+str(x_max)+"_"+str(x_min)+".jpg", frame[min(x_min, x_max):min(y_min, y_max),
						#											max(x_min, x_max):max(y_min, y_max)].copy())

            print(min(x_min, x_max),max(x_min, x_max),min(y_min, y_max),max(y_min, y_max))

      (x, y, w, h) = cv2.boundingRect(c)
      # (x, y, w, h) = (int(x-(w*a)), int(y-(h*b)), int(w*a), int(h*a))
      cv2.rectangle(frame, (int(x - (w * b)), int(y - (h * b))),
                    (x + int(w * (a + b)), y + int(h * (a + b))), (0, 255, 0),
                    2)
      text = "Issue"
      cv2.putText(frame, "Car Status: {}".format(text), (10, 20),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      cv2.putText(frame, "Issue", (x, int(y * 0.98)),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      cv2.putText(frame,
                  datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                  (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                  (0, 0, 255), 1)

    numpy_horizontal = np.concatenate((frame, cv2.cvtColor(frameDelta, cv2.COLOR_GRAY2RGB), cv2.cvtColor(thresh1, cv2.COLOR_GRAY2RGB)), axis=1)
    vals.append(np.mean(frame))
    #plt.imshow(frame)
    #plt.show()
#    cv2.imshow("thre", numpy_horizontal)

    #cv2.imshow("Frame Delta", frameDelta)
#    key = cv2.waitKey(1) & 0xFF

    i += 1

  plt.plot(vals)
  plt.show()


def closeDoor():

  model = models.load_model('yolo.h5')
  video = cv2.VideoCapture(0)

  while True:
        _, frame = video.read()

        #Convert the captured frame into RGB
        im = Image.fromarray(frame, 'RGB')

        #Resizing into 128x128 because we trained the model with this image size.
        im = im.resize((128,128))
        img_array = np.array(im)

        #Our keras model used a 4D tensor, (images x height x width x channel)
        #So changing dimension 128x128x3 into 1x128x128x3
        img_array = np.expand_dims(img_array, axis=0)

        #Calling the predict method on model to predict 'me' on the image
        prediction = int(model.predict(img_array)[0][0])

        #if prediction is 0, which means I am missing on the image, then show the frame in gray color.
        if prediction == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow("Capturing", frame)
        key=cv2.waitKey(1)
        if key == ord('q'):
                break
  video.release()
  cv2.destroyAllWindows()


B_open = tkinter.Button(top, text="Detect Issues", command=openDoor)
B_close = tkinter.Button(top, text="Classify", command=closeDoor)

B_open.pack()
B_close.pack()
top.mainloop()
