#Import opencv_python
#download haarcascade_frontalface_default.xml to same proj folder
#download haarcascade_smile.xml to same proj folder

import cv2 as cv

cascade_face = cv.CascadeClassifier("haarcascade_frontalface_default.xml")
cascade_smile = cv.CascadeClassifier("haarcascade_smile.xml")

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    g = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    f = cascade_face.detectMultiScale(
        g,
        scaleFactor = 1.3,
        minNeighbors = 5,
        minSize = (30,50)
    )

    for (x,y,w,h) in f: #x,y add as horizontal axis and vertical axis, so x and width go together and y and heright go together
        cv.rectangle(img, (x,y), (x+w,y+h), (255,0,0))
        gray_r = g[y:y+h, x:x+w]

        s = cascade_smile.detectMultiScale(
            gray_r,
            scaleFactor = 1.5,
            minNeighbors = 15,
            minSize = (25,25)
        )

        for i in s:
            if len(s) > 1:
                cv.putText(img, "SMILING", (x, y-30),
                cv.FONT_HERSHEY_SIMPLEX, 2, (0,255,0),
                3, cv.LINE_AA)

    cv.imshow("video", img)

    k = cv.waitKey(30) & 0xff

    if k ==27:
        break

cap.release()

cv.destroyAllWindows()


    
    






