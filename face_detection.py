import cv2 as cv
#We use xml files for any detection projects
#Import haarcascade_frontalface_default.xml

cas_face = cv.CascadeClassifier('haarcascade_frontalface_default.xml') #enter the xml file

cap = cv.VideoCapture(0) #camera access

while True: #Infinite loop so that the camera doesn't shut after capturing face
    ret, img = cap.read()
    print(ret) #prints t or f for detecting face

    #Converting frm rgb(bgr) to grayscale
    gr = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #Detect face
    face = cas_face.detectMultiScale(gr, 1.3, 4)

    #For rectangle around face
    for (x,y,w,h) in face:
        cv.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 4) #since x&w and y&h lie in same axis (0,0,255) represents bgr values

    cv.imshow("img", img)
    k = cv.waitKey(30) & 0xff   #waiting peiod of 30sec

    if k == 27:
        break

cap.release()
cv.destroyAllWindows() #to destroy the window with the capturing