import cv2, time
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    _, frame = cap.read()
    # Play the video in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame',gray)
    cv2.imwrite('gray.jpg',gray)
    if cv2.waitKey(1) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()