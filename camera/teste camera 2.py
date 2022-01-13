import cv2

webcam = cv2.VideoCapture(0)

if webcam.isOpened():
    validacao, frame = webcam.read()
    cv2.imwrite("FotoLira.png", frame)

webcam.release()
cv2.destroyAllWindows()