import cv2
import YoloCustmObjectDetection as YoloJD


# Webcam
cap = cv2.VideoCapture(0)  # , cv2.CAP_DSHOW or CAP_MSMF

# creat object for toy car detection
DetectObj = YoloJD.ToyCarDetection(capture_index=1, model_name='yolov5s.pt')

while True:
     # OpenCV
    success, img = cap.read()

    # yolo toy car detection on live camera stream
    results = DetectObj.score_frame(frame=img)

     # plot the bbox on car
    frame = DetectObj.plot_boxes(results, frame=img)

    cv2.imshow('My Image', img)

    k = cv2.waitKey(1) & 0xFF
    print(k)
    if k == 27:  # close on ESC key
        cv2.destroyAllWindows()
        cap.release()
        break



