import cv2
from detector import Detector

def show():
    cap = cv2.VideoCapture(0)
    detector = Detector()

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        detections = detector.detect(frame)

        for d in detections:
            p = (int(d.x), int(d.y))
            top_left = (int(p[0] - d.w/2), int(p[1] - d.h/2))
            bottom_right = (int(p[0] + d.w/2), int(p[1] + d.h/2))
            frame = cv2.circle(frame, p, 3, (255, 0, 0), 3)
            frame = cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
            frame = cv2.putText(frame, d.label, p, cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 0, 0), 2)

        # Display the resulting frame
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    cv2.VideoCapture(0)

if  __name__ == "__main__":
    show()
