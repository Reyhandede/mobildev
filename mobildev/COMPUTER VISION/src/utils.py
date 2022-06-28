import cv2


def image_paint(frame, box, text=None, face_points=None):
    xmin, ymin, xmax, ymax = list(map(int, box))
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

    if text is not None:
        cv2.putText(frame, text, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if face_points is not None:
        for n in face_points:
            nx, ny = list(map(int, n))
            cv2.circle(frame, (nx, ny), 2, (0, 255, 0), -1)
