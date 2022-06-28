import cv2
import numpy as np

from src.video_read import Video_Read
from src.utils import image_paint
import psutil
import time

st = time.time()

def main():
    video_read = Video_Read(0)

    model_path = "opencv_dnn_model.caffemodel"
    model_config = "opencv_dnn_model.prototxt"
    model = cv2.dnn.readNetFromCaffe(model_config, model_path)
    threshold_value = 0.7

    foto_size = (300, 300)
    for frame in video_read:
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, foto_size), 1.0, foto_size, [104.0, 117.0, 123.0])
        model.setInput(blob)
        faces = model.forward().squeeze()[:, 2:] # array.shape = [1, ?, 7] -> squeeze -> [15, 7]
        # [(olasilik, xmin, ymin, xmax, ymax), (olasilik, xmin, ymin, xmax, ymax), ...]
        faces = faces[faces[:, 0] > threshold_value]

        possibilities, faces = faces[:, 0], faces[:, 1:]
        # faces =  [(xmin, ymin, xmax, ymax), ....] -> 0-1
        frame_y, frame_x = frame.shape[:2]
        faces *= np.array([frame_x, frame_y, frame_x, frame_y])

        for box, olasilik in zip(faces, possibilities):
            image_paint(frame, box, f"{olasilik:.2f}")

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break


if __name__ == "__main__":
    main()
pid = os.getpid()
python_process = psutil.Process(pid)
memoryUse = python_process.memory_info()[0]/2.**30  # memory use in GB...I think
print('memory use:', memoryUse)


et = time.time()
print('Execution time:', et-st, 'seconds')