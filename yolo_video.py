"""
Run a YOLO_v3 style detection model on test video.
"""

from yolo import YOLO
from yolo import detect_video2



if __name__ == '__main__':
#    ./videos/person.jpg
#    video_path='path2your-video'
    video_path='./videos/test.mp4'
    detect_video2(YOLO(), video_path)
