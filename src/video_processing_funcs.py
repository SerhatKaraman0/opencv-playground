import cv2 as cv
import numpy as np
import os

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR    = os.path.abspath(os.path.join(BASE_DIR, 'data'))
IMGS_DIR    = os.path.abspath(os.path.join(DATA_DIR, 'imgs'))
VIDEOS_DIR  = os.path.abspath(os.path.join(DATA_DIR, 'videos')) 


def show_video_info(video_path: str):
    capture = cv.VideoCapture(video_path)
    
    height  =   capture.get(cv.CAP_PROP_FRAME_HEIGHT)
    width   =   capture.get(cv.CAP_PROP_FRAME_WIDTH)
    count   =   capture.get(cv.CAP_PROP_FRAME_COUNT)
    fps     =   capture.get(cv.CAP_PROP_FPS)

    print(f'height: {height}, width: {width}, count: {count}, fps: {fps}')
    capture.release()


def process_video(image, options : int = 1):
    dst = None

    if options == 0:
        dst = cv.bitwise_not(image)
    if options == 1:
        dst = cv.GaussianBlur(image, (0, 0), 15)
    if options == 2:
        dst = cv.Canny(image, 100, 200)
    
    return dst

def live_video_processing(cam_option: int = 1, index: int = 2):
    capture = cv.VideoCapture(cam_option)

    while True:
        ret, frame = capture.read()

        if ret:
            cv.imshow('video-input', frame)
            c = cv.waitKey(50)

            if c == 27:  
                break
            elif 48 <= c <= 50:  
                index = c - 48

            result = process_video(frame, index)
            cv.imshow('result', result)
        else:
            break

    capture.release()
    cv.destroyAllWindows()
            

def main():
    video_path = os.path.join(VIDEOS_DIR, 'video_example.mov')
    
    live_video_processing(1, 0)

if __name__ == "__main__":
    main()
    