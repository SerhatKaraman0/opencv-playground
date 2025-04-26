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

def background_foreground_extraction(video_path: str, threshold: int):
    video = cv.VideoCapture(video_path)

    fbgb = cv.createBackgroundSubtractorMOG2(history=250, varThreshold=threshold)

    while True:
        ret, frame = video.read()
        fgmask = fbgb.apply(frame)
        background = fbgb.getBackgroundImage()
        cv.imshow("input", frame)
        cv.imshow("mask", fgmask)
        cv.imshow("background", background)
        k = cv.waitKey()&0xff

        if k == 27:
            break


def klt_applications(video_path: str):
    video = cv.VideoCapture(video_path)
    ret, frame = video.read()

    if not ret:
        print("Error: Unable to read video file.")
        return

    prvs = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    hsv = np.zeros_like(frame)
    hsv[..., 1] = 255

    def dense_opt_flow(hsv, prvs):
        while True:
            ret, frame1 = video.read()
            if not ret:
                print("End of video or error reading frame.")
                break

            nextt = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            flow = cv.calcOpticalFlowFarneback(prvs, nextt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
            cv.imshow("Optical Flow", bgr)
            cv.imshow("Original Frame", frame1)
            k = cv.waitKey(30) & 0xff

            if k == 27:  # ESC key to exit
                break

            prvs = nextt

    dense_opt_flow(hsv, prvs)
    video.release()
    cv.destroyAllWindows()
    

def main():
    video_path = os.path.join(VIDEOS_DIR, 'video_example.mov')
    
    # live_video_processing(1, 0)
    # background_foreground_extraction(video_path=video_path, threshold=100)
    klt_applications(video_path=video_path)

if __name__ == "__main__":
    main()
    