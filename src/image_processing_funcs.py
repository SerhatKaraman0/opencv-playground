import cv2 as cv
import os
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR    = os.path.abspath(os.path.join(BASE_DIR, 'data'))
IMGS_DIR    = os.path.abspath(os.path.join(DATA_DIR, 'imgs'))
VIDEOS_DIR  = os.path.abspath(os.path.join(DATA_DIR, 'videos')) 


def show_image_info(image_path: str) -> str:
    img_to_binary = cv.imread(image_path)

    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(img_to_binary)
    print(f'min_val: {min_val}, max_val: {max_val}, min_loc: {min_loc}, max_loc: {max_loc}')
    
    image_shapes = img_to_binary.shape
    print(f'image_shape: {image_shapes}')

    means, std_dev = cv.meanStdDev(img_to_binary)
    print(f'mean: {means}, std_dev: {std_dev}')


def image_normalization(image_path: str, option: str) -> np.array:
    NORMALIZE_OPTIONS = ['min_max', 'inf', 'l1', 'l2']
    img_to_binary = cv.imread(image_path, cv.COLOR_BGR2GRAY)
    float_img_to_binary = np.float32(img_to_binary)

    # NORM_MINMAX
    gray_img = np.zeros(float_img_to_binary, dtype=np.float32)

    if not option in NORMALIZE_OPTIONS:
        raise ValueError('Invalid option for normalize option.')

    if option == 'min_max':
        cv.normalize(gray_img, dst=gray_img, alpha=0, beta=1.0, norm_type=cv.NORM_MINMAX)
        return gray_img 
    
    if option == 'inf':
        cv.normalize(gray_img, dst=gray_img, alpha=0, beta=1.0, norm_type=cv.NORM_INF)
        return gray_img 
    
    if option == 'l1':
        cv.normalize(gray_img, dst=gray_img, alpha=0, beta=1.0, norm_type=cv.NORM_L1)
        return gray_img 
    
    if option == 'l2':
        cv.normalize(gray_img, dst=gray_img, alpha=0, beta=1.0, norm_type=cv.NORM_L2)
        return gray_img 
    

def show_image(
               image_path      : str, 
               window_title    : str = 'Image Showing Window', 
               wait_time       : int = 5_000
               ) -> None:
    
    img_to_binary = cv.imread(image_path)
    cv.namedWindow(window_title, cv.WINDOW_AUTOSIZE)
    cv.imshow(window_title, img_to_binary)
    cv.waitKey(wait_time)


def grayscale_img(image_path: str, wait_time: int = 1) -> np.ndarray:
    img_to_binary = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    cv.imshow('Grayscaled Image', img_to_binary)
    cv.waitKey(wait_time)

    return img_to_binary


def save_image(image_binary: np.ndarray, path_to_save: str, file_name: str) -> str:
    new_file_dest = os.path.join(path_to_save, file_name)
    cv.imwrite(new_file_dest, image_binary)

    print(f'Image saved to the destination: {new_file_dest}')
    return new_file_dest


def draw_rectangle(
                   image_path         :         str, 
                   top_left           :         tuple[int, int], 
                   bottom_right       :         tuple[int, int], 
                   color              :         tuple[int, int, int], 
                   border_thickness   :         int,
                   wait_time          :         int  = 5_000,
                   is_save            :         bool = False
                   ) -> None:
    
    img_binary = cv.imread(image_path)

    if img_binary is None:
        print(f'Error: Unable to load image from {image_path}')
        return
    
    cv.rectangle(img_binary, top_left, bottom_right, color, border_thickness)
    cv.imshow('Rectangled Image', img_binary)
    cv.waitKey(wait_time)

    if is_save:
        file_name = os.path.basename(image_path)
        save_name = f"rectangled_{file_name}"

        save_dir  = os.path.join(IMGS_DIR, "RECTANGLED_IMAGES")
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, img_binary)
        print(f"Rectangled image saved at: {save_path}")



def merge_images(
                 first_img_path :    str, 
                 second_img_path:    str, 
                 merge_option   :    str,
                 wait_time      :    int,
                 is_save        :    bool = False
                ) -> None:
    
    OPTIONS = ["horizontal", "vertical"]

    first_img_to_binary = cv.imread(first_img_path)
    second_img_to_binary = cv.imread(second_img_path)

    if not merge_option in OPTIONS:
        raise ValueError("Invalid option for merge option.")
    
    if merge_option == "horizontal" or merge_option == "vertical":
        merged_img = None
        save_name = None

        if merge_option == "horizontal":
            merged_img = np.hstack((first_img_to_binary, second_img_to_binary))
            save_name  = "horizontally_merged_image.jpg"
        elif merge_option == "vertical":
            merged_img = np.vstack((first_img_to_binary, second_img_to_binary))
            save_name  = 'vertically_merged_image.jpg'

        if merged_img is not None:
            cv.imshow(f'{merge_option.capitalize()} Merged Image', merged_img)
            cv.waitKey(wait_time)

            if is_save:
                save_dir  = os.path.join(IMGS_DIR, "MERGED_IMAGES", merge_option)
                os.makedirs(save_dir, exist_ok=True)  
                save_path = os.path.join(save_dir, save_name)
                cv.imwrite(save_path, merged_img)
                print(f'{merge_option.capitalize()} merged image saved at: {save_path}')
    

def rotate_img(img_path     :   str, 
               rotate_dir   :   str, 
               wait_time    :   int  = 5_000, 
               is_save      :   bool = False
               ) -> None:
    
    OPTIONS = ['X', 'Y', 'XY']

    if not rotate_dir in OPTIONS:
       raise ValueError('Invalid option for rotate option.')

    img_to_binary = cv.imread(img_path)
    
    if rotate_dir == 'X':
        rotated_img = cv.flip(img_to_binary, 0)
        cv.imshow("x-flip", rotated_img)

    if rotate_dir == 'Y':
        rotated_img = cv.flip(img_to_binary, 1)
        cv.imshow("y-flip", rotated_img)

    if rotate_dir == 'XY':
        rotated_img = cv.flip(img_to_binary, -1)
        cv.imshow("xy-flip", rotated_img) 

    cv.waitKey(wait_time)

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"rotated_{rotate_dir}_{file_name}"

        save_dir  = os.path.join(IMGS_DIR, "ROTATED_IMAGES", rotate_dir)
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, rotated_img)
        print(f"Rotated image saved at: {save_path}")


def shift_img(
                 img_path: str, 
                 first_shift_arr    :   list[int, int, int], 
                 second_shift_arr   :   list[int, int, int], 
                 wait_time: int     =   5_000,
                 is_save: bool      =   False
                 ) -> None:
    
    img_to_binary = cv.imread(img_path)
    cols, rows = img_to_binary.shape[0], img_to_binary.shape[1]

    M = np.float32([first_shift_arr, second_shift_arr])

    shifted = cv.warpAffine(img_to_binary, M, (cols, rows))

    cv.imshow('Shifted Image', shifted)
    cv.waitKey(wait_time)

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"shifted_{file_name}"

        save_dir  = os.path.join(IMGS_DIR, "SHIFTED_IMAGES")
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, shifted)
        print(f"Shifted image saved at: {save_path}")


def rotate_img(
        img_path    : str,
        wait_time   : int   = 5_000,
        is_save     : bool  = False
):
    img_to_binary = cv.imread(img_path)
    cols, rows = img_to_binary.shape[0], img_to_binary.shape[1]
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 90, 1)
    
    rotated = cv.warpAffine(img_to_binary, M, (cols, rows))

    cv.imshow('Rotated Image', rotated)
    cv.waitKey(wait_time)

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"rotated_{file_name}"

        save_dir  = os.path.join(IMGS_DIR, "ROTATED_IMAGES", "opencv-301")
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, rotated)
        print(f"Shifted image saved at: {save_path}")


def histogram_and_img_hist(img_path: str):
    img_to_binary = cv.imread(img_path)
    if img_to_binary is None:
        raise ValueError(f"Error: Unable to load image from {img_path}")
    
    # Histogram calculation
    h, w, _ = img_to_binary.shape
    hist = np.zeros([256], dtype=np.int32)
    for row in range(h):
        for col in range(w):
            pv = img_to_binary[row, col]
            hist[pv] += 1
    
    # Plot histogram
    y_pos = np.arange(0, 256, 1, dtype=np.int32)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(y_pos, hist, align='center', color='r', alpha=0.5)
    plt.xticks(y_pos, y_pos)
    plt.ylabel('Frequency')
    plt.title('Histogram')

    # Image histogram using OpenCV
    plt.subplot(1, 2, 2)
    color = ('blue', 'green', 'red')
    for i, col in enumerate(color):
        hist = cv.calcHist([img_to_binary], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.xlim([0, 256])
    plt.title('Image Histogram')
    
    plt.show()


def bilateral_filter(img_path: str):
    img_to_binary = cv.imread(img_path)

    h, w = img_to_binary.shape[:2]

    dst = cv.bilateralFilter(img_to_binary)

######### NOT IMPLEMENTED FEATURES BUT COVERED IN THE COURSE #########
"""
Chapter 3:
    Gauss Smoothing Filter:
        - Used to reduce noise and detail in an image.
        - Applies a Gaussian kernel to smooth the image.
        - Commonly used as a preprocessing step in image processing tasks.

    Sobel Filter:
        - Used for edge detection by calculating the gradient of image intensity.
        - Highlights regions of high spatial frequency, such as edges.
        - Operates in horizontal, vertical, or both directions.

    Canny Edge Detection:
        - A multi-step algorithm to detect edges in an image.
        - Involves noise reduction, gradient calculation, non-maximum suppression, and edge tracking.
        - Produces a binary image with detected edges.

    Otsu Method:
        - Used for automatic image thresholding.
        - Determines an optimal threshold value to separate foreground and background.
        - Works by maximizing inter-class variance.

    Contours:
        - Used to detect and analyze shapes and boundaries in an image.
        - Represents a curve joining all continuous points along a boundary with the same intensity.
        - Useful for object detection and shape analysis.

    Hoffman Line Detection:
        - Used to detect straight lines in an image.
        - Based on the Hough Transform algorithm.
        - Effective for identifying lines in noisy or complex images.

    Hoffman Circle Detection:
        - Used to detect circular shapes in an image.
        - Based on the Hough Transform for circles.
        - Commonly used in applications like detecting coins or circular objects.
"""

################### CHAPTER 4 #######################
# Adding noise to an image 

def add_salt_and_pepper_noise(img_path: str, wait_time: int = 10_000, is_save: bool = False):
    img_to_binary = cv.imread(img_path)
    h, w = img_to_binary.shape[:2]
    nums = 10_000

    rows = np.random.randint(0, h, nums, dtype=np.int32)
    cols = np.random.randint(0, w, nums, dtype=np.int32)
    
    for i in range(nums):
        if i % 2 == 1:
            img_to_binary[rows[i], cols[i]] = (255, 255, 255)
        img_to_binary[rows[i], cols[i]] = (0, 0, 0)
    
    cv.imshow("new window", img_to_binary)
    cv.waitKey(wait_time) 

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"salt_and_pepper_{file_name}"

        save_dir  = os.path.join(IMGS_DIR, "NOISED_IMAGES")
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, img_to_binary)
        print(f"Salt and peppered image saved at: {save_path}")


# image sharpening 
def sharpen_image(img_path: str, wait_time: int = 10_000, is_save: bool =  False):
    img_to_binary = cv.imread(img_path)

    sharpen_op = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpen_img = cv.filter2D(img_to_binary, cv.CV_32F, sharpen_op)
    sharpen_img = cv.convertScaleAbs(sharpen_img)

    cv.imshow("Sharpened Image", sharpen_img)
    cv.waitKey(wait_time)

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"sharpened_{file_name}"

        save_dir  = os.path.join(IMGS_DIR, "SHARPENED_IMAGES")
        os.makedirs(save_dir, exist_ok=True)  
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, img_to_binary)
        print(f"Sharpened image saved at: {save_path}")

# Harris corner detection
def harris_corner_detection(img_path: str, wait_time: int = 10_000, is_save: bool = False) -> None:
    img = cv.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {img_path}")

    block_size = 2
    aperture_size = 3
    k = 0.04

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv.cornerHarris(gray, block_size, aperture_size, k)

    # Threshold for an optimal value, it may vary depending on the image.
    threshold_value = 0.01 * dst.max()
    ys, xs = np.where(dst > threshold_value)

    for (x, y) in zip(xs, ys):
        cv.circle(img, (x, y), 2, (0, 255, 0), 2)

    cv.imshow("Harris Corner Detection", img)
    cv.waitKey(wait_time)
    cv.destroyAllWindows()

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"harris_{file_name}"

        save_dir = os.path.join("images", "HARRIS_IMAGES")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, img)
        print(f"Harris-corner detected image saved at: {save_path}")
   

def shi_thomas_corner_detection(img_path: str, wait_time: int= 10_000, is_save: bool = False):
    img_to_binary = cv.imread(img_path)

    gray = cv.cvtColor(img_to_binary, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners=35,
        qualityLevel=0.05,
        minDistance=10
    )

    for pt in corners:
        print(pt)
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv.circle(img_to_binary, (x, y), 5, (int(b), int(g), int(r)), 2)

    cv.imshow("Shi Thomas Corner Detection", img_to_binary)
    cv.waitKey(wait_time)
    cv.destroyAllWindows()

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"shi_thomas_{file_name}"

        save_dir = os.path.join("images", "SHI_THOMAS_IMAGES")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, img_to_binary)
        print(f"SHİ-Thomas-corner detected image saved at: {save_path}") 


def subpixel_corner_detection(img_path: str, wait_time: int= 10_000, is_save: bool = False):
    img_to_binary = cv.imread(img_path)

    gray = cv.cvtColor(img_to_binary, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(
        gray,
        maxCorners=35,
        qualityLevel=0.05,
        minDistance=10
    )

    for pt in corners:
        print(pt)
        b = np.random.randint(0, 256)
        g = np.random.randint(0, 256)
        r = np.random.randint(0, 256)
        x = np.int32(pt[0][0])
        y = np.int32(pt[0][1])
        cv.circle(img_to_binary, (x, y), 5, (int(b), int(g), int(r)), 2)


    winSize = (3, 3)
    zeroZone = (-1, -1)

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 40, 0.001)
    corners = cv.cornerSubPix(gray, corners, winSize, zeroZone, criteria)

    for i in range(corners.shape[0]):
        print(" -- Refined Corner [", i, "] (", corners[i, 0, 0], ",", corners[i, 0, 1], ")")

    cv.imshow("Subpixel Corner Detection", img_to_binary)
    cv.waitKey(wait_time)
    cv.destroyAllWindows()

    if is_save:
        file_name = os.path.basename(img_path)
        save_name = f"subpixel_{file_name}"

        save_dir = os.path.join("images", "SUBPIXEL_IMAGES")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, save_name)
        cv.imwrite(save_path, img_to_binary)
        print(f"SUb pixel detected image saved at: {save_path}") 
    


def main():
    first_img_path = os.path.join(IMGS_DIR, 'grayscaled_woman.jpg')
    second_img_path = os.path.join(IMGS_DIR, 'woman.jpg')
    
    # show_image(img_path, wait_time=10_000)
    # grayscaled_img = grayscale_img(img_path, wait_time=1)
    # save_image(grayscaled_img, IMGS_DIR, "grayscaled_woman.jpg")
    
    # show_image_info(img_path)

    # draw_rectangle(first_img_path, (80, 100), (180, 230), (0, 0, 255), 8, 2_000, True)

    # merge_images(first_img_path, second_img_path, "horizontal", 2_000, True)

    # rotate_img(first_img_path, "X", 1_000, True)
    # rotate_img(first_img_path, "Y", 1_000, True)
    # rotate_img(first_img_path, "XY", 1_000, True)
    
    # first_shift_arr  = [1, 0, 70]
    # second_shift_arr = [0, 1, 30]
    
    # histogram_and_img_hist(second_img_path)

    subpixel_corner_detection(first_img_path, is_save=True)

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()