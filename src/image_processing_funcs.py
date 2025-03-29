import cv2 as cv
import os
import numpy as np

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
    
    rotate_img(first_img_path, is_save=True)
    
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()