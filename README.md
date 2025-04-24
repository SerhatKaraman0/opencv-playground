# OpenCV Learning Project

This repository contains code and resources for learning and experimenting with OpenCV, a popular computer vision library. The project includes functions for video and image processing, as well as sample data for testing.

## Project Structure

```
data/
    imgs/
        grayscaled_woman.jpg
        woman.jpg
        MERGED_IMAGES/
            horizontal/
                horizontally_merged_image.jpg
            vertical/
                vertically_merged_image.jpg
        RECTANGLED_IMAGES/
            rectangled_grayscaled_woman.jpg
        ROTATED_IMAGES/
            opencv-301/
                rotated_grayscaled_woman.jpg
            X/
                rotated_X_grayscaled_woman.jpg
            XY/
                rotated_XY_grayscaled_woman.jpg
            Y/
                rotated_Y_grayscaled_woman.jpg
        SHIFTED_IMAGES/
            shifted_grayscaled_woman.jpg
    videos/
        video_example.mov
src/
    image_processing_funcs.py
    video_processing_funcs.py
```

## Features

- **Video Processing**: Functions for handling and processing video files.
- **Image Processing**: Functions for manipulating and analyzing images.
- **Sample Data**: Includes sample videos and images for testing.

## Prerequisites

- Python 3.8 or higher
- OpenCV library
- Virtual environment (optional but recommended)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/opencv-learning.git
   cd opencv-learning
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Mac/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Add your video or image files to the `data/` directory.
2. Use the scripts in the `src/` directory to process the files. For example:
   ```bash
   python src/video_processing_funcs.py
   python src/image_processing_funcs.py
   ```


## Acknowledgments

- [Turkcell Geleceği Yazanlar Opencv Course](https://gelecegiyazanlar.turkcell.com.tr/egitimler/opencv/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Tutorials and resources from the OpenCV community.
