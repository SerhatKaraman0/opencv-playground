# OpenCV Learning Project

This repository contains code and resources for learning and experimenting with OpenCV, a popular computer vision library. The project includes comprehensive functions for video and image processing, deep learning tutorials with CNNs, and sample data for testing and learning.

## Project Structure

```
data/
    imgs/
        woman.jpg                          # Original sample image
        grayscaled_woman.jpg              # Processed grayscale version
        HARRISSED_IMAGES/                 # Harris corner detection results
        MERGED_IMAGES/                    # Image merging examples
            horizontal/
            vertical/
        NOISED_IMAGES/                    # Salt and pepper noise examples
        RECTANGLED_IMAGES/                # Rectangle drawing examples
        ROTATED_IMAGES/                   # Image rotation examples
            opencv-301/, X/, XY/, Y/
        SHARPENED_IMAGES/                 # Image sharpening results
        SHIFTED_IMAGES/                   # Image shifting examples
    videos/                               # Sample video files
src/
    image_processing_funcs.py             # Core image processing functions
    video_processing_funcs.py             # Video processing and analysis
    computer-vision/
        introduction/
            advanced_python_concepts.py   # Python concepts for CV
        deep-learning/
            introduction.ipynb            # Deep learning introduction
            CNN/
                CNN.ipynb                 # Convolutional Neural Networks
                main.ipynb               # CNN main examples
                30-38+CNN.pdf            # CNN reference material
        pytorch-playground/               # PyTorch experiments
    images/                              # Additional image resources
    logs/                                # Processing logs
requirements.txt                         # Python dependencies
```

## Features

### Image Processing
- **Basic Operations**: Grayscale conversion, image normalization, display and saving
- **Geometric Transformations**: Image rotation, shifting, merging (horizontal/vertical)
- **Drawing Operations**: Rectangle drawing and shape overlay
- **Filtering & Enhancement**: Bilateral filtering, image sharpening
- **Noise Processing**: Salt and pepper noise addition and removal
- **Corner Detection**: Harris corner detection, Shi-Tomasi corner detection, sub-pixel corner refinement
- **Statistical Analysis**: Image histograms, min/max value analysis, mean and standard deviation

### Video Processing
- **Basic Video Operations**: Video information display, frame processing
- **Real-time Processing**: Live video feed processing with multiple filter options
- **Motion Analysis**: Background/foreground extraction, KLT (Kanade-Lucas-Tomasi) tracking
- **Optical Flow**: Dense optical flow computation and visualization

### Deep Learning & Computer Vision
- **CNN Tutorials**: Comprehensive Convolutional Neural Network examples and theory
- **PyTorch Playground**: Experimental deep learning implementations
- **Educational Materials**: Advanced Python concepts for computer vision
- **Jupyter Notebooks**: Interactive learning environment with step-by-step tutorials

### Sample Data & Examples
- **Sample Images**: Test images with various processed versions (rotated, filtered, etc.)
- **Video Examples**: Sample videos for testing processing algorithms
- **Generated Results**: Pre-processed example outputs for reference

## Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)
- Jupyter Notebook or JupyterLab (for interactive tutorials)

### Dependencies
The project uses the following main libraries:
- **OpenCV** (4.12.0.88): Computer vision and image processing
- **NumPy** (2.2.6): Numerical computing and array operations
- **Matplotlib** (3.10.5): Plotting and visualization
- **Pillow** (11.3.0): Additional image processing capabilities

See `requirements.txt` for the complete list of dependencies with specific versions.

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
   # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install Jupyter (if not included in requirements):
   ```bash
   pip install jupyter
   ```

## Usage

### Basic Image Processing

1. **Add your images to the `data/imgs/` directory**

2. **Run individual image processing functions:**
   ```python
   from src.image_processing_funcs import *
   
   # Display image information
   show_image_info('data/imgs/woman.jpg')
   
   # Convert to grayscale
   gray_img = grayscale_img('data/imgs/woman.jpg')
   
   # Apply corner detection
   harris_corner_detection('data/imgs/woman.jpg', is_save=True)
   
   # Add noise and sharpen
   add_salt_and_pepper_noise('data/imgs/woman.jpg', is_save=True)
   sharpen_image('data/imgs/woman.jpg', is_save=True)
   ```

3. **Run the main processing script:**
   ```bash
   python src/image_processing_funcs.py
   ```

### Video Processing

1. **Add video files to the `data/videos/` directory**

2. **Process videos:**
   ```python
   from src.video_processing_funcs import *
   
   # Show video information
   show_video_info('data/videos/your_video.mp4')
   
   # Process live camera feed
   live_video_processing(cam_option=1)
   
   # Extract background/foreground
   background_foreground_extraction('data/videos/your_video.mp4', threshold=50)
   ```

3. **Run video processing script:**
   ```bash
   python src/video_processing_funcs.py
   ```

### Interactive Learning with Jupyter Notebooks

1. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

2. **Open the learning notebooks:**
   - `src/computer-vision/deep-learning/introduction.ipynb` - Deep learning fundamentals
   - `src/computer-vision/deep-learning/CNN/CNN.ipynb` - Convolutional Neural Networks
   - `src/computer-vision/deep-learning/CNN/main.ipynb` - CNN examples

3. **Explore advanced concepts:**
   - Check `src/computer-vision/introduction/advanced_python_concepts.py` for Python CV techniques
   - Review `src/computer-vision/deep-learning/CNN/30-38+CNN.pdf` for theoretical background

### Example Function Usage

```python
# Image merging
merge_images(
    img1_path='data/imgs/woman.jpg',
    img2_path='data/imgs/grayscaled_woman.jpg',
    direction='horizontal',
    is_save=True
)

# Image rotation with different axes
rotate_img('data/imgs/woman.jpg', axis='X', angle=45, is_save=True)

# Bilateral filtering for noise reduction
bilateral_filter('data/imgs/woman.jpg')
```

## Deep Learning & CNN Tutorial Content

This project includes comprehensive educational materials for learning deep learning and computer vision:

### 📚 Learning Path

1. **Start with Python Fundamentals:**
   - `src/computer-vision/introduction/advanced_python_concepts.py`
   - Essential Python concepts for computer vision work

2. **Deep Learning Introduction:**
   - `src/computer-vision/deep-learning/introduction.ipynb`
   - Fundamentals of deep learning for computer vision

3. **Convolutional Neural Networks:**
   - `src/computer-vision/deep-learning/CNN/CNN.ipynb` - Comprehensive CNN tutorial
   - `src/computer-vision/deep-learning/CNN/main.ipynb` - Practical CNN examples
   - `src/computer-vision/deep-learning/CNN/30-38+CNN.pdf` - Theoretical reference material

4. **Experimental Playground:**
   - `src/computer-vision/pytorch-playground/` - PyTorch experiments and advanced implementations

### 🎯 Learning Objectives

- Understand core computer vision concepts with OpenCV
- Learn image processing techniques and their applications
- Master CNN architecture and implementation
- Apply deep learning to computer vision problems
- Gain hands-on experience with real-world examples

### 💡 Getting Started with Learning

1. Begin with basic image processing functions in `src/image_processing_funcs.py`
2. Experiment with video processing in `src/video_processing_funcs.py`
3. Work through the Jupyter notebooks in order
4. Try the example functions with your own images
5. Explore the deep learning materials for advanced topics

## Acknowledgments

- [Turkcell Geleceği Yazanlar Opencv Course](https://gelecegiyazanlar.turkcell.com.tr/egitimler/opencv/)
- [OpenCV Documentation](https://docs.opencv.org/)
- Tutorials and resources from the OpenCV community.
