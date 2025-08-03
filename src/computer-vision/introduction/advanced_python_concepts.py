## Iterators -> efficient looping and memory management 
## every time we do next(iterator) gives the next value until the end of the list 

my_list = list(range(1, 7))
iterator = iter(my_list)


## Generator 
## same thing with iterators doesnt store the values in the memory which makes it memory efficient 
## useful for the things like reading large files 

def square(n: int):
    for i in range(n):
        yield i ** 2


## Decorators => add functionality and methods without modifying the function itself 
## function copy
"""
def welcome():
    return "Welcome stranger"

wel = welcome  -
wel()           |
del welcome     | => Both will print the message even the function is deleted from the memory
wel()           |
               _
"""
## closures 
"""
def main_welcome(msg: str):
    def sub_welcome_method():
        print("This is a welcome message")
        print(msg)
        print("Another message")
    return sub_welcome_method()

main_welcome("eeeeee")

-------------------------------

def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator 

@repeat(3)
def say_hello():
    print("hello")
"""

import logging
import os

LOGGING_BASE_DIR = "/Users/user/Desktop/Projects/opencv-learning/src/logs"
LOGGING_FOLDER_DIR = "computer_vision"
LOGGING_FILENAME = "COMPUTER_VISION_ADVANCED_PYTHON_CONCEPTS_LOG.log"

def setup_log_config(base_dir, folder_dir, filename):
    # Ensure the log directory exists
    log_dir = os.path.join(base_dir, folder_dir)
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, filename)

    # Clear any existing handlers to avoid duplicates
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path, mode='a'),  # Changed to append mode
            logging.StreamHandler()
        ],
        force=True  # Force reconfiguration
    )

# Configure logging once
setup_log_config(LOGGING_BASE_DIR, LOGGING_FOLDER_DIR, LOGGING_FILENAME)
logger = logging.getLogger(__name__)
logger.debug("Hello logging - this should appear in both console and file")
logger.info("This is an info message")
logger.warning("This is a warning message")
logger.error("This is an error message")