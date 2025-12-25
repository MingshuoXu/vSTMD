import os
import re
from typing import List, Optional, Union
import glob

import json
import numpy as np
from scipy.ndimage import maximum_filter
import torch.nn.functional as F
import cv2
import torch


def custom_serialize(obj, indent=2, current_level=0):
    if isinstance(obj, list):
        current_level += 1
        if any(isinstance(item, (list, dict)) for item in obj):
            # Contains nested structures, not the last level
            indent_str = ' ' * (indent * current_level)
            items = [f"{indent_str}{custom_serialize(item, indent, current_level)}" for item in obj]
            outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
            return '[\n' + ',\n'.join(items) + '\n' + outer_indent + ']'
        else:
            # Last level, all basic types
            return '[' + ', '.join(json.dumps(item, ensure_ascii=False) for item in obj) + ']'
    
    elif isinstance(obj, dict):
        current_level += 1
        indent_str = ' ' * (indent * current_level)
        items = []
        for key, value in obj.items():
            serialized_value = custom_serialize(value, indent, current_level)
            items.append(f'{indent_str}{json.dumps(key, ensure_ascii=False)}: {serialized_value}')
        
        outer_indent = ' ' * (indent * (current_level - 1)) if current_level > 1 else ''
        return '{\n' + ',\n'.join(items) + '\n' + outer_indent + '}'
    
    else:
        return json.dumps(obj, ensure_ascii=False)
    

def nms(matrix, window_size=8, device='cpu'):
    """
    Perform non-maximum suppression on the matrix using maximum_filter for efficiency.

    Args:
        matrix: Input matrix (numpy.ndarray)
        window_size: Neighborhood window size, must be odd
        device: Computing device ('cpu' or 'cuda')

    Returns:
        nms_matrix: Matrix after non-maximum suppression
    """

    if device == 'cpu':
        # Define neighborhood structure (a window of all ones)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
        local_max_cv2 = cv2.dilate(matrix, kernel)
        # Calculate local maximum at each position
        nms_matrix = matrix * (matrix == local_max_cv2)
    else:
        # Use max pooling to get local maximum
        local_max = F.max_pool2d(
            matrix, 
            kernel_size=window_size*2+1, 
            stride=1, 
            padding=window_size
        )
        
        nms_matrix = matrix * (matrix == local_max)

    return nms_matrix


def show_result_in_cv2(window_name, color_img, result, show_number=10, device='cpu'):
    # Visualize the result
    if device == 'cpu':
        if np.max(result['response']) == 0:
            return True
        response_matrix = nms(result['response'])
        direction_matrix = result['direction']
    else:
        if torch.max(result['response']) == 0:
            return True
        response_matrix = nms(result['response'], device='cuda').squeeze(0).squeeze(0).cpu().numpy()
        direction_matrix = result['direction'].squeeze(0).squeeze(0).cpu().numpy()


    # Find the N largest values in response_matrix
    response_flat = response_matrix.flatten()
    top_n_indices = np.argsort(response_flat)[-show_number:][::-1]
    top_n_values = response_flat[top_n_indices]
    top_n_coords = np.unravel_index(top_n_indices, response_matrix.shape)

    # Convert to list of (y, x) coordinates with values
    targets = list(zip(top_n_coords[0], top_n_coords[1], top_n_values))

    # Draw circles at detected target positions
    for y, x, val in targets:
        cv2.circle(color_img, (int(x), int(y)), 5, (0, 0, 255), 2)

    cv2.imshow(window_name, color_img)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:  # ESC pressed -> attempt graceful exit
        return False
        
    return True

    

class FrameIterator:
    """
    A flexible iterator class that can retrieve images frame-by-frame from a video file
    or from a sequence of numerically sorted image files.
    """

    def __init__(self, input_path: str, is_video: bool = True, is_silence: bool = True):
        """
        Initialize the iterator.

        Parameters:
            input_path (str):
                - If is_video is True: full path to the video file.
                - If is_video is False: path to folder containing image sequence.
            is_video (bool): Specifies whether input is a video or image sequence.
        """
        self.input_path = input_path
        self.is_video = is_video
        self.current_index = 0
        self.total_frames = 0
        self.is_open = False
        self.is_silence = is_silence    

        if self.is_video:
            self._init_video_source()
        else:
            self._init_image_sequence_source()

    # --- Video processing logic ---
    def _init_video_source(self):
        """ Initialize video file reading. """
        if not os.path.isfile(self.input_path):
            print(f"Error: Video file not found: {self.input_path}")
            return

        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            print(f"Error: Unable to open video file: {self.input_path}")
            return

        # Get total frame count
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.is_open = True
        if self.is_silence is False:
            print(f"Successfully opened video file. Total frames: {self.total_frames}")

        self.img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def _get_next_frame_from_video(self) -> Optional[cv2.typing.MatLike]:
        """ Read next frame from video. """
        if not self.is_open:
            return None
        
        ret, frame = self.cap.read()
        if ret:
            self.current_index += 1
            return frame
        else:
            # Video reading completed or error occurred
            self.release()
            return None

    # --- Image sequence processing logic ---
    def _init_image_sequence_source(self):
        """ Initialize image sequence folder reading. """
        if not os.path.isdir(self.input_path):
            print(f"Error: Folder not found: {self.input_path}")
            return

        # Find all image files and sort in numerical order
        self.image_files = self._get_sorted_image_files(self.input_path)
        
        if not self.image_files:
            print(f"Error: No image files found in folder {self.input_path}")
            return

        self.total_frames = len(self.image_files)
        self.is_open = True
        if self.is_silence is False:
            print(f"Successfully loaded image sequence. Total images: {self.total_frames}")

        # Read first image to get dimension information
        first_image = cv2.imread(self.image_files[0], cv2.IMREAD_COLOR)
        self.img_height, self.img_width = first_image.shape[:2]

    def _get_next_frame_from_sequence(self) -> Optional[cv2.typing.MatLike]:
        """ Read next image from image sequence. """
        if not self.is_open or self.current_index >= self.total_frames:
            self.release()
            return None

        file_path = self.image_files[self.current_index]
        # cv2.IMREAD_COLOR ensures image is read in color mode
        frame = cv2.imread(file_path, cv2.IMREAD_COLOR)
        
        if frame is None:
             print(f"Warning: Unable to read image file: {file_path}")
        else:
            self.current_index += 1

        return frame

    # --- Core interfaces and helper functions ---
    def get_next_frame(self) -> Optional[cv2.typing.MatLike]:
        """
        [Public interface] Get next image (or frame).

        Returns:
            Optional[cv2.typing.MatLike]: Returns OpenCV image (NumPy array) if successful;
                                       Returns None if end of sequence reached or error occurred.
        """
        if self.is_video:
            color_img =  self._get_next_frame_from_video()
        else:
            color_img =  self._get_next_frame_from_sequence()

        if color_img is None:
            return None, None, False
        else:
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            return gray_img, color_img, True

    def release(self):
        """ Release resources. """
        if self.is_open:
            if self.is_video and hasattr(self, 'cap') and self.cap.isOpened():
                self.cap.release()
            self.is_open = False
        if self.is_silence is False:
            print("\nResources released.")

    def __del__(self):
        """ Ensure resources are released when object is destroyed. """
        self.release()
        
    # --- Natural sorting helper function ---
    @staticmethod
    def _natural_sort_key(s: str) -> List[Union[str, int]]:
        """ Helper function: natural sorting for image sequence. """
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)
        ]

    def _get_sorted_image_files(self, folder_path: str) -> List[str]:
        """ Find and naturally sort image files. """
        extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff')
        all_files = []
        for ext in extensions:
            all_files.extend(glob.glob(os.path.join(folder_path, '*' + ext)))
        
        image_files = [f for f in all_files if os.path.isfile(f)]
        
        # Sort by filename using natural sort
        return sorted(image_files, key=lambda f: self._natural_sort_key(os.path.basename(f)))
