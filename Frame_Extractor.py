import os
import cv2
import numpy as np
import json
import argparse
import time
from datetime import datetime, timedelta
import sys
import shutil
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import importlib.util
import torch
from scenedetect import open_video, SceneManager, FrameTimecode
from scenedetect.detectors import ContentDetector, ThresholdDetector

# Check if PyTorch with CUDA support is available
def check_gpu_support():
    """
    Check if GPU support is available using PyTorch.
    Returns: (cuda_available, gpu_name, cuda_info)
    """
    try:
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_name = torch.cuda.get_device_name(0)
            cuda_info = {
                'name': gpu_name,
                'total_memory': torch.cuda.get_device_properties(0).total_memory,
                'cuda_version': torch.version.cuda
            }
            return True, gpu_name, cuda_info
    except Exception as e:
        print(f"Error during GPU check with PyTorch: {str(e)}")
        return False, "GPU not available", {}
    
    return False, "GPU not available", {}

# Check GPU support
CUDA_AVAILABLE, GPU_NAME, CUDA_INFO = check_gpu_support()

# Default configuration
DEFAULT_CONFIG = {
    'max_frames': 2000,
    'sharpness_window': 7,
    'use_gpu': CUDA_AVAILABLE,
    'distribution_method': 'proportional',
    'min_scene_duration': 10.0,  # seconds
    'max_frames_per_scene': 5,
    'output_format': 'jpg',
    'jpg_quality': 95,
    'output_dir': 'extracted_frames',
    'scene_threshold': 27.0,
    'frames_per_10s': 1,  # How many frames to extract every 10 seconds of scene
    'batch_size': 4 if CUDA_AVAILABLE else 1,  # Process multiple frames simultaneously
    'start_time': "00:00:00",  # Analysis start point (HH:MM:SS)
    'end_time': "",        # Analysis end point (empty = until end of video)
    'use_time_range': False   # Whether to use the specified time range
}

def format_time(seconds):
    """Converts seconds to HH:MM:SS format"""
    td = timedelta(seconds=seconds)
    return str(td).split('.')[0]

def time_to_seconds(time_str):
    """Converts a time string HH:MM:SS to seconds"""
    if not time_str:
        return 0
    
    parts = time_str.split(':')
    if len(parts) == 3:  # HH:MM:SS
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:  # MM:SS
        m, s = parts
        return int(m) * 60 + float(s)
    else:  # Only seconds
        return float(time_str)

def seconds_to_time(seconds):
    """Converts seconds to HH:MM:SS format"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"[:8]

# Function to install required packages if needed
def ensure_gpu_packages():
    """Checks and suggests installation of GPU packages if necessary"""
    if not CUDA_AVAILABLE:
        print("\n⚠️ GPU support not detected!")
        print("To enable GPU acceleration:")
        print("1. Make sure you have NVIDIA drivers installed")
        print("2. Install the necessary packages with:")
        print("   pip install -r requirements_gpu.txt")
        print("\nNOTE: The script will still run in CPU mode")
        
        choice = input("\nDo you want to continue without GPU acceleration? (y/n): ").lower()
        if choice not in ['y', 'yes']:
            print("\nInstallation aborted. Install GPU requirements and try again.")
            sys.exit(1)
    return CUDA_AVAILABLE

def calculate_sharpness_cpu(image):
    """
    Calculate the sharpness level of an image using Laplacian variance (CPU).
    Higher values = sharper image.
    """
    # Convert to grayscale if necessary
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian and its variance
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

def calculate_sharpness_pytorch(image, device=None):
    """
    Calculate the sharpness level of an image using PyTorch.
    Higher values = sharper image.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Convert to grayscale if necessary
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Convert to PyTorch tensor and move to GPU
        img_tensor = torch.from_numpy(gray.astype(np.float32)).to(device)
        
        # Create Laplacian kernel
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=device).view(1, 1, 3, 3)
        
        # Add batch and channel dimensions
        if len(img_tensor.shape) == 2:
            img_tensor = img_tensor.unsqueeze(0).unsqueeze(0)
        
        # Apply Laplacian filter
        with torch.no_grad():
            laplacian = torch.nn.functional.conv2d(img_tensor, laplacian_kernel, padding=1)
            
            # Calculate variance (sharpness index)
            var = torch.var(laplacian).item()
            
        return var
    except Exception as e:
        print(f"⚠️ GPU error during sharpness calculation: {str(e)}")
        print("⚠️ Switching to CPU mode...")
        # Fallback to CPU calculation if there are errors
        return calculate_sharpness_cpu(image)

def find_sharpest_frame_cpu(cap, target_frame, window_size=5):
    """
    Search for the sharpest frame in a window around the target one (CPU).
    Returns the index of the sharpest frame and the frame itself.
    """
    start_frame = max(0, target_frame - window_size)
    end_frame = target_frame + window_size
    
    best_sharpness = -1
    best_frame = None
    best_idx = -1
    
    for idx in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if ret:
            sharpness = calculate_sharpness_cpu(frame)
            
            if sharpness > best_sharpness:
                best_sharpness = sharpness
                best_frame = frame
                best_idx = idx
    
    # Return to original index
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    
    return best_idx, best_frame, best_sharpness

def find_sharpest_frame_pytorch(cap, target_frame, window_size=5, use_gpu=False):
    """
    Search for the sharpest frame in a window around the target one using PyTorch.
    Returns the index of the sharpest frame and the frame itself.
    """
    # If GPU is not requested or not available, use CPU
    if not use_gpu or not torch.cuda.is_available():
        return find_sharpest_frame_cpu(cap, target_frame, window_size)
    
    start_frame = max(0, target_frame - window_size)
    end_frame = target_frame + window_size
    
    best_sharpness = -1
    best_frame = None
    best_idx = -1
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        for idx in range(start_frame, end_frame + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            
            if ret:
                sharpness = calculate_sharpness_pytorch(frame, device)
                
                if sharpness > best_sharpness:
                    best_sharpness = sharpness
                    best_frame = frame
                    best_idx = idx
        
        # Return to original index
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        
        return best_idx, best_frame, best_sharpness
    except Exception as e:
        print(f"⚠️ Error during GPU processing: {str(e)}")
        print("⚠️ Switching to CPU mode...")
        return find_sharpest_frame_cpu(cap, target_frame, window_size)

def process_frame_batch_pytorch(cap, frame_indices, window_size=5):
    """
    Process a batch of frames in parallel using PyTorch.
    Returns a list of tuples (frame_idx, best_idx, best_frame, sharpness)
    """
    if not torch.cuda.is_available():
        results = []
        for idx in frame_indices:
            best_idx, best_frame, sharpness = find_sharpest_frame_cpu(cap, idx, window_size)
            results.append((idx, best_idx, best_frame, sharpness))
        return results
    
    results = []
    device = torch.device("cuda")
    
    for frame_idx in frame_indices:
        # Search in the frame window
        best_idx, best_frame, best_sharpness = find_sharpest_frame_pytorch(
            cap, frame_idx, window_size, True
        )
        results.append((frame_idx, best_idx, best_frame, best_sharpness))
    
    return results

def calculate_frames_for_scene(scene_duration, config):
    """
    Calculate the number of frames to extract for a scene based on its duration.
    """
    if config['distribution_method'] == 'fixed':
        return min(config['max_frames_per_scene'], 1)
    else:  # proportional
        # Calculate how many frames based on duration (1 frame every X seconds)
        frames_to_extract = max(1, int(scene_duration / (10.0 / config['frames_per_10s'])))
        return min(frames_to_extract, config['max_frames_per_scene'])

def extract_frames_from_scenes(video_path, config):
    """
    Extract frames from detected scenes in a video, choosing the sharpest ones
    """
    # Get movie name from path
    video_filename = os.path.basename(video_path)
    video_name = os.path.splitext(video_filename)[0]
    
    print(f"\n{'='*70}")
    print(f"   EXTRACTING FRAMES FROM {video_name.upper()}")
    print(f"{'='*70}\n")
    
    # Create output directory
    output_dir = os.path.expanduser(config['output_dir'])
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a specific subfolder for this movie
    film_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(film_output_dir, exist_ok=True)
    
    # Open the video
    video = open_video(video_path)
    
    # Get video information
    video_fps = video.frame_rate
    total_frames = video.duration.get_frames()
    video_duration = total_frames / video_fps
    
    # Convert start and end points to seconds and frames
    start_seconds = time_to_seconds(config['start_time']) if config['use_time_range'] else 0
    end_seconds = time_to_seconds(config['end_time']) if config['use_time_range'] and config['end_time'] else float('inf')
    
    # Convert to frames
    start_frame = int(start_seconds * video_fps) if config['use_time_range'] else 0
    end_frame = int(end_seconds * video_fps) if config['use_time_range'] and end_seconds < float('inf') else int(total_frames)
    
    # If we're using a time range, modify the video passed to the scene manager
    if config['use_time_range']:
        # Create frame timecodes for start and end
        start_timecode = FrameTimecode(timecode=start_frame, fps=video_fps)
        end_timecode = FrameTimecode(timecode=min(end_frame, total_frames), fps=video_fps)
    else:
        start_timecode = None
        end_timecode = None
    
    print(f"📹 Video information:")
    print(f"   • File: {video_filename}")
    print(f"   • FPS: {video_fps:.2f}")
    print(f"   • Duration: {format_time(video_duration)}")
    print(f"   • Total frames: {total_frames}")
    
    # Show time interval information only if enabled
    if config['use_time_range']:
        if start_seconds > 0:
            print(f"   • Start point: {config['start_time']} ({format_time(start_seconds)})")
            print(f"   • Start frame: {start_frame}")
        if end_seconds < float('inf'):
            print(f"   • End point: {config['end_time']} ({format_time(end_seconds)})")
            print(f"   • End frame: {end_frame}")
        
        print(f"   • Portion to analyze: {format_time(end_seconds - start_seconds)}")
    
    # Show GPU status
    if config['use_gpu'] and CUDA_AVAILABLE:
        print(f"   • GPU acceleration: ✅ Active")
        print(f"   • Detected GPU: {GPU_NAME}")
        if 'total_memory' in CUDA_INFO:
            print(f"   • GPU Memory: {CUDA_INFO['total_memory'] / (1024*1024):.1f} MB")
        print(f"   • Batch size: {config['batch_size']}")
    else:
        print(f"   • GPU acceleration: ❌ Inactive")
    
    print(f"   • Sharpness search window: ±{config['sharpness_window']} frames")
    print(f"   • Distribution method: {config['distribution_method'].capitalize()}")
    print(f"   • Frames per 10s of scene: {config['frames_per_10s']}")
    print()
    
    # Initialize detector to find scenes
    scene_manager = SceneManager()
    detector = ContentDetector(threshold=config['scene_threshold'])
    scene_manager.add_detector(detector)
    
    print("🎬 Starting scene detection...")
    
    # SOLUTION: If we're using a time range, we need to first ensure we set the start/end frames
    # to analyze only that portion of the video
    
    if config['use_time_range']:
        print(f"   • Analyzing video portion: {format_time(start_seconds)} - {format_time(min(end_seconds, video_duration))}")
        
        # Create a subsection of the video using OpenCV to limit the analysis area
        # This is a much more efficient and direct solution
        
        # Open the video with OpenCV
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("⚠️ Error opening the video.")
            return
        
        # Set the initial position of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Create a temporary video file for the portion of interest
        temp_path = os.path.join(os.path.dirname(video_path), f"temp_{int(time.time())}.mp4")
        
        # Get video details needed for the writer
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create a video writer for the temporary file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))
        
        # Number of frames to extract
        frames_to_extract = end_frame - start_frame
        
        print(f"   • Preparing temporary file for portion analysis...")
        
        # Read and write the frames in the portion of interest
        frame_count = 0
        while cap.isOpened() and frame_count < frames_to_extract:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            frame_count += 1
            
            # Show progress
            if frame_count % 100 == 0:
                progress = (frame_count / frames_to_extract) * 100
                print(f"   • Preparation: {progress:.1f}% completed", end="\r")
        
        # Release resources
        cap.release()
        out.release()
        print(f"   • Preparation completed: {frame_count} frames extracted")
        
        # Now use the temporary file for scene detection
        temp_video = open_video(temp_path)
        
        # Detect scenes in the temporary file (which contains only the portion of interest)
        scene_manager.detect_scenes(temp_video, show_progress=True)
        
        # Get the detected scenes
        scene_list_relative = scene_manager.get_scene_list()
        
        # Convert scenes to absolute coordinates (relative to the original video)
        scene_list = []
        for scene in scene_list_relative:
            scene_start, scene_end = scene
            # Add the start offset
            absolute_start = FrameTimecode(timecode=scene_start.get_frames() + start_frame, fps=video_fps)
            absolute_end = FrameTimecode(timecode=scene_end.get_frames() + start_frame, fps=video_fps)
            scene_list.append((absolute_start, absolute_end))
        
        # Delete the temporary file
        try:
            os.remove(temp_path)
            print(f"   • Temporary file deleted")
        except OSError as e:
            print(f"   • Error deleting temporary file: {e}")
    else:
        print("   • Analyzing entire video")
        scene_manager.detect_scenes(video, show_progress=True)
        scene_list = scene_manager.get_scene_list()
    
    # Get list of detected scenes
    num_scenes = len(scene_list)
    
    print(f"\n✅ Detected scenes: {num_scenes}")
    print(f"🎯 Maximum target: {config['max_frames']} total frames\n")
    
    # Calculate total number of frames to extract based on scene duration
    total_estimated_frames = 0
    scene_frame_counts = []
    
    for scene in scene_list:
        scene_start_time, scene_end_time = scene
        scene_duration = (scene_end_time.get_frames() - scene_start_time.get_frames()) / video_fps
        frames_for_scene = calculate_frames_for_scene(scene_duration, config)
        scene_frame_counts.append(frames_for_scene)
        total_estimated_frames += frames_for_scene
    
    # Adjust the number of frames if we exceed the limit
    if total_estimated_frames > config['max_frames']:
        reduction_factor = config['max_frames'] / total_estimated_frames
        scene_frame_counts = [max(1, int(count * reduction_factor)) for count in scene_frame_counts]
        total_estimated_frames = sum(scene_frame_counts)
    
    print(f"📊 Extraction strategy:")
    print(f"   • Expected frames: {total_estimated_frames}")
    if config['distribution_method'] == 'proportional':
        print(f"   • Distribution: Proportional to scene duration")
        print(f"   • Frames per 10s of scene: {config['frames_per_10s']}")
    else:
        print(f"   • Distribution: Fixed ({config['max_frames_per_scene']} per scene)")
    print()
    
    # Extract frames
    frame_count = 0
    start_time = time.time()
    
    print(f"{'='*70}")
    print(f"   STARTING FRAME EXTRACTION")
    print(f"{'='*70}\n")
    
    # Open the video for extraction
    cap = cv2.VideoCapture(video_path)
    
    # Prepare a queue of frames to process for batch processing
    for scene_idx, (scene, frames_to_extract) in enumerate(zip(scene_list, scene_frame_counts)):
        scene_start_time, scene_end_time = scene
        scene_start_frame = scene_start_time.get_frames()
        scene_end_frame = scene_end_time.get_frames()
        scene_duration = (scene_end_frame - scene_start_frame) / video_fps
        
        print(f"\n🎞️ Scene {scene_idx+1}/{num_scenes}")
        print(f"   • Start: {format_time(scene_start_time.get_seconds())}")
        print(f"   • End: {format_time(scene_end_time.get_seconds())}")
        print(f"   • Duration: {scene_duration:.2f}s")
        print(f"   • Frames to extract: {frames_to_extract}")
        
        # If there are no frames to extract, move to the next scene
        if frames_to_extract <= 0:
            continue
            
        # Calculate indices of frames to extract
        if scene_end_frame - scene_start_frame <= frames_to_extract:
            # Take all available frames if there are fewer than needed
            frame_indices = list(range(int(scene_start_frame), int(scene_end_frame)))
        else:
            # Distribute frames evenly in the scene
            step = (scene_end_frame - scene_start_frame) / frames_to_extract
            frame_indices = [int(scene_start_frame + i * step) for i in range(frames_to_extract)]
        
        # If we've reached the maximum frame limit, stop
        if frame_count >= config['max_frames']:
            print(f"\n🎯 Reached maximum limit of {config['max_frames']} frames!")
            break
        
        # Batch extraction with GPU
        if config['use_gpu'] and CUDA_AVAILABLE:
            # Split frames into batches
            batch_size = min(config['batch_size'], len(frame_indices))
            batches = [frame_indices[i:i + batch_size] for i in range(0, len(frame_indices), batch_size)]
            
            print(f"   • Processing in {len(batches)} batches with GPU...")
            
            for batch_idx, batch in enumerate(batches):
                if frame_count >= config['max_frames']:
                    break
                    
                # Process batch of frames
                results = process_frame_batch_pytorch(cap, batch, config['sharpness_window'])
                
                # Save the results
                for i, (original_idx, best_idx, best_frame, sharpness) in enumerate(results):
                    if frame_count >= config['max_frames']:
                        break
                        
                    if best_frame is not None:
                        # Use the time of the frame actually chosen
                        frame_time = best_idx / video_fps
                        filename = f"scene_{scene_idx+1:03d}_{frame_time:.2f}s.{config['output_format']}"
                        filepath = os.path.join(film_output_dir, filename)
                        
                        # Save the frame
                        if config['output_format'].lower() == 'jpg':
                            cv2.imwrite(filepath, best_frame, [cv2.IMWRITE_JPEG_QUALITY, config['jpg_quality']])
                        else:
                            cv2.imwrite(filepath, best_frame)
                        
                        frame_count += 1
                        
                        # Calculate delta relative to initially chosen frame
                        frame_delta = best_idx - original_idx
                        delta_info = f"(delta: {frame_delta:+d})" if frame_delta != 0 else "(original frame)"
                        
                        # Calculate ETA
                        if frame_count > 0:
                            elapsed_time = time.time() - start_time
                            frames_per_second = frame_count / elapsed_time
                            frames_left = min(config['max_frames'], total_estimated_frames) - frame_count
                            eta = frames_left / frames_per_second if frames_per_second > 0 else 0
                            
                            print(f"   ✓ Frame {i+1}/{len(batch)} of batch {batch_idx+1}: {filename} | Sharpness: {sharpness:.2f} {delta_info} | ETA: {format_time(eta)}")
                        else:
                            print(f"   ✓ Frame {i+1}/{len(batch)} of batch {batch_idx+1}: {filename} | Sharpness: {sharpness:.2f} {delta_info}")
                
                # Show progress after each batch
                if frame_count > 0:
                    progress_percentage = frame_count / total_estimated_frames * 100
                    print(f"   📊 Progress: {progress_percentage:.1f}% completed | {frame_count}/{total_estimated_frames} frames")
        else:
            # Standard extraction (non-parallel)
            for i, frame_idx in enumerate(frame_indices):
                if frame_count >= config['max_frames']:
                    break
                    
                # Find the sharpest frame in the window
                best_idx, best_frame, sharpness = find_sharpest_frame_cpu(
                    cap, frame_idx, config['sharpness_window']
                )
                
                if best_frame is not None:
                    # Use the time of the frame actually chosen
                    frame_time = best_idx / video_fps
                    filename = f"scene_{scene_idx+1:03d}_{frame_time:.2f}s.{config['output_format']}"
                    filepath = os.path.join(film_output_dir, filename)
                    
                    # Save the frame
                    if config['output_format'].lower() == 'jpg':
                        cv2.imwrite(filepath, best_frame, [cv2.IMWRITE_JPEG_QUALITY, config['jpg_quality']])
                    else:
                        cv2.imwrite(filepath, best_frame)
                    
                    frame_count += 1
                    
                    # Calculate delta relative to initially chosen frame
                    frame_delta = best_idx - frame_idx
                    delta_info = f"(delta: {frame_delta:+d})" if frame_delta != 0 else "(original frame)"
                    
                    # Calculate remaining time
                    elapsed_time = time.time() - start_time
                    if frame_count > 0:
                        avg_time_per_frame = elapsed_time / frame_count
                        frames_left = min(config['max_frames'], total_estimated_frames) - frame_count
                        eta = avg_time_per_frame * frames_left
                        print(f"   ✓ Frame {i+1}/{len(frame_indices)}: {filename} | Sharpness: {sharpness:.2f} {delta_info} | ETA: {format_time(eta)}")
                    else:
                        print(f"   ✓ Frame {i+1}/{len(frame_indices)}: {filename} | Sharpness: {sharpness:.2f} {delta_info}")
        
        # Update statistics after each scene
        progress_percentage = (scene_idx + 1) / num_scenes * 100
        scenes_left = num_scenes - scene_idx - 1
        if scenes_left > 0 and frame_count > 0:
            elapsed_time = time.time() - start_time
            frames_per_second = frame_count / elapsed_time
            estimated_total_time = total_estimated_frames / frames_per_second
            time_left = max(0, estimated_total_time - elapsed_time)
            
            print(f"   📊 Progress: {progress_percentage:.1f}% completed | {frame_count}/{total_estimated_frames} frames | Estimated time: {format_time(time_left)}")
    
    # Close the video
    cap.release()
    
    print(f"\n{'='*70}")
    print(f"   EXTRACTION COMPLETED!")
    print(f"{'='*70}\n")
    print(f"✨ Total frames extracted: {frame_count}")
    print(f"⏱️ Total time: {format_time(time.time() - start_time)}")
    print(f"📁 Output directory: {film_output_dir}")
    print(f"\nThe extracted frames are ready for training your LoRA model!")

def find_video_files(directory):
    """Find all video files in the specified directory"""
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm']
    video_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    return video_files

def save_config(config, filepath):
    """Save the configuration to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=4)

def load_config(filepath):
    """Load the configuration from a JSON file"""
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            loaded_config = json.load(f)
            
            # Make sure all keys from DEFAULT_CONFIG are present
            for key in DEFAULT_CONFIG:
                if key not in loaded_config:
                    loaded_config[key] = DEFAULT_CONFIG[key]
                    
            return loaded_config
    return DEFAULT_CONFIG.copy()

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print the program header"""
    print(f"\n{'='*70}")
    print(f"   FRAME EXTRACTOR")
    if CUDA_AVAILABLE:
        print(f"   GPU: {GPU_NAME}")
    else:
        print(f"   GPU: Not available")
    print(f"{'='*70}\n")

def print_menu(header, options, back_option=True):
    """Print a menu with numbered options"""
    clear_screen()
    print_header()
    
    print(f"{header}\n")
    
    for i, option in enumerate(options):
        print(f"{i+1}. {option}")
    
    if back_option:
        print(f"\n0. Go back")
    
    return input("\nChoice: ")

def menu_time_range(video_path, config):
    """Menu for setting the time range for analysis"""
    video_filename = os.path.basename(video_path)
    
    # Get video duration
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("\n⚠️ Unable to open the video to check duration.")
        return config
        
    # Get video information
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    duration_str = format_time(duration)
    cap.release()
    
    clear_screen()
    print_header()
    print(f"TIME RANGE - {video_filename}\n")
    print(f"Total video duration: {duration_str}\n")
    # Display current status
    if config['use_time_range']:
        start_time = config['start_time']
        end_time = config['end_time'] if config['end_time'] else "End of video"
        print(f"Current range: {start_time} → {end_time}\n")
    else:
        print("Time range: Not used (entire video)\n")
    
    # Menu options
    options = [
        "Enable/Disable time range" + (" [Enabled]" if config['use_time_range'] else " [Disabled]"),
        "Set start point",
        "Set end point",
        "Use entire video (reset range)"
    ]
    
    choice = print_menu("Time range options:", options)
    
    if choice == '0':
        return config
    elif choice == '1':
        # Enable/Disable range
        config['use_time_range'] = not config['use_time_range']
        status = "enabled" if config['use_time_range'] else "disabled"
        print(f"\n✅ Time range {status}.")
        input("\nPress ENTER to continue...")
    elif choice == '2' and config['use_time_range']:
        # Set start point
        clear_screen()
        print_header()
        print(f"SET START POINT - {video_filename}\n")
        print(f"Total video duration: {duration_str}")
        print(f"Format: HH:MM:SS or MM:SS\n")
        
        new_value = input("New start point (leave empty for start of video): ")
        
        if new_value:
            try:
                seconds = time_to_seconds(new_value)
                # Check if the time is valid
                if seconds >= 0:
                    if seconds < duration:
                        config['start_time'] = new_value
                        print(f"\n✅ Start point set: {new_value} ({format_time(seconds)})")
                        
                        # Update end point if necessary
                        end_seconds = time_to_seconds(config['end_time']) if config['end_time'] else float('inf')
                        if end_seconds < seconds:
                            config['end_time'] = ""
                            print(f"⚠️ End point reset to end of video (was before start point)")
                    else:
                        print(f"\n⚠️ Start point is beyond video duration ({duration_str}).")
                else:
                    print("\n⚠️ Time must be positive.")
            except ValueError:
                print("\n⚠️ Invalid time format. Use HH:MM:SS or MM:SS.")
        else:
            config['start_time'] = "00:00:00"
            print("\n✅ Start point reset to beginning of video.")
        
        input("\nPress ENTER to continue...")
    elif choice == '3' and config['use_time_range']:
        # Set end point
        clear_screen()
        print_header()
        print(f"SET END POINT - {video_filename}\n")
        print(f"Total video duration: {duration_str}")
        print(f"Current start point: {config['start_time']}")
        print(f"Format: HH:MM:SS or MM:SS\n")
        
        new_value = input("New end point (leave empty for end of video): ")
        
        if new_value:
            try:
                seconds = time_to_seconds(new_value)
                # Check if the time is valid
                if seconds >= 0:
                    # Check if end point is after start point
                    start_seconds = time_to_seconds(config['start_time'])
                    if seconds > start_seconds:
                        config['end_time'] = new_value
                        print(f"\n✅ End point set: {new_value} ({format_time(seconds)})")
                        if seconds > duration:
                            print(f"⚠️ Note: End point is beyond video duration ({duration_str}), end of video will be used.")
                    else:
                        print(f"\n⚠️ End point must be after start point ({config['start_time']}).")
                else:
                    print("\n⚠️ Time must be positive.")
            except ValueError:
                print("\n⚠️ Invalid time format. Use HH:MM:SS or MM:SS.")
        else:
            config['end_time'] = ""
            print("\n✅ End point reset to end of video.")
            
        input("\nPress ENTER to continue...")
    elif choice == '4':
        # Reset range (use entire video)
        config['use_time_range'] = False
        config['start_time'] = "00:00:00"
        config['end_time'] = ""
        print("\n✅ Time range reset (entire video).")
        input("\nPress ENTER to continue...")
    else:
        print("\n⚠️ Invalid choice. Try again.")
        input("\nPress ENTER to continue...")
    
    # Return to time range menu
    return menu_time_range(video_path, config)
    
def main_menu():
    """Main application menu"""
    # Check GPU at startup
    ensure_gpu_packages()
    
    config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')
    config = load_config(config_file)
    
    while True:
        clear_screen()
        print_header()
        
        # Find all videos
        current_dir = os.path.dirname(os.path.abspath(__file__))
        video_files = find_video_files(current_dir)
        
        if not video_files:
            print("❌ No video files found in the current directory.")
            print("Please add video files (mp4, mkv, avi, mov, wmv) in the same folder as the script.")
            input("\nPress ENTER to exit...")
            sys.exit(1)
        
        print("Videos found in the folder:")
        for i, file in enumerate(video_files):
            print(f"{i+1}. {file}")
        
        choice = input("\nSelect a video (1-{}) or 'q' to exit: ".format(len(video_files)))
        
        if choice.lower() == 'q':
            print("\nGoodbye!")
            sys.exit(0)
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(video_files):
                selected_video = video_files[idx]
                video_path = os.path.join(current_dir, selected_video)
                
                # Menu for the selected video
                menu_video(video_path, config, config_file)
            else:
                print("\n⚠️ Invalid choice. Try again.")
                input("\nPress ENTER to continue...")
        except ValueError:
            print("\n⚠️ Enter a valid number.")
            input("\nPress ENTER to continue...")

def menu_video(video_path, config, config_file):
    """Menu for the selected video"""
    video_filename = os.path.basename(video_path)
    
    while True:
        options = [
            f"Start extraction with default parameters",
            f"Customize parameters",
            f"Set time range",
            f"View parameter descriptions",
            f"Reset to default parameters"
        ]
        
        choice = print_menu(f"Selected video: {video_filename}", options)
        
        if choice == '0':
            return
        elif choice == '1':
            # Start with default parameters
            extract_frames_from_scenes(video_path, config)
            input("\nPress ENTER to return to menu...")
        elif choice == '2':
            # Customize parameters
            config = menu_customize_parameters(video_path, config, config_file)
        elif choice == '3':
            # Set time range
            config = menu_time_range(video_path, config)
            save_config(config, config_file)
        elif choice == '4':
            # View parameter descriptions
            menu_parameter_descriptions()
        elif choice == '5':
            # Reset to default parameters
            config = DEFAULT_CONFIG.copy()
            save_config(config, config_file)
            print("\n✅ Parameters reset to default values.")
            input("\nPress ENTER to continue...")
        else:
            print("\n⚠️ Invalid choice. Try again.")
            input("\nPress ENTER to continue...")

def menu_customize_parameters(video_path, config, config_file):
    """Menu for customizing parameters"""
    while True:
        options = [
            f"Maximum number of frames [{config['max_frames']}]",
            f"Sharpness search window [{config['sharpness_window']}]",
            f"GPU usage [{'Enabled' if config['use_gpu'] else 'Disabled'}]",
            f"Frame distribution [{'Proportional' if config['distribution_method'] == 'proportional' else 'Fixed'}]",
            f"Frames per 10 seconds [{config['frames_per_10s']}]",
            f"Max frames per scene [{config['max_frames_per_scene']}]",
            f"Output format [{config['output_format']}]",
            f"JPG quality [{config['jpg_quality']}]",
            f"Output directory [{config['output_dir']}]",
            f"Scene detection threshold [{config['scene_threshold']}]",
            f"GPU batch size [{config['batch_size']}]",
            f"Start with these parameters"
        ]
        
        choice = print_menu("CUSTOMIZE PARAMETERS", options)
        
        if choice == '0':
            # Save configuration and go back
            save_config(config, config_file)
            return config
        elif choice == '1':
            # Maximum number of frames
            try:
                new_value = int(input("\nEnter new value for maximum number of frames: "))
                if new_value > 0:
                    config['max_frames'] = new_value
                    print(f"\n✅ Value updated: {new_value}")
                else:
                    print("\n⚠️ Value must be greater than 0.")
            except ValueError:
                print("\n⚠️ Enter a valid number.")
        elif choice == '2':
            # Sharpness search window
            try:
                new_value = int(input("\nEnter new value for sharpness search window: "))
                if new_value > 0:
                    config['sharpness_window'] = new_value
                    print(f"\n✅ Value updated: {new_value}")
                else:
                    print("\n⚠️ Value must be greater than 0.")
            except ValueError:
                print("\n⚠️ Enter a valid number.")
        elif choice == '3':
            # GPU usage
            if not CUDA_AVAILABLE:
                print("\n⚠️ GPU not available on your system.")
                input("\nPress ENTER to continue...")
                continue
                
            gpu_choice = input("\nEnable GPU acceleration? (y/n): ").lower()
            if gpu_choice in ['y', 'yes']:
                config['use_gpu'] = True
                print("\n✅ GPU acceleration enabled.")
            elif gpu_choice in ['n', 'no']:
                config['use_gpu'] = False
                print("\n✅ GPU acceleration disabled.")
            else:
                print("\n⚠️ Invalid choice.")
        elif choice == '4':
            # Distribution method
            dist_choice = input("\nChoose distribution method (p=proportional, f=fixed): ").lower()
            if dist_choice in ['p', 'proportional']:
                config['distribution_method'] = 'proportional'
                print("\n✅ Proportional distribution set.")
            elif dist_choice in ['f', 'fixed']:
                config['distribution_method'] = 'fixed'
                print("\n✅ Fixed distribution set.")
            else:
                print("\n⚠️ Invalid choice.")
        elif choice == '5':
            # Frames per 10 seconds
            try:
                new_value = float(input("\nEnter number of frames to extract every 10 seconds: "))
                if new_value > 0:
                    config['frames_per_10s'] = new_value
                    print(f"\n✅ Value updated: {new_value}")
                else:
                    print("\n⚠️ Value must be greater than 0.")
            except ValueError:
                print("\n⚠️ Enter a valid number.")
        elif choice == '6':
            # Max frames per scene
            try:
                new_value = int(input("\nEnter maximum number of frames per scene: "))
                if new_value > 0:
                    config['max_frames_per_scene'] = new_value
                    print(f"\n✅ Value updated: {new_value}")
                else:
                    print("\n⚠️ Value must be greater than 0.")
            except ValueError:
                print("\n⚠️ Enter a valid number.")
        elif choice == '7':
            # Output format
            format_choice = input("\nChoose output format (jpg/png): ").lower()
            if format_choice in ['jpg', 'jpeg']:
                config['output_format'] = 'jpg'
                print("\n✅ JPG format set.")
            elif format_choice in ['png']:
                config['output_format'] = 'png'
                print("\n✅ PNG format set.")
            else:
                print("\n⚠️ Invalid format. Use jpg or png.")
        elif choice == '8':
            # JPG quality
            try:
                new_value = int(input("\nEnter JPG quality (1-100): "))
                if 1 <= new_value <= 100:
                    config['jpg_quality'] = new_value
                    print(f"\n✅ Quality set: {new_value}")
                else:
                    print("\n⚠️ Quality must be between 1 and 100.")
            except ValueError:
                print("\n⚠️ Enter a valid number.")
        elif choice == '9':
            # Output directory
            new_value = input("\nEnter output directory: ")
            if new_value:
                config['output_dir'] = new_value
                print(f"\n✅ Directory set: {new_value}")
            else:
                print("\n⚠️ Invalid directory.")
        elif choice == '10':
            # Scene detection threshold
            try:
                new_value = float(input("\nEnter scene detection threshold (1-100, lower values=more scenes): "))
                if 1 <= new_value <= 100:
                    config['scene_threshold'] = new_value
                    print(f"\n✅ Threshold set: {new_value}")
                else:
                    print("\n⚠️ Threshold must be between 1 and 100.")
            except ValueError:
                print("\n⚠️ Enter a valid number.")
        elif choice == '11':
            # GPU batch size
            if not CUDA_AVAILABLE:
                print("\n⚠️ GPU not available. Setting ignored.")
                input("\nPress ENTER to continue...")
                continue
                
            try:
                new_value = int(input("\nEnter GPU batch size (1-10): "))
                if 1 <= new_value <= 10:
                    config['batch_size'] = new_value
                    print(f"\n✅ Batch size set: {new_value}")
                else:
                    print("\n⚠️ Size must be between 1 and 10.")
            except ValueError:
                print("\n⚠️ Enter a valid number.")
        elif choice == '12':
            # Start with these parameters
            save_config(config, config_file)
            extract_frames_from_scenes(video_path, config)
            input("\nPress ENTER to return to menu...")
        else:
            print("\n⚠️ Invalid choice. Try again.")
            input("\nPress ENTER to continue...")
        
        # Save configuration
        save_config(config, config_file)
    
    return config

def menu_parameter_descriptions():
    """Show detailed parameter descriptions"""
    clear_screen()
    print_header()
    
    print("PARAMETER DESCRIPTIONS\n")
    
    print("1. Maximum number of frames")
    print("   Determines how many frames to extract in total from the video.")
    print("   Higher values = larger dataset, but more processing time.")
    print("   Recommended: 2000-5000 for most videos.")
    
    print("\n2. Sharpness search window")
    print("   Number of frames before and after to analyze to find the sharpest frame.")
    print("   Higher values = sharper frames, but more processing time.")
    print("   Recommended: 5-10 to balance quality and speed.")
    
    print("\n3. GPU usage")
    print("   Enables/disables GPU acceleration for image processing operations.")
    print("   Recommended: Enabled if a compatible NVIDIA GPU is available.")
    
    print("\n4. Frame distribution")
    print("   'Proportional': extracts more frames for long scenes, fewer for short scenes.")
    print("   'Fixed': extracts the same number of frames for each scene.")
    print("   Recommended: Proportional for better coverage.")
    
    print("\n5. Frames per 10 seconds")
    print("   How many frames to extract every 10 seconds of scene (in proportional mode).")
    print("   Higher values = more frames for long scenes.")
    print("   Recommended: 1-2 for most cases.")
    
    print("\n6. Max frames per scene")
    print("   Upper limit of frames that can be extracted from a single scene.")
    print("   Recommended: 5-10 to avoid too many similar images.")
    
    print("\n7. Output format")
    print("   'jpg': more compact, slightly lower quality.")
    print("   'png': better quality, larger files.")
    print("   Recommended: jpg for most cases.")
    
    print("\n8. JPG quality")
    print("   Quality/compression level for JPG files (1-100).")
    print("   Higher values = better quality, larger files.")
    print("   Recommended: 85-95 for a good balance.")
    
    print("\n9. Output directory")
    print("   The folder where extracted frames will be saved.")
    print("   A subfolder with the video name is automatically created.")
    
    print("\n10. Scene detection threshold")
    print("   Sensitivity in detecting scene changes (1-100).")
    print("   Lower values = more scenes detected.")
    print("   Recommended: 25-35 for standard videos, 15-25 for videos with many quick cuts.")
    
    print("\n11. GPU batch size")
    print("   Number of frames to process simultaneously with the GPU.")
    print("   Higher values = more speed, but more GPU memory required.")
    print("   Recommended: 4-8 for GPUs with 8GB+ memory, 2-4 for GPUs with less memory.")
    
    print("\n12. Time range")
    print("   Defines a specific portion of the video to analyze.")
    print("   Useful for skipping opening/closing credits or focusing on specific scenes.")
    print("   Set through the 'Set time range' menu.")
    
    input("\nPress ENTER to return to previous menu...")

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nExecution interrupted by user.")
    except Exception as e:
        print(f"\n\nUnexpected error: {str(e)}")
        input("\nPress ENTER to exit...")
