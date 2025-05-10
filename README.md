# Frame_Extractor
Automatically detects scenes in a video and extracts the sharpest frames for training AI models. Supports GPU acceleration with PyTorch, customizable parameters, time-range selection, and high-quality frame output (JPG/PNG). Perfect for preparing datasets for LoRA fine-tuning.

# Frame Extractor for LoRA Training

This program automatically extracts high-quality frames from video files to create training datasets for LoRA models. The program automatically divides the video into scenes, selects the sharpest frames, and saves them in image format.

## Key Features

- **Automatic scene detection**: Identifies scene changes in the video
- **Intelligent frame selection**: Chooses the sharpest frames for each scene
- **GPU acceleration**: CUDA support for faster processing (optional)
- **Intuitive interface**: Interactive command-line menu
- **Customizable configuration**: Adjust all parameters to suit your needs

## System Requirements

- Python 3.7 or higher
- OpenCV
- PyTorch (for GPU acceleration)
- SceneDetect
- Other Python packages (listed in requirements.txt)
- NVIDIA GPU with compatible drivers (optional, for GPU acceleration)

## Installation

### 1. Clone or download this repository:

```bash
git clone https://github.com/Tranchillo/Frame_Extractor.git
cd Frame_Extractor
```

### 2. Creating a virtual environment (recommended)

It's recommended to use a Python virtual environment to avoid conflicts with other installations:

```bash
# Create the virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

Once the virtual environment is activated, you should see `(venv)` at the beginning of the command line, indicating that you're working in the isolated environment.

### 3. Installing dependencies

#### Check if you have an NVIDIA GPU available

Run the `nvidia-smi` command to check if you have an NVIDIA GPU and which drivers are installed:

```bash
nvidia-smi
```

If the command works, you should see output similar to this:

```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.146.02    Driver Version: 535.146.02    CUDA Version: 12.2   |
| ...                                                                          |
```

Take note of the CUDA version reported (in the example it's 12.2).

#### Installing PyTorch with CUDA support

Based on the CUDA version shown by `nvidia-smi`, choose the correct installation command:

- For CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

- For CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### Installing other dependencies

Install the other required dependencies:

```bash
pip install opencv-python numpy tqdm scenedetect
```

## Usage

### Starting the program

Place the video files (.mp4, .mkv, .avi, etc.) in the same folder as the program, then run:

```bash
# Make sure the virtual environment is activated
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Start the program
python Frame_Extractor.py
```

### Main menu

At startup, the program will display a menu with the available videos in the current folder. Select a number to choose the video to process.

### Selected video menu

After selecting a video, a menu will appear with the following options:

1. **Start extraction with default parameters**: Begin frame extraction immediately
2. **Customize parameters**: Modify extraction parameters
3. **Set time range**: Choose a specific portion of the video
4. **View parameter descriptions**: Detailed information about parameters
5. **Reset to default parameters**: Reset all settings

### Customizable Parameters

All parameters can be adjusted based on your needs:

- **Maximum number of frames**: How many frames to extract in total
- **Sharpness search window**: To select the sharpest frames
- **GPU usage**: Enable/disable hardware acceleration
- **Frame distribution**: Proportional or fixed for each scene
- **Frames per 10 seconds**: Sampling density for long scenes
- **Max frames per scene**: Limit to avoid too many similar images
- **Output format**: JPG or PNG
- **JPG quality**: Compression level for JPG files
- **Output directory**: Where to save the extracted frames
- **Scene detection threshold**: Sensitivity in detecting scene changes
- **GPU batch size**: To optimize parallel processing

### Time Range

You can also set a specific time range to focus on a particular part of the video:

1. **Enable/disable time range**: Enable the use of a range
2. **Set start point**: In HH:MM:SS format
3. **Set end point**: In HH:MM:SS format
4. **Use entire video**: Reset to use the entire video

## Output

The extracted frames are saved in the specified directory (default: `extracted_frames`) in a subfolder with the video file name. Each frame is named with the scene number and timestamp.

## Tips for Better Results

1. **Settings for high-quality videos**:
   - Increase the maximum number of frames to 3000-5000
   - Use proportional distribution
   - Set a wider sharpness search window (7-10)

2. **Settings for fast performance**:
   - Reduce the maximum number of frames to 1000-2000
   - Enable GPU acceleration if available
   - Use a smaller sharpness search window (3-5)

3. **Extracting specific scenes**:
   - Use the "Set time range" option
   - Specify the exact start and end points in HH:MM:SS format

## Troubleshooting

### GPU Issues

If you encounter problems with GPU acceleration:

1. **Check compatibility**: Make sure you installed PyTorch with the correct CUDA version for your drivers
2. **Disable GPU acceleration**: If problems persist, you can always use CPU mode
3. **Update drivers**: Sometimes you may need to update your NVIDIA drivers

### Memory Errors

If the program crashes due to memory problems:

1. **Reduce GPU batch size**: Try lower values
2. **Process fewer frames**: Decrease the maximum number of frames
3. **Process a smaller range**: Use the time range option to process the video in parts

## Italian Version

An Italian version of the program is also available: `Estrattore_Immagini.py`. It works exactly the same way but with all menus and messages in Italian.

## License

This software is distributed under the MIT license.

---

For questions, suggestions, or bug reports, open an issue on GitHub: https://github.com/Tranchillo/Frame_Extractor/issues
