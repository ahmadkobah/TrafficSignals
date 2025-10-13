# Dataset Structure for Traffic Signal Classifier

## How to Organize Your Training Data

### Folder Structure:
```
dataset/
├── train/                    ← 80% of your images go here
│   ├── stop/                 ← Put all STOP sign images here
│   ├── yield/                ← Put all YIELD sign images here
│   ├── speed_limit/          ← Put all SPEED LIMIT images here
│   ├── warning/              ← Put all WARNING sign images here
│   ├── no_entry/             ← Put all NO ENTRY sign images here
│   ├── mandatory/            ← Put all MANDATORY sign images here
│   └── information/          ← Put all INFORMATION sign images here
└── validation/               ← 20% of your images go here
    ├── stop/                 ← Different STOP images for testing
    ├── yield/                ← Different YIELD images for testing
    ├── speed_limit/          ← Different SPEED LIMIT images for testing
    ├── warning/              ← Different WARNING images for testing
    ├── no_entry/             ← Different NO ENTRY images for testing
    ├── mandatory/            ← Different MANDATORY images for testing
    └── information/          ← Different INFORMATION images for testing
```

## Instructions:

### Step 1: Collect Images
- Gather traffic signal images from various sources
- Aim for at least 50-100 images per class
- Use different angles, lighting conditions, and backgrounds

### Step 2: Sort Images
- Put 80% of images in `train/` folders
- Put 20% of images in `validation/` folders
- Make sure validation images are different from training images

### Step 3: Supported File Formats
- JPG/JPEG (recommended)
- PNG
- BMP
- TIFF

### Step 4: Image Quality Tips
- Clear, well-lit images work best
- Minimum resolution: 100x100 pixels
- Traffic signal should be clearly visible
- Avoid blurry or heavily distorted images

### Example:
If you have 100 stop sign images:
- Put 80 images in `train/stop/`
- Put 20 images in `validation/stop/`

## Ready to Train?

Once you've organized your images, run:
```bash
python main.py
```

Choose option 2 (GUI) and click "Train New Model"!

## Need More Categories?

You can add more folders for different types of signs:
- Create new folders with descriptive names
- Keep the same structure in both train/ and validation/
- Examples: pedestrian_crossing, school_zone, construction, etc.