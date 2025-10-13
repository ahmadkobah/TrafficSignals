# Example: How to Add Your Images

## Step-by-Step Guide:

1. **Navigate to the dataset folder** (should be open in Explorer now)

2. **For each type of traffic signal you have:**
   - Go to `train/[signal_type]/` folder
   - Copy 80% of your images there
   - Go to `validation/[signal_type]/` folder  
   - Copy the remaining 20% there

3. **Example with Stop Signs:**
   ```
   If you have 100 stop sign images:
   - Copy 80 images to: train/stop/
   - Copy 20 images to: validation/stop/
   ```

4. **File naming suggestions:**
   ```
   train/stop/stop_001.jpg
   train/stop/stop_002.jpg
   train/yield/yield_001.jpg
   validation/stop/stop_val_001.jpg
   ```

## Minimum Requirements:
- **At least 20 images per class** (for basic training)
- **50-100+ images per class** (for good accuracy)
- **Different images in train vs validation**

## Ready to Train:
Once you have images in the folders, run:
```bash
python main.py
```
Choose option 2, then "Train New Model"!