import json
import os
import glob
from PIL import Image, ImageDraw, ImageFont

os.makedirs("contact_sheets", exist_ok=True)
videos = [f"L22_V{str(i).zfill(3)}" for i in range(1, 13)]

for vid in videos:
    frames_dir = f"data/processed-validation/m27-representative-12-videos/{vid}/frames"
    images = sorted(glob.glob(f"{frames_dir}/*.jpg"))
    if not images:
        continue
    
    # Select 25 evenly spaced frames
    step = max(1, len(images) // 25)
    selected = images[::step][:25]
    
    # 5x5 grid
    w, h = 1280 // 4, 720 // 4 # downscale to fit
    grid = Image.new('RGB', (w * 5, h * 5))
    
    for i, img_path in enumerate(selected):
        if i >= 25: break
        img = Image.open(img_path).resize((w, h))
        
        # Draw frame ID
        frame_id = os.path.basename(img_path).split('.')[0]
        draw = ImageDraw.Draw(img)
        draw.rectangle([0, 0, 100, 30], fill="black")
        draw.text((5, 5), frame_id, fill="white")
        
        x = (i % 5) * w
        y = (i // 5) * h
        grid.paste(img, (x, y))
        
    grid.save(f"contact_sheets/{vid}.jpg", quality=80)
    print(f"Saved contact_sheets/{vid}.jpg")

