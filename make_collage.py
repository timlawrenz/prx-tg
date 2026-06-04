import os
from PIL import Image, ImageDraw, ImageFont

def create_comparison_collage(img_indices, category, out_path):
    arm_g_dir = f"experiments/asym-flow-ablation/runs/2026-05-21_1315/validation/step0002000/{category}/"
    arm_i_dir = f"experiments/2026-06-01_1224/validation/step0002000/{category}/"
    
    # Prefix for images
    if category == "reconstruction":
        prefix = "recon"
    elif category == "text_only":
        prefix = "text_only"
    else:
        prefix = "img"
    
    # Load images
    images_g = []
    images_i = []
    
    for idx in img_indices:
        filename = f"{prefix}_{idx:05d}.png"
        path_g = os.path.join(arm_g_dir, filename)
        path_i = os.path.join(arm_i_dir, filename)
        
        if os.path.exists(path_g) and os.path.exists(path_i):
            # Open and resize to a manageable size, say 512x512
            img_g = Image.open(path_g).resize((512, 512), Image.Resampling.LANCZOS)
            img_i = Image.open(path_i).resize((512, 512), Image.Resampling.LANCZOS)
            images_g.append(img_g)
            images_i.append(img_i)
        else:
            print(f"Missing {filename}")
            
    if not images_g:
        print(f"No images found for {category}")
        return

    # Layout dimensions
    padding = 20
    header_height = 80
    row_height = 512
    col_width = 512
    
    num_rows = len(images_g)
    num_cols = 2
    
    collage_width = padding + (col_width + padding) * num_cols
    collage_height = header_height + padding + (row_height + padding) * num_rows
    
    # Create canvas
    bg_color = (250, 247, 242) # #FAF7F2
    text_color = (51, 51, 51)  # #333333
    collage = Image.new('RGB', (collage_width, collage_height), color=bg_color)
    draw = ImageDraw.Draw(collage)
    
    # Draw headers
    # We might not have a TTF font easily available, so we try to load a default TTF or just scale up basic font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 32)
        font_sub = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf", 24)
    except:
        try:
            font = ImageFont.truetype("NotoSans-Bold.ttf", 32)
            font_sub = ImageFont.truetype("NotoSans-Bold.ttf", 24)
        except:
            font = ImageFont.load_default()
            font_sub = font
        
    title = f"Step 2000 Comparison: {category.capitalize()}"
    draw.text((padding, padding), title, fill=text_color, font=font)
    
    draw.text((padding, header_height - 30), "Arm G (AsymFlow BF16)", fill=text_color, font=font_sub)
    draw.text((padding * 2 + col_width, header_height - 30), "Arm I (FP8 Native)", fill=text_color, font=font_sub)
    
    # Paste images
    y_offset = header_height
    for r in range(num_rows):
        x_g = padding
        x_i = padding * 2 + col_width
        
        collage.paste(images_g[r], (x_g, y_offset))
        collage.paste(images_i[r], (x_i, y_offset))
        
        y_offset += row_height + padding

    collage.save(out_path)
    print(f"Saved {out_path}")

indices_recon = [0, 4, 8, 12]
create_comparison_collage(indices_recon, "reconstruction", "arm_g_vs_i_recon.png")

indices_text = [0, 5, 10, 15]
create_comparison_collage(indices_text, "text_only", "arm_g_vs_i_text.png")
