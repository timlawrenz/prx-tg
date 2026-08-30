import torch
import open_clip

print("Creating model...", flush=True)
m, _, prep = open_clip.create_model_and_transforms("ViT-B/32", pretrained="openai")
m = m.cuda().eval()
print("Model on cuda, running forward...", flush=True)
x = torch.randn(1, 3, 224, 224).cuda()
with torch.no_grad():
    y = m.encode_image(x)
print("CLIP forward OK:", y.shape, flush=True)
