import torch
from PIL import Image

from internvlu import InternVLUPipeline

model_path = "/path/to/InternVL-U"

pipe = InternVLUPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

image = Image.open("test_milk.png").convert("RGB")

out = pipe(
    prompt="Edit this image: change the background to be darker.",
    image=image,
    generation_mode="image",   

    # ===== VAE-guided prune =====
    umm_vae_guided_prune=True,
    umm_prune_ratio=0.5,       # keep 50% visual tokens
    umm_min_keep_tokens=32,
    umm_latent_pool=2,
    umm_merge_scaling=1.0,
    umm_prune_debug=True,

    # ===== diffusion generation args =====
    generator=torch.Generator(device="cuda").manual_seed(42),
)


edited_img = out.images[0]
edited_img.save("edited.png")
