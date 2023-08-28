from adapter import T2IAdapter
from pipeline_xl_adapter import StableDiffusionXLAdapterPipeline
import torch
# from controlnet_aux import SamDetector
from PIL import Image
# import urllib
from torchvision import transforms

import cv2
import random
import numpy as np

resolution = 1024
n_steps = 50 # 40
high_noise_frac = 0.8
seed = -1

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

dtype = torch.float16
device = "cuda"

print(r'loading adapter......')
# SargeZT/t2i-adapter-sdxl-segmentation
adapter = T2IAdapter.from_pretrained("/mnt/nfs/file_server/public/mingjiahui/models/SargeZT--t2i-adapter-sdxl-multi/segmentation/").to(
    dtype=dtype, device=device
)
print(f'loading sdxl......')
base = StableDiffusionXLAdapterPipeline.from_pretrained(
    "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/",
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    adapter=adapter,
).to(device)

# print(f'loading sdxl refiner......')
# from diffusers import DiffusionPipeline
# refiner = DiffusionPipeline.from_pretrained(
#     "/mnt/nfs/file_server/public/mingjiahui/models/stabilityai--stable-diffusion-xl-refiner-1.0",
#     # text_encoder_2=base.text_encoder_2,
#     # vae=base.vae,
#     torch_dtype=torch.float16,
#     use_safetensors=True,
#     variant="fp16",
# )

print(f'loading sdxl refiner......')
from diffusers import StableDiffusionPipeline
refiner = StableDiffusionPipeline.from_single_file(
    '/mnt/nfs/file_server/public/mingjiahui/models/stabilityai--stable-diffusion-xl-refiner-1.0/sd_xl_refiner_1.0.safetensors',
    torch_dtype=torch.float16,
    variant="fp16",
)
refiner.to("cuda")

# sam = SamDetector.from_pretrained("ybelkada/segment-anything", subfolder="checkpoints")
# #%%
# input_url = "https://ca-times.brightspotcdn.com/dims4/default/167bfeb/2147483647/strip/true/crop/2400x1600+0+0/resize/1200x800!/quality/80/?url=https%3A%2F%2Fcalifornia-times-brightspot.s3.amazonaws.com%2Fb9%2F06%2F58ccf43e4162bd2115c662cc087e%2Fla-photos-1staff-692345-me-0120-heightened-alert-dtla12.IK.JPG"
# with urllib.request.urlopen(input_url) as url:
#     with open("input.jpg", "wb") as f:
#         f.write(url.read())
# input_image = Image.open("input.jpg")
#
# new_size = (resolution, resolution)
# input_image = input_image.resize(new_size)

# # Segment the image
# preprocessed_image = sam(input_image)
# display(preprocessed_image)
depth_path = r'/home/mingjiahui/data/demo_mask.png'
preprocessed_image = cv2.imread(depth_path)
cv2.imwrite('/home/mingjiahui/data/2_depth.jpg', preprocessed_image)

# # new add load data
# print(f'prepare input.....')
# img_path = r'/mnt/nfs/file_server/public/mingjiahui/data/inference_test/000000000285.jpg'       # 632
# depth_path = img_path.replace('.jpg', '.depth.png')
# input_image = cv2.imread(img_path)
# preprocessed_image = cv2.imread(depth_path)
# cv2.imwrite('/home/mingjiahui/data/2.jpg', input_image)
# cv2.imwrite('/home/mingjiahui/data/2_depth.jpg', preprocessed_image)


transform = transforms.Compose(
    [
        transforms.Resize(
            resolution,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),
        transforms.CenterCrop(resolution),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(dtype=dtype),
    ]
)

preprocessed_image = Image.fromarray(preprocessed_image).convert('RGB')
preprocessed_image_t = transform(preprocessed_image).unsqueeze(0)

prompt = "An zombie riding a unicorn in New York City"
# txt_path = img_path.replace('.jpg', '.txt')
# with open(txt_path, 'r')as file:
#     prompt = file.readline().strip()

print(f'start inference ......')
base_output = base(
    prompt=prompt,
    height=resolution,
    width=resolution,
    num_inference_steps=n_steps,
    denoising_end=high_noise_frac,
    output_type="latent",
    image=preprocessed_image_t,
    guidance_scale=9.0,
).images

image = refiner(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_start=high_noise_frac,
    image=base_output,
).images[0]
# display(image)

# w refiner
image.save('/home/mingjiahui/data/2_res_refiner.jpg')

# w/o refiner
from basicsr.utils import tensor2img
from diffusers import AutoencoderKL
print(f'loading vae')
vae = AutoencoderKL.from_pretrained(
                "/mnt/nfs/file_server/public/lipengxiang/sdxl_1_0/",
                subfolder="vae",
                revision=None,
            )
res = vae.decode(base_output.to(torch.float32).cpu() / vae.config.scaling_factor, return_dict=False)[0]
res = (res / 2 + 0.5).clamp(0, 1)
res = tensor2img(res)

print(type(res))
print(res.shape)
cv2.imwrite('/home/mingjiahui/data/2_res_base.jpg', res)
