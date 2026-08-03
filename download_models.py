from huggingface_hub import hf_hub_download
import joblib

base_repo_id = 'stabilityai/stable-diffusion-3.5-large'

files = [
    'sd3.5_large.safetensors',
    'text_encoders/clip_l.safetensors',
    'text_encoders/clip_g.safetensors',
    'text_encoders/t5xxl_fp16.safetensors'
]

for file in files:
    print('downloading ', file)
    hf_hub_download(base_repo_id, file, local_dir="models")

print('downloading depth controlnet')
hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_depth.safetensors", local_dir="models")


