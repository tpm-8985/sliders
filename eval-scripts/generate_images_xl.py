#%%writefile eval-scripts/generate_images_colab.py
import torch
from PIL import Image
import argparse
import os, json, random
import pandas as pd
import sys
import gc
import copy

# 1. 修正路徑與載入
sys.path.append(os.getcwd())
try:
    # 直接匯入模組本身，方便我們修改全域變數
    import trainscripts.textsliders.lora as lora_module
    from trainscripts.textsliders.lora import LoRANetwork, DEFAULT_TARGET_REPLACE, UNET_TARGET_REPLACE_MODULE_CONV
except ImportError:
    print("[Warning] 找不到 'trainscripts'，請確認你在 sliders 資料夾內。")
    pass

from safetensors.torch import load_file
from diffusers import StableDiffusionXLPipeline

# 自動尋找 Output Class
try:
    from diffusers import StableDiffusionXLPipelineOutput
except ImportError:
    try:
        from diffusers.pipelines.stable_diffusion_xl import StableDiffusionXLPipelineOutput
    except ImportError:
        from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

def flush():
    torch.cuda.empty_cache()
    gc.collect()

# --- [關鍵修復 1] 手動覆寫 call 函數，徹底避開 diffusers 的 bug ---
@torch.no_grad()
def call(self, prompt=None, num_inference_steps=50, guidance_scale=5.0, 
         network=None, start_noise=None, scale=None, generator=None, **kwargs):
    
    height = kwargs.get('height', 1024)
    width = kwargs.get('width', 1024)
    
    (prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.encode_prompt(
        prompt=prompt, device=self.device, do_classifier_free_guidance=True
    )
    
    self.scheduler.set_timesteps(num_inference_steps, device=self.device)
    timesteps = self.scheduler.timesteps
    
    latents = self.prepare_latents(
        1, self.unet.config.in_channels, height, width, prompt_embeds.dtype, self.device, generator, None
    )

    add_text_embeds = pooled_prompt_embeds
    
    # [FIX] 手動建構 Time IDs，不呼叫 self._get_add_time_ids
    # SDXL 需要 6 個數值: (original_h, original_w, crop_top, crop_left, target_h, target_w)
    # 我們這裡模擬 1024x1024 的標準輸入
    original_size = (1024, 1024)
    crops_coords_top_left = (0, 0)
    target_size = (1024, 1024)
    
    # 拼接成 list
    add_time_ids_list = list(original_size + crops_coords_top_left + target_size)
    # 轉成 Tensor
    add_time_ids = torch.tensor([add_time_ids_list], dtype=prompt_embeds.dtype).to(self.device)
    
    if True: 
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(self.device)
    add_text_embeds = add_text_embeds.to(self.device)
    # add_time_ids 已經在 device 上了，只需要 repeat
    add_time_ids = add_time_ids.repeat(1, 1)

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if network is not None:
                if t > start_noise:
                    network.set_lora_slider(scale=0)
                else:
                    network.set_lora_slider(scale=scale)
            
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            
            context = network if network else torch.no_grad()
            with context:
                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=prompt_embeds, 
                    added_cond_kwargs=added_cond_kwargs, return_dict=False
                )[0]

            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            progress_bar.update()

    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
    image = self.image_processor.postprocess(image, output_type="pil")
    return StableDiffusionXLPipelineOutput(images=image)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--prompts_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--scales', type=str, default="0,2")
    parser.add_argument('--start_noise', type=int, default=750)
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()
    
    scales = [float(x) for x in args.scales.split(',')]
    device = args.device if torch.cuda.is_available() else "cpu"
    
    # Apply Patch
    StableDiffusionXLPipeline.__call__ = call
    
    print(f"Loading SDXL Pipeline on {device}...")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        'stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16
    ).to(device)
    
    print(f"Loading Slider: {args.model_name}")
    
    # --- [關鍵修復 2] 強制注入卷積層定義 (解決 Unexpected keys 問題) ---
    # 因為 __init__ 不接受參數，我們直接修改它依賴的全域變數
    # 這招很暴力，但對這種 Research Code 非常有效
    print("Injecting Conv modules into default config...")
    lora_module.DEFAULT_TARGET_REPLACE = DEFAULT_TARGET_REPLACE + UNET_TARGET_REPLACE_MODULE_CONV
    
    # 初始化網路 (現在它會自動吃到 Conv 層了)
    network = LoRANetwork(
        pipe.unet, rank=4, multiplier=1.0, alpha=1.0, 
        train_method='noxattn', 
    ).to(device, dtype=torch.float16)
    
    try:
        if args.model_name.endswith('.safetensors'):
            state_dict = load_file(args.model_name)
        else:
            state_dict = torch.load(args.model_name, map_location=device)
        
        # 嘗試嚴格載入 (如果成功，代表所有權重都對上了)
        network.load_state_dict(state_dict, strict=True)
        print("Successfully loaded slider weights (Strict Mode)!")
        
    except Exception as e:
        print(f"Standard loading failed: {e}")
        print("Attempting lenient loading...")
        keys = network.load_state_dict(state_dict, strict=False)
        # 如果 Unexpected keys 還是很多，代表注入失敗，但至少不會崩潰
        print(f"Loaded with strict=False. Missing: {len(keys.missing_keys)}, Unexpected: {len(keys.unexpected_keys)}")
        
    try:
        df = pd.read_csv(args.prompts_path)
        prompts = df['prompt'].tolist()
        if 'evaluation_seed' in df.columns:
            seeds = df['evaluation_seed'].tolist()
        else:
            seeds = [random.randint(0, 100000) for _ in range(len(prompts))]
    except Exception as e:
        print(f"Error reading CSV: {e}")
        sys.exit(1)

    print(f"Target Scales: {scales}")
    
    for idx, prompt in enumerate(prompts):
        seed = int(seeds[idx])
        if idx % 5 == 0: print(f"Processing [{idx}/{len(prompts)}]...")

        for scale in scales:
            model_basename = os.path.basename(args.model_name)
            out_dir = os.path.join(args.save_path, model_basename, str(scale).replace('.0', ''))
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, f"{idx}.png")
            
            if os.path.exists(out_file): continue
            
            generator = torch.manual_seed(seed)
            image = pipe(
                prompt=prompt, num_inference_steps=50, generator=generator, 
                network=network, start_noise=args.start_noise, scale=scale
            ).images[0]
            image.save(out_file)
            
    print("\nDone!")
    flush()