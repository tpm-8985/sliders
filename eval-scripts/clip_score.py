import torch
from PIL import Image
import os
import pandas as pd
import numpy as np
import re
import argparse
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate CLIP weights for retraining')
    parser.add_argument('--im_path', help='生成圖片的資料夾 (例如 results/muscular.pt)', type=str, required=True)
    parser.add_argument('--prompt', help='目標 Prompt (例如 "muscular")', type=str, required=True)
    parser.add_argument('--prompts_path', help='原始 CSV 路徑', type=str, required=True)
    # 加入這一行讓你指定要用哪個 scale 的圖來算權重 (通常選效果最強的，例如 2.0)
    parser.add_argument('--target_scale', help='要用來算分的資料夾名稱 (例如 "2")', type=str, default="2") 
    parser.add_argument('--device', help='cuda device', type=str, default='cuda')
    
    args = parser.parse_args()

    # 1. 初始化
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # 2. 準備路徑
    # 我們只鎖定 "target_scale" 這個資料夾 (例如 '2')，因為我們要用這些圖的表現來決定權重
    im_folder = os.path.join(args.im_path, args.target_scale)
    if not os.path.exists(im_folder):
        print(f"Error: 找不到圖片資料夾 {im_folder}")
        print(f"請確認你的圖片都在 {args.im_path}/{args.target_scale} 裡面")
        exit(1)

    print(f"Calculating weights based on images in: {im_folder}")
    
    # 3. 讀取 CSV
    df = pd.read_csv(args.prompts_path)
    if 'case_number' not in df.columns:
        df['case_number'] = df.index
    
    # 設定索引方便查找
    df.set_index('case_number', inplace=True, drop=False)
    
    # 準備新欄位
    df['raw_clip_score'] = np.nan
    df['clip_weight'] = 1.0 # 預設權重為 1

    # 4. 準備 Prompt 特徵
    target_prompt = args.prompt.strip()
    with torch.no_grad():
        inputs_text = processor(text=[target_prompt], return_tensors="pt", padding=True).to(device)
        text_features = model.get_text_features(**inputs_text)
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

    # 5. 遍歷圖片算分
    images = sorted_nicely([f for f in os.listdir(im_folder) if f.endswith(('.png', '.jpg'))])
    scores_buffer = {}

    print("Scoring images...")
    for image_file in tqdm(images):
        try:
            # 解析檔名 {case}_{idx}.png
            case_str = image_file.split('_')[0].replace('.png', '')
            if not case_str.isdigit(): continue
            case_number = int(case_str)

            if case_number not in df.index: continue

            # 讀圖
            im = Image.open(os.path.join(im_folder, image_file)).convert("RGB")
            inputs_image = processor(images=im, return_tensors="pt").to(device)

            # 算分
            with torch.no_grad():
                image_features = model.get_image_features(**inputs_image)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
                
                # 計算 Cosine Similarity (數值約 0.1 ~ 0.3)
                score = (image_features @ text_features.T).item()
                
            if case_number not in scores_buffer:
                scores_buffer[case_number] = []
            scores_buffer[case_number].append(score)

        except Exception as e:
            print(f"Error: {e}")

    # 6. 填入原始分數
    for case_num, scores in scores_buffer.items():
        df.loc[case_num, 'raw_clip_score'] = np.mean(scores)

    # === 關鍵步驟：計算權重 (Normalization) ===
    # 我們只對有分數的行做處理
    valid_mask = df['raw_clip_score'].notna()
    
    if valid_mask.sum() > 0:
        scores = df.loc[valid_mask, 'raw_clip_score'].values
        mean_score = np.mean(scores)
        std_score = np.std(scores) + 1e-6 # 避免除以 0
        
        print(f"\nStats - Mean: {mean_score:.4f}, Std: {std_score:.4f}")
        
        # Z-Score 正規化：(x - mean) / std
        z_scores = (scores - mean_score) / std_score
        
        # 轉換為權重：
        # 我們希望權重中心在 1.0
        # 並且限制範圍在 0.5 ~ 2.0 之間 (避免權重過大或過小)
        # 這裡設定系數 0.5，代表 1 個標準差的分數差異，會導致權重改變 0.5
        weights = 1.0 + (z_scores * 0.5)
        
        # Clip 限制範圍
        weights = np.clip(weights, 0.1, 3.0)
        
        df.loc[valid_mask, 'clip_weight'] = weights
    else:
        print("Warning: 沒有算出任何分數！")

    # 處理空值 (如果有些 prompt 沒生成圖片，權重設回 1.0)
    df['clip_weight'] = df['clip_weight'].fillna(1.0)

    # 7. 存檔
    # 存成一個新的 CSV，檔名加上 _weighted
    output_filename = args.prompts_path.replace('.csv', '_weighted.csv')
    df.to_csv(output_filename, index=False)
    
    print(f"\n[Success] Training CSV generated: {output_filename}")
    print("欄位 'clip_weight' 已準備好，平均值約為 1.0")
    print(df[['case_number', 'prompt', 'raw_clip_score', 'clip_weight']].head())