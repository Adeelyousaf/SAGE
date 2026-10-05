# 1) Save the safe-embedding codebook (Stable Diffusion v1.4 text encoder)
python save_codebook.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --csv_path datasets/safe_prompts_copro_sexual.csv \
    --save_dir codebook_copro_sexual_sd14

# 2) Train SAGE (DES losses + ESP + LSA regularizers), paper settings
python train_sage.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --codebook_dir codebook_copro_sexual_sd14 \
    --unsafe_csv_path datasets/unsafe_prompts_copro_sexual.csv \
    --safe_csv_path datasets/safe_prompts_copro_sexual.csv \
    --output_dir checkpoints/sage_copro_sexual \
    --num_epochs 2 \
    --learning_rate 1e-5 \
    --batch_size 128 \
    --lambda_safe 0.3 \
    --save_every 1 \
    --ablation 1 2 3 \
    --concept_prompt "nudity" \
    --concept_guidance_scale 205.0 \
    --local_k 15 \
    --alpha 1.0 \
    --lambda_lsa 0.1 \
    --lambda_esp 2.0 \
    --safe_embedding_path checkpoints/sage_copro_sexual/safe_embeddings.pth
