# SAGE on Stable Diffusion v1.4
# (SAGE checkpoints share the DES checkpoint format; --training_method des loads --text_encoder_path)
python generate.py \
    --model_path CompVis/stable-diffusion-v1-4 \
    --device cuda:0 \
    --prompts_csv "datasets/coco_prompts.csv" \
    --output_path "t2i_coco" \
    --start_idx 0 \
    --end_idx 128 \
    --training_method des \
    --text_encoder_path "checkpoints/sage_copro_sexual/1e-05_0.3/checkpoint-2.pt"
