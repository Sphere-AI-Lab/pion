# convert the checkpoint to huggingface format for torch_dist
CKPT=
python tools/checkpoint/convert_to_hf.py \
    --input_dir "$CKPT" \
    --output_dir "${CKPT}_hf" \
    --num_attention_heads 32 \
    --num_key_value_heads 32

# convert the checkpoint to huggingface format for torch

CKPT=
CKPT_DIR=$(dirname "$CKPT")
python tools/checkpoint/convert_rng_to_hf.py \
    --checkpoint_file "$CKPT" \
    --output_dir "${CKPT_DIR}/hf_rng" \
    --num_attention_heads 32 \
    --num_key_value_heads 32
echo "Done!"
