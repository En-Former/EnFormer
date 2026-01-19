PORT=29511 \
python visualize.py \
--image ../assets/lighthouse.jpg \
--shape 224 \
--model enformer_small \
--stage 3 \
--head 0 \
--checkpoint enformer_small.pth.tar \
--alpha 0.5 \
--map_type assignment \
--part_method forward_fast \
--center_type peak \
--output_dir ../assets/vis \
--interpolation bilinear \
# --no_draw_center
