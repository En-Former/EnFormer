PORT=29511 \
python3 validate.py \
/data/imagenet1k \
--model enformer_small \
-b 1024 \
--checkpoint enformer_small.pth.tar \
--amp
