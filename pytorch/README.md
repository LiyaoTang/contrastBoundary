
## Installation for Point Transformer

This repository is modified from [point-transformer](https://github.com/POSTECH-CVLab/point-transformer), please find the step-by-step installation guide there.

## Training

To train the point-transformer with CBL, run:

    bash tool/test.sh s3dis origin_multi-Ua-concat-latent_contrast-Ua-softnn-latent-label-l2-w.1

## Testing

Similarly, to test the point-transformer with [pretrained model](https://drive.google.com/drive/folders/1_ppwnrAu6VRqENTPWPt-3KFqCCTtfsFC?usp=sharing), run:

    bash tool/test.sh s3dis origin_multi-Ua-concat-latent_contrast-Ua-softnn-latent-label-l2-w.1

Note: you may need to symlink the `exp` dir and `results` dir, after unzip the pretrained model.
