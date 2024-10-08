# DCT2net: An Interpretable Shallow CNN for Image Denoising (IEEE TIP'22)
Sébastien Herbreteau and Charles Kervrann

## Requirements

The repo supports python 3.8 + pytorch 1.8.1 + numpy 1.21.2 + skimage 0.19.2 + PIL 8.2.0.

## Install

To install in an environment using pip:

```
python -m venv .dct2net_env
source .dct2net_env/bin/activate
pip install /path/to/DCT2net
```

## Datasets
All the models are trained on BSD400 and tested on BSD68 and Set12. Simply modify the argument ``--in_folder`` for training on other datatets. Only one model is trained for all noise levels between 1 and 55.

## Results

### Gray denoising
(Training on BSD400)

| Model   | Params | sigma=15 | sigma=25 | sigma=50 |
|---------|:------:|:-------:|:--------:|:--------:|
| DnCNN   |   556k          |   31.72       |    29.23      |    26.23      |
| BM3D   |   -    |   31.07       |    28.57      |    25.62      |
| **DCT2net** |  29k    |   31.09       |  28.64       |   25.68       |


## Run the Code

To train a new model for gray denoising:
```
python ./trainer.py --in_folder /path/to/dataset
```

## Pretrained model

To denoise an image with DCT2net (remove ``--add_noise`` if it is already noisy):
```
python ./dct2net_denoiser.py --sigma 25 --add_noise --in ./test_images/102061.png --out ./denoised.png --model_name ./saved_models/dct2net.p
```

To denoise an image with DCT/DCT2net (remove ``--add_noise`` if it is already noisy):
```
python ./dct-dct2net_denoiser.py --sigma 25 --add_noise --in ./test_images/102061.png --out ./denoised.png --model_name ./saved_models/dct2net.p
```


## Acknowledgements

This work was supported by Bpifrance agency (funding) through the LiChIE contract. Computations  were performed on the Inria Rennes computing grid facilities partly funded by France-BioImaging infrastructure (French National Research Agency - ANR-10-INBS-04-07, “Investments for the future”).

We would like to thank R. Fraisse (Airbus) for fruitful  discussions. 

## Citation
```BibTex
@ARTICLE{9799727,
  author={Herbreteau, Sébastien and Kervrann, Charles},
  journal={IEEE Transactions on Image Processing}, 
  title={DCT2net: An Interpretable Shallow CNN for Image Denoising}, 
  year={2022},
  volume={31},
  number={},
  pages={4292-4305},
  doi={10.1109/TIP.2022.3181488}
}
```
