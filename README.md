# Image classifier

Automatically classifies images using Keras and TensorFlow given example images

## Required

- Python 3
- Internet connection (to download MobileNetV2)

Python modules:
- `tensorflow`, `keras`
- `numpy`
- `Pillow`
- `more_itertools`
- `tqdm`

## Use

```
usage: main.py [-h] training unknown

positional arguments:
  training    directory containing images for training classifier. Images must
              be organized in subdirectories for each class.
  unknown     directory containing images to classify. Subdirectories for each
              class will be automatically generated here.

optional arguments:
  -h, --help  show this help message and exit
```

Example input directory structure to distinguish cats and dogs:

```
Training/
  Cat/
    Cat1.jpg
    Cat2.jpg
  Dog/
    Dog1.jpg
    Dog2.jpg

Unknown/
  Garfield.jpg
  Pluto.jpg
  Felix.jpg
  Snoopy.jpg
```

After running `python main.py Training/ Unknown/` , the `Unknown/` directory will be automatically organized like this:

```
Unknown/
  Cat/
    Felix.jpg
    Garfield.jpg
  Dog/
    Pluto.jpg
    Snoopy.jpg
```

This repository contains a subset of [Kaggle Cats and Dogs Dataset from Microsoft](https://www.microsoft.com/en-us/download/details.aspx?id=54765). To run the example, run `python main.py Training/ Unknown/` and browse the `Unknown/` directory when it's finished.

## Author, license
Copyright (C) 2019 Alan Tseng
License: MIT License

The images in the `Training/` and `Unknown/` directories are from Microsoft and isn't part of this software's license.

