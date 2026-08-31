# Coin Detection and Recognition

A classical computer-vision pipeline that combines circle detection and appearance-based classification for Turkish coins.

## Highlights

- Canny edge detection
- Custom Hough circle voting
- Non-maximum suppression for circle candidates
- Coin crop normalization
- Histogram of Oriented Gradients (HOG) implemented with NumPy/OpenCV primitives
- Linear SVM classification
- Support for obverse/reverse coin classes
- CLI-oriented, portable project structure

## Project structure

```text
coin-detection-recognition/
├── src/
│   └── coin_pipeline.py
├── requirements.txt
└── README.md
```

## Dataset layout

Classification expects class names at the beginning of each image filename, for example:

```text
data/train/
├── 1TL_obverse_001.jpg
├── 1TL_reverse_001.jpg
├── 50kr_obverse_001.jpg
└── ...
```

The original experiment used 12 labels covering six denominations and both coin sides. The dataset is not included in the repository.

## Train and evaluate the classifier

```bash
python src/coin_pipeline.py classify --train data/train --test data/test
```

## Detect circles in one image

```bash
python src/coin_pipeline.py detect --image path/to/coins.jpg --output outputs/detected.jpg --min-radius 25 --max-radius 120 --threshold 45
```

The custom Hough implementation is intentionally educational and can be slow on large images. Resize high-resolution inputs before detection when needed.
