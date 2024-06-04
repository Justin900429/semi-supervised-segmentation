# README

## Introduction

This repo provides an example code of performing semi-supervised semantic segmentation with [FixMatch](https://github.com/kekmodel/FixMatch-pytorch). As this repo is made for Data Science courses at NYCU, the data is provided by ourselves.

## Installation

>[!Tip]
> PyTorch is provided in the `requirements.txt` with a CUDA 12.x version. Please choose the appropriate version if you have a special requirement.

```shell
pip install -r requirements.txt
```

## Dataset Setup

Please download the dataset from [here](https://drive.google.com/file/d/1seiemd2silpWHIfbRDVaEhOfkh2rk2G6/view?usp=drive_link).

```shell
gdown 1seiemd2silpWHIfbRDVaEhOfkh2rk2G6
unzip -q 2024_ds_hw_6.zip

# (Optional) 
#  Check the MD5 of the dataset if you want :)
#  It should be b2a77d0a092646a5db1508a80cfa892f 
md5sum 2024_ds_hw_6.zip
```

The structure of the dataset is shown below:

```plaintext
📦 Data Root
┣ 📂 images
┃ ┣  🖼️ 1.jpg       # Training image
┃ ┗  …
┣  📂 segment
┃ ┣  🖼️ 1_color.png # Visualization
┃ ┣  🖼️ 1_mask.png  # Labeling data
┗ ┗ …
```

## Training

To train the model with SSL, please execute:

```shell
# single gpu
python train.py --config configs/default.yaml
# multiple-gpu
accelerate launch --multi_gpu --num_processes={NUM_GPU} train.py --config configs/default.yaml
```

If you want to train the model without SSL, please execute:

```shell
# single gpu
python train.py --config configs/default.yaml --opts TRAIN.USE_SSL False
# multiple-gpu
accelerate launch --multi_gpu --num_processes={NUM_GPU} train.py --config configs/default.yaml --opts TRAIN.USE_SSL False
```

Users can open [aim](https://github.com/aimhubio/aim) to check the training process by:

```shell
aim up
```

## Inference

To perform inference, please execute:

```shell
# With SSL
python train.py --config configs/default.yaml --save-path {PLACE_TO_SAVE} --test --opts MODEL.CHECKPOINT {PATH_TO_CHECKPOINT}

# Without SSL
python train.py --config configs/default.yaml  --save-path {PLACE_TO_SAVE} --test --opts MODEL.CHECKPOINT {PATH_TO_CHECKPOINT} TRAIN.USE_SSL False

# For example:
python train.py --config configs/default.yaml  --save-path deep_learning_is_fun --test --opts MODEL.CHECKPOINT best.pth
```

If the users do not specify the `save-path`, it will save the result in the `prediction` folder.

## Evaluation

Once you have the prediction result, you can evaluate the result by:

```shell
python evaluate.py --gt {PLACE_TO_GROUND_TRUTH} --pred {PLACE_TO_PREDICTION} 

# For example:
python evaluate.py --gt private_gt --pred prediction
```

>[!Tip]
>
> * The ground truth folder should have the same structure as the prediction folder.
> * We don't provide the ground truth in this repo, you can create it with the validation set.
> * You can have more files in the ground truth folder than the prediction folder, but you can't have more files in the prediction folder than the ground truth folder. The evaluation will only consider the files that exist in prediction folder.

The structure of the two folders is shown below:

```plaintext
📦 gt
┃ 🖼️ 1_mask.png
┗ …
📦 pred
┃ 🖼️ 1_mask.png
┗ …
```

## Submission

Please submit the prediction folder to the [Kaggle](https://www.kaggle.com/c/2024-ds-hw-6/submit). As kaggle only accept the csv file, you can use the following command to convert the prediction to the csv file:

```shell
python create_submission.py --pred {PLACE_TO_PREDICTION} --save-file {SAVED_FILE_NAME}
```

If the users do not specify the `save-file`, it will save the result in the `submission.csv`.

## Project Template

Please check [deep-learning-template](https://github.com/Justin900429/deep-learning-template) for more information.
