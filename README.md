# Dissertation Project : Music To Motion
This project use the dataset used by the AI choreographer by Google found [here](https://google.github.io/aistplusplus_dataset/index.html)
but implements the models described by [Aud2Repr2Pose](https://github.com/GestureGeneration/Speech_driven_gesture_generation_with_autoencoder)

The main goal of the project is to generate Dances by listening to music and the main methods used are :

- Representation Learning used to encode the motions into lower dimensional vector
- Recurrent neural networks used to map music to the encodes motions

To replicate the experiments, follow the next steps :

# 1. Pre-Processing

## MOTION

Let's first download and process the motions representation :

### 1. Download AIST++ Dataset (Annotations)
It consists of a set of Annotations(camera_params, 2d_points, 3d_points and SMPL_params) but we are interested only in the 3D keypoints :

Click [here to download](https://storage.cloud.google.com/aist_plusplus_public/20210308/keypoints3d.zip)

### 2. Extract and Normalise
The motions provided are in COCO-format saved as `.pkl` files. 

We can extract the optimized keypoints animations (`keypoints3d_optim`)  and normalise them by running :
```
python run_estimate_translation.py \
 --keypoints3d_path ANNO_DIR \
 --normalised3d_path OUT_DIR
```

## MUSIC

There is no link to download the music files, but videos are provided thus we extract from videos.

### 1. Download AIST++ Dataset (Videos)
Follow the instructions from the [AIST++ website](https://google.github.io/aistplusplus_dataset/download.html)

### 2. Convert to .mp3
```
python run_extraction_to_mp3.py \
	--videos_dir VIDEO_DIR \
	--save_dir OUT_DIR
```

### 3. Extract Audio Features (MFCC, CHROMA, Amplitude Envelope)
From the .mp3 files the following musical features are extracted : 20-dim MFCC, 12-dim Chroma, 1-dim hot beats, 1-dim hot peaks, 1-dim envelope.
They are saved in .pkl files 

```
python extract_musical_features.py \
	--audio_dir AUDIO_DIR \
	--output_dir OUT_DIR
```

# 2. TRAIN

## Representation Learning
First we want to apply RL to the input motion vector

### 1. Train the Autoencoder 
```
python train_DAE.py
```

###  2. Encode all Dataset 
We use a simple Fully Connected <b>AutoEncoder</b> to reduce the dimension of the motion representation :
<br/>
-<b> Input :</b> motions(1.2)  + add_noise  (54-dim)
<br/>
-<b>Output (Latent Space) :</b> encoded motion (27-dim)
<br/>

```
python learn_dataset_encoding.py \
 --model_parameters PTH_FILE \
 —-to_encode_dir DATA_DIR \
 --encoded_dir OUT_DIR 
```

## Recurrent Neural Network (Music to Motion)

### 1. Train
# dance_revolution
# dance_revolution
# dance_revolution
