# Dissertation Project : Music To Motion

<a href="https://www.youtube.com/watch?v=_GsBnTyi1Q4" title="Music Caster Video Demo">
  <p align="center">
    <img width="75%" src="https://img.youtube.com/vi/_GsBnTyi1Q4/maxresdefault.jpg" alt="Music Caster Video Demo Thumbnail"/>
  </p>
</a>

https://www.youtube.com/watch?v=_GsBnTyi1Q4



This project use the dataset used by the AI choreographer by Google found [here](https://google.github.io/aistplusplus_dataset/index.html)
but implements the models described by [DanceRevolution](https://github.com/stonyhu/DanceRevolution/)

The main goal of the project is to generate Dances by listening to music and the main methods used are :

- Music Encoder with Transformer Architecture
- Dance Decoder with LSTM architecture

To replicate the experiments, follow the next steps :

# 1. Pre-Processing

## MOTION

Let's first download and process the motions representation :

## 1. Download AIST++ Dataset (Annotations)
It consists of a set of Annotations(camera_params, 2d_points, 3d_points and SMPL_params) but we are interested only in the 3D keypoints :

Click [here to download](https://storage.cloud.google.com/aist_plusplus_public/20210308/keypoints3d.zip)


## 2. Extract Audio Features (MFCC, CHROMA, Amplitude Envelope)

The music can be downloaded here :
https://aistdancedb.ongaaccel.jp/database_download/

From the .mp3 files the following musical features are extracted : 20-dim MFCC, 12-dim Chroma, 1-dim hot beats, 1-dim hot peaks, 1-dim envelope.
They are saved in .pkl files 

```
python extract_musical_features.py \
	--audio_dir AUDIO_DIR \
	--output_dir OUT_DIR
```

and finally rename the music as the dances so that it is easier to split the dataset in train and test :

```
python expand_music_features.py \
	--audio_dir AUDIO_DIR \
	--motion_dir MOTION_DIR \
	--output_dir OUT_DIR
```

## 3. Split the dataset

python pre_processing/run_split_dataset.py \
--motion_dir MOTION_DIR \
--audio_dir AUDIO_DIR \
--out_dir OUT_DIR

## 4. TRAIN

```
python train.py --data_dir ../small_dataset/ \
--d_model 240 \
--n_layers 2 \
--n_heads 8 \
--inner_d 1024 \
--learning_rate 0.0001 \
--lambda_v 0.01 \
--dropout 0.1 \
--max_epochs 4000 
```

## 5. INFER

```
python inference.py \
    --test_dir TEST_DIR \
    --model MODEL
```
