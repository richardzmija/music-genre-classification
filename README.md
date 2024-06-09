# 🎵 Classification of music into genres 🎵
This project explores the possiblity of using various machine
learning techniques for classifying pieces of music into music
genres based on provided sample audio files.

## Dataset used
To train models for this classification task I have used the [GTZAN Dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) both for audio files and choice of features to extract from these files. It provides audio files from 10 music genres and is well suited to approaches using neural networks as well as the conventional machine learning algorithms.

## Models considered
I have trained and compared two models: one using Gradient Boosting Machines and the other using Random Forests. They both achieved a very high accuracy of 88% for Random Forest and 92% for Gradient Boosting Machines using XGBoost. To achieve such accuracy with Random Forest I had to use a lot of trees together with considerable max depth which led to the model weighing 10x more than the Gradient Boosting Machine model created using XGBoost. Also the training process of GBM was much quicker. This led me to using GBM model for the classifier.

## Functionality implemented
The classifier is controlled by a GUI which loads the files from the local file system and allows the user to communiate with the classifier. The functionality can be divided into two groups:
- file loading and playback
- prediction, probability estimation, prediction history

## Conclusions
The problem of classifying music files into genres automatically, long considered very difficult, has seen remarkable progress with the advent of modern machine learning techniques. It is important to note that music genres can be subjective and vary between cultures. Some songs can also blend multiple genres and therefore the division into a neat groups might be difficult, if not impossible. The trained model shows however that using advanced signal processing techniques, clever selection of features and high-quality datasets we may use supervised learning techniques for teaching an agent to recognize music genres even if the division is not always obvious or precisely stated.
