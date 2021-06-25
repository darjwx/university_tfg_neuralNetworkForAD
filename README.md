# Neural Network models for Autonomous Driving

This project contains different models, from classifiers, to regression models
to directly estimate the steering and speed values, using the NuScenes dataset.

## Models

#### [CNN light](https://github.com/darjwx/university_tfg_neuralNetworkForAD/blob/main/network_classifier_simple.py)
Simple CNN designed to be light. First model developed in this project.
#### [ResNet](https://github.com/darjwx/university_tfg_neuralNetworkForAD/blob/main/network_classifier_resnet.py)
Residual Network or ResNet of 18 layers with Identity blocks.
#### [CNN and LSTM](https://github.com/darjwx/university_tfg_neuralNetworkForAD/blob/main/network_rnn_lstm.py)
The same CNN architecture as CNN light, but with LSTM layers for temporal context.
#### [Regression](https://github.com/darjwx/university_tfg_neuralNetworkForAD/blob/main/network_rnn_lstm_reg.py)
The same CNN architecture as CNN light, but using LSTM layers and estimating steering and speed values.
#### [Aided Regression](https://github.com/darjwx/university_tfg_neuralNetworkForAD/blob/main/network_rnn_lstm_areg.py)
Uses a ResNet18 in the convolutional stage and LSTM layers. Combines regression and classification to improve the results.

## Dependencies
* Python -- 3.6.9
* PyTorch -- 1.7.1
* CUDA -- 11.0
* Numpy -- 1.19.5
* OpenCV -- 4.5.1
* Pandas -- 1.1.5
* tqdm -- 4.56.0
* NuScenes devkit -- 1.1.2
* sklearn -- 0.24.1
* Seaborn -- 0.11.1
* Matplotlib -- 3.3.4

## Usage
Each model has different parameters you can tune to change how they work.
All the commands follow the same structure:

`python3 <name of the model's file> --<conf-1> --<conf-2-> --<conf-3>`

###### Configurations.

Common
* --epochs: Number of epochs.
* --lr: Learning rate.
* --batch: Batch size.
* --res: Resolution of the input images.
* --weights: Class' weights.
* --canbus: Whether to use CAN bus data as input.
* --route: Route to where the dataset is located.
* --tb: Whether to save TensorBoard logs.
* --save: Whether to save the model's state.
* --load: Whether to load a saved model.

LSTM configs
* --hidden: LSTM hidden size.
* --layers: LSTM number of layers.

Regression model
* --coef: Coefficient to calculate the accuracy.

Aided Regression
* --lw: Loss weights
* --predf: Whether to use ground truth or predictions to filter the regression targets.
* --weights_sp: Weights for the speed class.
* --weights_st: Weights for the steering class.
* --video: Whether to build a video with the results.

The following lines will contain a command example for each model.
***

###### CNN light

```
python3 network_classifier_simple.py --route=/data/sets/nuscenes/ --weights 1. 5.68 5.51 --save=models/simple-1.pth --tb=runs/simple
```

###### ResNet18
```
python3 network_classifier_resnet.py --route=/data/sets/nuscenes/ --weights 1. 5.68 5.51 --save=models/resnet-1.pth --tb=runs/resnet
```

###### CNN with LSTM layers
```
python3 network_rnn_lstm.py --route=/data/sets/nuscenes/ --weights 1. 5.68 5.51 --save=models/lstm-class-1.pth --tb=runs/lstm-class
```

###### Regression model
```
python3 network_rnn_lstm_reg.py --route=/data/sets/nuscenes/ --save=models/lstm-reg-1.pth --tb=runs/lstm-reg
```

###### Aided Regression
```
python3 network_rnn_lstm_areg.py  --video=val_info_areg.avi --predf=y --route=/data/sets/nuscenes/ --weights_sp 4. 1. --save=models/areg-1.pth --tb=runs/areg
```
***
