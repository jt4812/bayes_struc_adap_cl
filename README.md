## Bayesian Continual Learning with Adaptive Network Depth and Width

### Supervised Continual Learning
#### Fully Connected Experiments
The hyperparameter settings for different experiments are provided in `hyper_params` folder. You can load and run the experiments by running the following commands from the project directory. 
```shell
python -m src.run_cl --file <split_fashion_mnist/split_mnist/>.json --experiment_name <experiment_name>
```


#### CNN Experiment with AlexNet Architecture
You can load and run the experiments by running the following commands from the project directory. 
```shell
python -m src.run_cl --file <split_cifar10/split_cifar100_10/split_tiny_imagenet_10>.json --experiment_name <experiment_name>
```

#### CNN Experiment with FullyConvolutional Architecture

For experiments with fully convolutional network with truncation K=24, you can run the following command.
```shell
python -m src.run_cl --file <split_cifar10/split_cifar100_10/split_tiny_imagenet_10>_fullyconv.json --architecture alexnet_long --n_conv_layers 24 --experiment_name <experiment_name>
```
### Unsupervised Continual Learning
1. Run the following command for training and evaluation of one-MNIST VAE experiment.

```shell
python -m src.vae.run --train --eval_ll
```
2. Run the following command for training and evaluation of not-MNIST VAE experiment.
```shell
python -m src.vae.run_notmnist --train --eval_ll
```