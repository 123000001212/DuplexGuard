# DuplexGuard: Safeguarding Deletion Right in Machine Unlearning via Duplex Watermarking
This repo includes the pytorch implementation of DuplexGuard.
## Environmental setup
The recommended python version is 3.8.16.

Make sure you have a pytorch environment with all packages in `requirements.txt`. 

Or just create a new environment and run `pip install -r requirements.txt`.

## Run the code
### Quick Start
Please refer to `exp.ipynb` for our example run on CIFAR-10 with ResNet-18.

### Craft $D_a$ and $D_s$
```
python main-SA.py --net="ResNet18" --dataset="CIFAR10" --data_path="/home/data" --eps=8 --patch_size=12
```
### Verify the unlearned model
```
python eval_p-value.py --net="ResNet18" --dataset="CIFAR10" --data_path="/home/data" --eps=8 --patch_size=12
```

## Acknowledgments
We would like to thank [*Sleeper Agent Repository*](https://github.com/hsouri/Sleeper-Agent) as their code helped us a lot.

## Citations
To be updated...