# Python Scripting

We use the same code as we studied in our last session (Notebook:
[Probing with Linear Classifiers](../notebooks/linear_classifier_probe.ipynb)). We convert the 
code into a Python package named [`deepcsf`](https://github.com/DeepLearning-JupyterBook/deeplearning-jupyterbook.github.io/tree/master/src/deepcsf).

To execute this code, first, we have to activate our virtual environment containing necessary
packages like PyTorch (check the [environment setup tutorial](environment_setup.md)).


Assuming you are already in the *server* directory where the `deepcsf` module is, to train a
network:

    python main.py

And to test the trained network:

    python main.py --test_net <CHECKPOINT_PATH>

The `CHECKPOINT_PATH` is the path to the saved checkpoint in the training script, by default, it's saved
at `csf_out/checkpoint.pth.tar`.

## Python Module

Jupyter Notebook provides an interactive programming environment. This is very useful in several 
scenarios such as:
* prototyping ideas
* exploring data
* plotting results
* demo codes
* etc.

However, training real-world deep networks often consist of a larger magnitude of code which
is difficult to manage in Notebooks. To this end, we should create Python modules and scripts:
* **Python script**: is an executable file that can be executed in the terminal, e.g., 
```python <SCRIPT_PATH>.py```.
* **Python module**: contains function definitions similar to a third-party library or a package.

In this tutorial, we have created a minimal Python package called `deepcsf` following this 
structure:
```
python_script/
└── deepcsf/                # Python package
    └── __init__.py         # __init__.py is required to import the directory as a package
    └── csf_main.py         # training/testing routines
    └── dataloader.py       # dataset-related code
    └── models.py           # the architecture of the network
    └── utils.py            # common utility functions
└── main.py                 # executable script
```

**NOTE**: This tutorial contains a single Python package and a single script, a more complex project 
often contains several packages and scripts.

### Arguments

The [argparse](https://docs.python.org/3/library/argparse.html) module makes it easy to write 
user-friendly command-line interfaces. Our `main.py` module receives several arguments. We can see
the list of arguments by calling:

    python main.py --help

Which outputs:

    usage: main.py [-h] [--epochs EPOCHS] [--initial_epoch INITIAL_EPOCH] [--batch_size BATCH_SIZE]
                   [--train_samples TRAIN_SAMPLES] [--num_workers NUM_WORKERS] [--lr LR] 
                   [--momentum MOMENTUM] [--weight_decay WEIGHT_DECAY] [--out_dir OUT_DIR] 
                   [--test_net TEST_NET] [--resume RESUME]
    
    options:
      -h, --help            show this help message and exit
      --epochs EPOCHS       number of epochs of training
      --initial_epoch INITIAL_EPOCH
                            the staring epoch
      --batch_size BATCH_SIZE
                            size of the batches
      --train_samples TRAIN_SAMPLES
                            Number of train samples at each epoch
      --num_workers NUM_WORKERS
                            Number of CPU workers
      --lr LR               SGD: learning rate
      --momentum MOMENTUM   SGD: momentum
      --weight_decay WEIGHT_DECAY
                            SGD: weight decay
      --out_dir OUT_DIR     the output directory
      --test_net TEST_NET   the path to test network
      --resume RESUME       the path to training checkpoint