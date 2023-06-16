# Python Scripting

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

In this tutorial, we will create a minimal Python package from the same code as we studied in our 
last session (Notebook: [Probing with Linear Classifiers](../notebooks/linear_classifier_probe.ipynb)).
This example measures the contrast sensitivity function (CSF) of deep neural networks, therefore,
we have named the package as 
[`deepcsf`](https://github.com/DeepLearning-JupyterBook/deeplearning-jupyterbook.github.io/tree/master/src/).


## Python Package

The `deepcsf` Python package consists of the following structure:
```
src/
├── deepcsf/                # Python package
│   ├── __init__.py         # __init__.py is required to import the directory as a package
│   ├── csf_main.py         # training/testing routines
│   ├── dataloader.py       # dataset-related code
│   ├── models.py           # the architecture of the network
│   └── utils.py            # common utility functions
└── main.py                 # executable script
```

Essentially, we have split the code in our notebook into several Python modules each containing 
a particular functionality.

```{admonition} Nested packages
:class: note
This tutorial contains a single Python package and a single script, a more complex project 
often contains several packages and scripts. This is an easy process: split out the functionality 
you want into separate folders and include an empty \_\_init\_\_.py file.
```

### Execution

To execute this code, first, we have to activate our virtual environment containing necessary
packages like PyTorch (check the [environment setup tutorial](environment_setup.md)).

In your terminal, navigate to the *src* directory where the `deepcsf` package is. To train a
network:

    python main.py

And to test the trained network:

    python main.py --test_net <CHECKPOINT_PATH>

The `CHECKPOINT_PATH` is the path to the saved checkpoint in the training script, by default, it's saved
at `csf_out/checkpoint.pth.tar`.

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

In order to pass an argument to our script, we first specify the argument name followed by its value
. Similarly, to pass several arguments we separate them by an empty space, for example:

    python main.py --batch_size 32 --epochs 10

Specifies a `batch_size` of 32 and 10 `epochs`.

Adding arguments to your script is very easy, for instance:

``` python
# make an instance of ArgumentParser
parser = argparse.ArgumentParser()
# The add_argument() method attaches individual argument specifications to the parser.
parser.add_argument("--epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
# The parse_args() method runs the parser and places the extracted data in an argparse.Namespace object.
args = parser.parse_args()
```


```{admonition} Make use of the full potential of argparse
:class: hint
The argparse module offers several useful features, including:
* Type of argument (e.g., string, float, boolean, etc.)
* Whether an argument is optional or required
* Limiting the list of values to predefined choices
* etc.

An explanation of these features goes beyond the scope of this tutorial. Please check the official
[argparse documentation](https://docs.python.org/3/library/argparse.html).
```

## Logging

The core functionality of our Python script `deepcsf` is identical to its corresponding Jupyter 
notebook. However, we have added a few functionalities in `csf_main.py` to save/load models and 
log the progress, which we go through it in this section.

### Dumping arguments
We store the value of all variables in `argparse.Namespace` in a JSON file. This is handy in several
scenarios, for instance when running multiple experiments using the same code with different 
parameters.

``` python
def save_arguments(args):
    """Dumping all arguments in a JSON file"""
    json_file_name = os.path.join(args.out_dir, 'args.json')
    with open(json_file_name, 'w') as fp:
        json.dump(dict(args._get_kwargs()), fp, sort_keys=True, indent=4)
```

### Saving Checkpoints

We should save the weights of our network and parameters for optimiser frequently (e.g., at the end 
of each epoch):
1. To resume training,
2. To test the network with new stimuli.

Often a `dict` is stored containing **all** variables that are required to load a network/optimiser 
again. In our example:
``` python
utils.save_checkpoint(
    {
        # to know to which epoch this checkpoint belongs
        'epoch': epoch,
        # related variables to create the network and load its weights
        'network': {
            'arch': arch,
            'layer': layer,
            # state_dict() contains the network's weights
            'state_dict': network.state_dict()
        },
        # to normalise input signal correctly
        'preprocessing': {'mean': args.mean, 'std': args.std},
        # parameters of optimiser are required to resume training
        'optimizer': optimizer.state_dict(),
        # to input network with a correct input size
        'target_size': args.target_size,
    },
    # the directory where the checkpoint is saved
    args.out_dir
)
```

```{admonition} Checkpoints should be complete!
:class: important
Double-check that you are saving all the necessary parameters/variables before starting a long 
training process. It would be very painful not to be able to use a network trained for several days!
```

### Resuming training

Resuming training from a given checkpoint is a desirable feature (e.g., because the training process 
was interrupted, or because you want to obtain better performance). When you resume the training 
from a checkpoint, you should load all the necessary variables from the checkpoint. For example in our code:

``` python
# if resuming a previously training process
if args.resume is not None:
    # openning the checkpoint file
    checkpoint = torch.load(args.resume, map_location='cpu')
    # loading the network with the weights from the checkpoint
    network.load_state_dict(checkpoint['network']['state_dict'])
    # setting the epoch to the checkpoint epoch
    args.initial_epoch = checkpoint['epoch'] + 1
    # loading the optimiser parameters from the checkpoint
    optimizer.load_state_dict(checkpoint['optimizer'])
```

### Testing a network

To test a network we only need to load the weights of the network and several other stored variables 
such as the optimiser state is irrelevant. From our example:

``` python
checkpoint = torch.load(args.test_net, map_location='cpu')
network.load_state_dict(checkpoint['network']['state_dict'])
network = network.to(args.device)
network.eval()
```