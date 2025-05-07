# Data Poisoning Attack Defense and Evolutionary Domain Adaptation for Federated Medical Image Segmentation


## Environment

- The code is developed with CUDA 12.2, ***Python >= 3.10.0***, ***PyTorch >= 2.0.0***

    0. [Optional but recommended] create a new conda environment.
        ```
        conda create -n fedpure python=3.10.0
        ```
        And activate the environment.
        ```
        conda activate fedpure
        ```

    1. Install the requirements
        ```
        pip install -r requirements.txt
        ```



Clients 1, 2, 3, and 5 consist of the M&Ms dataset.

Clients 4, 6, 7, and 8 consist of the ACDC dataset.



The commands are as follows.

```
# Training for AdaShield-FL in Federated Learning

python federated_train.py


# Testing for AdaShield-FL in Federated Learning

python test.py

```

For detailed parameter adjustments, refer to exp_DRS.py in the experiments directory.


