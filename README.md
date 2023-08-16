### GAG: Gripper-Aware Grasp refinement

First example of grasp refinement based on gripper configuration and pose.
Note how the gripper moves upwards to avoid collision with the ball,
it opens wider, and it rotates slightly to avoid collisions with the box.

<img src="media/refinement_example.png" width="600"/>

### todo

- perhaps even merge the Ferrari Canny thing into the BURG toolkit
  - it is pretty much encapsulated anyway

## Installation

Use [anaconda](https://www.anaconda.com/), create an environment called `conv_onet` using
```
conda env create -f convonets/environment.yaml
conda activate conv_onet
```
**Note**: you might need to (just do it) install **torch-scatter** manually following [the official instruction](https://github.com/rusty1s/pytorch_scatter#pytorch-140):
```
pip uninstall torch-scatter
pip install torch-scatter==2.0.4 -f https://pytorch-geometric.com/whl/torch-1.4.0+cu101.html
```

Next, compile the extension modules and then install the package.
You can do this via
```
python setup.py build_ext --inplace
pip install -e .
```

This will also install the BURG Toolkit, which is required for dataset creation.
And perhaps later on for some visualisations.