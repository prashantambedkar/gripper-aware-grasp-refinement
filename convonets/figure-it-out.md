### configs

The config files inherit from each other. 
In case of `my_demo.yaml` it is loading more general cfgs first:
- default.yaml
  - room_combine.yaml (both 3-planes and volumetric)
    - my_demo.yaml

Each method has a config as well, however in this repo there is only `conv_onet`
method and no other methods, which makes the code less readable.

### terms and definitions for data loading

- **fields:**
  - generally holds stuff relating to the data
  - initialised with some output of the model, so annotations I 
  presume?
  - In test mode with my_demo, no fields are loaded.
  - However, later it is populated with input_fields and some idx field
- **inputs_field:**
  - this is basically an object of class that corresponds to input data
    type, there is one class for each input_type.
  - see `src/config.py:get_inputs_field()` and `src/data/fields.py`
- **input_type:**
  - i can change it to **partial_pointcloud**
    - it will automatically crop sth off of the input pc
    - produces only a partial reconstruction though!!!
    - the cropped away stuff is not being reconstructed... :((
  - however, i will probably want to feed in partial point clouds and set 
    the property to **pointcloud**, such that it doesn't alter the input
  - for both, they:
    - subsample to `pointcloud_n`, in our case 10k
    - add noise based on `pointcloud_noise`, in our case 0.005
- **multi_file:**
  - if we use multiple files as input, it chooses one of them randomly
    (other files are NOT used)
  - the base filename is `pointcloud_file` and then `_00.npz` will be added
    (except when set to null, then it expects the ending as well)
  - if multifile is active, we need a subdirectory with the base filename;
    if not then the file should be directly in the directory
- point cloud files:
  - need to have fields `points` and `normals`
  - normals are only used in evaluation, but need to be there
  - some also have `semantics`, but it still runs when they are missing
  - they are weirdly normalised in the files
- **Shapes3dDataset:**
  - this is the torch dataset class
  - if **classes** (in config, here categories) is none, then all classes
    (i.e. sub-directories) will be used.


### what to check

- what happens if I feed in a partial point cloud?
  - so the reconstruction result is obviously awful
  - however, even if I feed in a full point cloud it looks quite bad!
  - this might be due to:
    - not seen these shapes before... probably try chairs and tables?
    - some normalisation of PCs?
    - their demo input data seem to have more points on the objects than
      on the wall/ground... which I consider cheating

### how to train with an own dataset

gag contains 600 scenes, 200 each with 3, 4, and 5 objects from the YCB 
object set. The test set consists of 20x3=60 scenes with different
objects from the YCB.

The directory and file structure is exactly the same. We do not use
multi-file, only single-file option. When switching to multi-file, we
probably also need multi-files for the iou points!

For the sake of testing, I created val.lst file which lists validation
scenes, but at the moment they are the same as the test set. This should
be changed in the future to allow clean model selection.

As a config file, I started off based on room_3plane config, i.e.
using the 3 feature planes and no grid. To fit it into memory, I reduced
the dimension of the feature planes to 64 (from 128) without any further
adjustments. Perhaps it is better to instead reduce the batch size --
I should experiment with this. It currently uses around 5GB of memory.
It uses the 4view point cloud, not individual views.

#### scaling

The scaling applied in ConvONet is fixed, it expects my data to be in
[-0.55, 0.55] when padding is set to 0.1; or [-0.5, 0.5] with zero pad.
Since my scenes are in [0, 0.297] I rescale everything upon loading 
as follows:
```Python
points /= 0.297  # scales to [0, 1]
points -= 0.5    # shifts to [-0.5, 0.5], i.e. use padding = 0
```
This is done in the `src/data/fields.py` while loading PointField and
PointCloudField. The normalisation of ConvONet is still active, I did
not mess around with that.

Regarding Ferrari-Canny score, this procedure should be fine, as we 
  - keep aspect ratio
  - use constant scale factor for all scenes

The constant scale factor changes the actual score, but I assume
it is something the network can deal with, it simply learns on a
different scale.

#### Other stuff
- What is an iteration, what is an epoch? How long should we train?
  - [link to github question](https://github.com/autonomousvision/convolutional_occupancy_networks/issues/22)
  - If you train with a batch size of 24, you need to train for 2500 epoch (~300K iterations).
  - One iteration = one batch, i.e. also depends on the dataset size
- Retraining using `room_3plane.yaml` works fine it seems
  - just about uses all my GPU memory with batch size 24