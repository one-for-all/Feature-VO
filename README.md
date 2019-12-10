# Feature-based Monocular Visual Odometry

## Dependencies

This is a python2 project, and the following packages are required

* matplotlib
* mpl_toolkits
* opencv-contrib-python==3.4.2.17
* tqdm
* numpy
* scipy

----------------------------------

## Datasets

Because the datasets are large (on the order of hundreds of MB), they are not included in this project directly. They can be downloaded from https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html. In particular, the Living Room 'lr kt2' dataset used for plots in the paper can be downloaded from this link: http://www.doc.ic.ac.uk/~ahanda/living_room_traj2_frei_png.tar.gz.

Once the dataset has been downloaded, perform the following steps:

1. Create a directory called `datasets` at the root of this project.
2. Unzip the dataset, and place it in the datasets folder.
3. Rename the `*.freibug` file as `groundtruth.txt`.

----------------------------------

## To Run the Code

### To estimate the trajectory for a dataset

1. `python main.py` which optionally takes the path to the dataset folder. This will estimate the trajectory, and write it into a file.
2. `python plot_traj.py` will plot the trajectory against the ground-truth.

Things that can be set in the `main.py` are:

* `estimate_scale` which indicates whether we want to estimate scale.
* `step` is the step size.

### To obtain the errors with different step sizes

* `python experiment.py`

This will print at the end the errors for step sizes ranging from 30 to 2.

Note that the data obtained for the error plot in the paper has been copied to `plot_util/plot.py`. Therefore, calling `python plot.py` will generate the exact same figure as in paper.