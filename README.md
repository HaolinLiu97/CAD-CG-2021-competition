# single-view RGBD dataset for CAD & CG 2021 competition

### Visualize the data
you can run the draw_bbox.py to visualize the 3d data of the dataset. Open3D is required for the visualization.

### number of class
There are totally 9 class in the dataset.

### How to convert depth image to distance to camera plane
```
distance=(1-depth_value/255.0)*10
```