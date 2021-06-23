# How to evaluate depth estimation
The competitor will need to save the result which has the filename as `3dfront-dataset-batch-split-10k-0-cam-(index)/3dfront-dataset-batch-split-10k-0-cam-(index).exr'(exr is recommended since it can store float image). And put all the results depth image in one folder, zip it and submit them to us. <br>
Make sure the depth image is distance to camera plane in meter unit, and closer object have smaller depth value.

## Run the evaluation program
```angular2
bash py_entrance.sh input_param.json output_report.json
```
The input_param includes the path of the answer directory and the predicted results directory. 
Please make sure your predicted results is saved as the same format of the provided example in result_dir.