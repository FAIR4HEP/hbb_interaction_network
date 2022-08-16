1. run random train-test split
    first run below to save the random splitted train test data(submit job to run this)
        python generate_train_test.py
    then run the inference:
        python IN_dataGenerator_random.py IN_training 1 0 --De 20 --Do 24 --hidden 60



1. run the conversion from pytorch to onnx, and use onnx to inference
    set up environemnt:
        module load conda/2021-06-28
        conda activate base
    run model:
        python pytorch_onnx_inference.py
    also include a notebook:
        pytorch2onnx.ipynb
        
2. run the conversion from onnx to tensorrt, and use tensorrt  to inference 
    set up environment:
        module load conda/2021-11-30
        conda activate base
    run model:
        python tensorrt_inference.py

3. use singularity container
    start the container:
        if you are first time use this container, need to build the instance first:
            singularity instance start  --nv ./trt_torch_new.sif tensorrt
        after build the instance:
            singularity shell --bind //grand/RAPINS/ruike/new_hbb/test://home/ruike/merge_IN/notebook_code/test_hbb //home/ruike/trt_torch_new.sif ls //home/ruike/merge_IN/notebook_code/test_hbb
            pip install -U scikit-learn

4. throughput & gpu throughput plot 
    the plots are saved in throughput_gpu_use.ipynb file
            
        