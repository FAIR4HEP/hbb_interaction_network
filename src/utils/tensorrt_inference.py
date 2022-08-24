import tensorrt as trt
import pycuda.driver as cuda
import tensorrt as trt
import time
import numpy as np
import glob
import tqdm
import sklearn.metrics as _m
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from engine import build_engine

sample_size = 1800000
batch_sizes= [200,400,600,800,1000,1200,1400,1500,1600,1800,2000,2200,2400,2600,3000,3400,3800,4200]
save_path = '//grand/RAPINS/ruike/new_hbb/test/'# './test_hbb/'#

test_2_arrays = []
test_3_arrays = []
target_test_arrays = []

for test_file in sorted(glob.glob(save_path + 'test_*_features_2.npy')):
    print("!!",test_file)
    test_2_arrays.append(np.load(test_file))
test_2 = np.concatenate(test_2_arrays)
for test_file in sorted(glob.glob(save_path + 'test_*_features_3.npy')):
    test_3_arrays.append(np.load(test_file))
test_3 = np.concatenate(test_3_arrays)
for test_file in sorted(glob.glob(save_path + 'test_*_truth_0.npy')):
    target_test_arrays.append(np.load(test_file))
label = np.concatenate(target_test_arrays)
test_2 = np.swapaxes(test_2, 1, 2)
test_3 = np.swapaxes(test_3, 1, 2)
print(test_2.shape)


#####
# Make device context
#####
cuda.init()
device = cuda.Device(0)  # GPU id 0
device_ctx = device.make_context()


#####
# inference func
#####
def run_inference(test,test_sv,batch_size, i):
  # Load image to memory buffer
    start_time =time.perf_counter() #time.process_time() #
    preprocessed = test.ravel()
    preprocessed_sv = test_sv.ravel()

    np.copyto(h_input_1, preprocessed)
    np.copyto(h_input_2, preprocessed_sv)
    with engine.create_execution_context() as exec_ctx:
        # Transfer data to device (GPU)
        cuda.memcpy_htod_async(d_input_1, h_input_1, stream)
        cuda.memcpy_htod_async(d_input_2, h_input_2, stream)

        # Run inference
        #exec_ctx.profiler = trt.Profiler()
        #start_time = time.process_time()
        exec_ctx.execute_async(batch_size, bindings=[int(d_input_1),int(d_input_2), int(d_output)],stream_handle = stream.handle)
        #exec_ctx.execute(batch_size, bindings=[int(d_input_1),int(d_input_2), int(d_output)])

        stream.synchronize()

        # Transfer predictions back from device (GPU)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)

        # Synchronize the stream
        stream.synchronize()
        stop_time = time.perf_counter() #time.process_time() #
        time_ = stop_time-start_time
        out_ = h_output

        # Return ndarray
        return out_,time_

##############################
# start to run inference
##############################
overall_auc =[]
overall_throughput =[]
for batch_size in batch_sizes:
    #####
    # Load ONNX with TensorRT ONNX Parser,
    #   Create an Engine,
    #     and Serialize into a .plan file
    #####
    model_path = "../../models/trained_models/onnx_model/5_10_gnn_%s.onnx"%batch_size
    build_engine(model_path, batch_size)
    mean_throughput = []
    mean_auc=[]
    for n in range(10):
        roc_score =[]
        result_arr=[]
        time_arr=[]
        label_ =[]
        count = 0
        pbar = tqdm.tqdm(range(int(sample_size/batch_size)-1))
        for i in pbar:
            start_idx = i*batch_size
            test = test_2[1+start_idx:1+start_idx+batch_size]
            test_sv = test_3[1+start_idx:1+start_idx+batch_size]
            label_batch = label[1+start_idx:1+start_idx+batch_size]

            #####
            # Load Engine and Define Inferencing Function
            #####
            # Create page-locked memory buffers (which won't be swapped to disk)
            h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume((30,60)), dtype=trt.nptype(trt.float32))
            h_input_2 = cuda.pagelocked_empty(batch_size * trt.volume((14,5)), dtype=trt.nptype(trt.float32))
            h_output = cuda.pagelocked_empty(batch_size * trt.volume((1, 2)), dtype=trt.nptype(trt.float32))

            # Allocate device memory
            d_input_1 = cuda.mem_alloc(h_input_1.nbytes)
            d_input_2 = cuda.mem_alloc(h_input_2.nbytes)
            d_output = cuda.mem_alloc(h_output.nbytes)

            # Create stream
            stream = cuda.Stream()

            # Load (Deserialize) engine
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            trt_runtime = trt.Runtime(TRT_LOGGER)

            with open("../../models/trained_models/tensorrt_models/5_10_gnn.plan", 'rb') as f:
                engine_data = f.read()
                engine = trt_runtime.deserialize_cuda_engine(engine_data)

            result,time_ = run_inference(test,test_sv,batch_size, i)
            result = result.reshape((batch_size,-1))
            time_arr.append(time_)
            for x in label_batch:
                label_.append(x.tolist())
            for x in result:
                result_arr.append(x.tolist())
        print(len(result_arr), batch_size)

        ##############################
        # calculate auc, throughput
        ##############################
        clip_soft_res = []
        clip_label = []
        for i in range(len(label_)):
            if [0.,0,] == label_[i]:
                continue
            else:
                temp = softmax(result_arr[i], axis=0)
                clip_soft_res.append(temp)
                clip_label.append(label_[i])

        fpr, tpr, thresholds = _m.roc_curve(np.array(clip_label)[:,1], np.array(clip_soft_res)[:,1])
        auc = _m.auc(fpr, tpr)   ##roc_score
        print("auc:%.9f"% auc)
        print("acc %.9f"% accuracy_score(np.array(clip_label)[:,1], np.array(clip_soft_res)[:,1]>0.5) )
        print("time %.9f"% np.mean(time_arr[:]))
        print( "through put:%.4f"%  (batch_size/np.mean(time_arr[:])))

        temp = batch_size/np.mean(time_arr[:])
        mean_throughput.append(temp)
        mean_auc.append(auc)

    overall_auc.append(np.mean(mean_auc))
    overall_throughput.append(mean_throughput)

np.save("../../models/trained_models/tensorrt_models/throughput.npy",overall_throughput)

print(overall_auc)
print(overall_throughput)
device_ctx.pop()
