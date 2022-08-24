import tensorrt as trt


#####
# Load ONNX with TensorRT ONNX Parser,
#   Create an Engine,
#     and Serialize into a .plan file
#####
def build_engine(model_path, batch_size):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
        1
    ) as network, builder.create_builder_config() as config, trt.OnnxParser(network, TRT_LOGGER) as parser:

        # .max_batch_size = batch_size
        builder.max_batch_size = batch_size
        config.max_workspace_size = 256 << 20
        # Load ONNX
        with open(model_path, "rb") as model:
            #         parser.parse(model.read())
            #         print(network.get_layer(network.num_layers - 1).get_output().shape)
            if not parser.parse(model.read()):
                print(parser.get_error(0))

        # Create engine
        network.get_input(0).shape = [batch_size, 30, 60]
        network.get_input(1).shape = [batch_size, 14, 5]
        engine = builder.build_engine(network, config)
        # Serialize engine in .plan file
        buf = engine.serialize()
        with open("../../models/trained_models/tensorrt_models/5_10_gnn.plan", "wb") as f:
            f.write(buf)
