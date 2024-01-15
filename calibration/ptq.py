import tensorrt as trt
import os
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2

def preprocess_v1(image_raw, width=640, height=640):
    h, w, c = image_raw.shape
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
    # Calculate widht and height and paddings
    r_w = width / w
    r_h = height / h
    if r_h > r_w:
        tw = width
        th = int(r_w * h)
        tx1 = tx2 = 0
        ty1 = int((height - th) / 2)
        ty2 = height - th - ty1
    else:
        tw = int(r_h * w)
        th = height
        tx1 = int((width - tw) / 2)
        tx2 = width - tw - tx1
        ty1 = ty2 = 0
    # Resize the image with long side while maintaining ratio
    image = cv2.resize(image, (tw, th))
    # Pad the short side with (128,128,128)
    image = cv2.copyMakeBorder(
        image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, (128, 128, 128)
    )
    image = image.astype(np.float32)
    # Normalize to [0,1]
    image /= 255.0
    # HWC to CHW format:
    image = np.transpose(image, [2, 0, 1])
    # CHW to NCHW format
    #image = np.expand_dims(image, axis=0)
    # Convert the image to row-major order, also known as "C order":
    #image = np.ascontiguousarray(image)
    return image

class yolov5EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, imgpath, batch_size, channel, inputsize=[640, 640]):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = 'yolov5.cache'
        self.batch_size = batch_size
        self.Channel = channel
        self.height = inputsize[0]
        self.width = inputsize[1]
        self.imgs = [os.path.join(imgpath, file) for file in os.listdir(imgpath) if file.endswith('jpg')]
        np.random.shuffle(self.imgs)
        self.imgs = self.imgs[:1000]
        self.batch_idx = 0
        self.max_batch_idx = len(self.imgs) // self.batch_size
        self.calibration_data = np.zeros((self.batch_size, 3, self.height, self.width), dtype=np.float32)
        # self.data_size = trt.volume([self.batch_size, self.Channel, self.height, self.width]) * trt.float32.itemsize
        self.data_size = self.calibration_data.nbytes
        self.device_input = cuda.mem_alloc(self.data_size)
        # self.device_input = cuda.mem_alloc(self.calibration_data.nbytes)

    def free(self):
        self.device_input.free()

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        try:
            batch_imgs = self.next_batch()
            if batch_imgs.size == 0 or batch_imgs.size != self.batch_size * self.Channel * self.height * self.width:
                return None
            cuda.memcpy_htod(self.device_input, batch_imgs)
            return [int(self.device_input)]
        except:
            print('get batch wrong')
            return None
    def next_batch(self):
        if self.batch_idx < self.max_batch_idx:
            batch_files = self.imgs[self.batch_idx * self.batch_size: \
                                    (self.batch_idx + 1) * self.batch_size]
            batch_imgs = np.zeros((self.batch_size, self.Channel, self.height, self.width),
                                  dtype=np.float32)
            for i, f in enumerate(batch_files):
                img = cv2.imread(f)  # BGR
                img = preprocess_v1(img)
                batch_imgs[i] = img
            self.batch_idx += 1
            print("batch:[{}/{}]".format(self.batch_idx, self.max_batch_idx))
            return np.ascontiguousarray(batch_imgs)
        else:
            return np.array([])
    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
            f.flush()
            # os.fsync(f)


def get_engine(onnxFile, engine_file_path, cali_img, mode='FP32', workspace_size=4096):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""
    logger = trt.Logger(trt.Logger.VERBOSE)
    def build_engine():
        assert mode.lower() in ['fp32', 'fp16', 'int8'], "mode should be in ['fp32', 'fp16', 'int8']"

        
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()

        #config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # workspace_sizeMiB
        # 构建精度
        if mode.lower() == 'fp16':
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

        if mode.lower() == 'int8':
            print('trt.DataType.INT8')
            config.flags |= 1 << int(trt.BuilderFlag.INT8)
            config.flags |= 1 << int(trt.BuilderFlag.FP16)
            calibrator = yolov5EntropyCalibrator(cali_img, 1, 3, [640, 640])
            # config.set_quantization_flag(trt.QuantizationFlag.CALIBRATE_BEFORE_FUSION)
            config.int8_calibrator = calibrator
        # if True:
        #     config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
        parser = trt.OnnxParser(network, logger)
        if not os.path.exists(onnxFile):
            print("Failed finding ONNX file!")
            exit()
        print("Succeeded finding ONNX file!")
        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read()):
                print("Failed parsing .onnx file!")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
            print("Succeeded parsing .onnx file!")

        profile = builder.create_optimization_profile()
        profile.set_shape(network.get_input(0).name, [1,3,640,640], [1,3,640,640],[1,3,640,640]) #                         min=(1, 3, 640, 640), opt=(4, 3, 640, 640), max=(24, 3, 640, 640))
        config.add_optimization_profile(profile)
        # config.set_calibration_profile(profile)
        print("Completed parsing of ONNX file")
        print("Building an engine from file {}; this may take a while...".format(onnx_file_path))
        # plan = builder.build_serialized_network(network, config)
        # engine = runtime.deserialize_cuda_engine(plan)
        engine = builder.build_serialized_network(network, config)
        if engine == None:
            print("Failed building engine!")
            exit()
        print("Succeeded building engine!")
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            # f.write(plan)
            f.write(engine)
        return engine
        
    if os.path.exists(engine_file_path):
        # If a serialized engine exists, use it instead of building an engine.
        print("Reading engine from file {}".format(engine_file_path))
        with open(engine_file_path, "rb") as f, trt.Runtime(logger) as runtime:
            return runtime.deserialize_cuda_engine(f.read())
    else:
        return build_engine()


def main(onnx_file_path, engine_file_path, cali_img_path, mode='FP32'):
    # Try to load a previously generated YOLOv3-608 network graph in ONNX format:
    get_engine(onnx_file_path, engine_file_path, cali_img_path, mode)


if __name__ == "__main__":
    onnx_file_path = '/work/simple_yolov5_demo/model/gddi_person_se5.onnx'
    engine_file_path = "/work/simple_yolov5_demo/model/gddi_person_se5_model_int8.plan"
    cali_img_path = '/work/simple_yolov5_demo/calibration_image/coco1000/'
    main(onnx_file_path, engine_file_path, cali_img_path, mode='int8')