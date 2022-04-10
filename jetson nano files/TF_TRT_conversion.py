from helper import ModelOptimizer

PRECISION = "FP16" # Options are "FP32", "FP16", or "INT8"

model_dir = 'saved_model'

opt_model = ModelOptimizer(model_dir)

model_fp16 = opt_model.convert(model_dir+'_trt_fp16', precision=PRECISION)
#model_int8 = opt_model.convert(model_dir+'_trt_int8', precision=PRECISION)