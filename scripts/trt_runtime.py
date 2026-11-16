"""TensorRT 10 explicit-batch runtime. No CUDA initialization on import."""
from pathlib import Path
import numpy as np

def build(onnx, engine, max_batch=32, fp16=False):
    import tensorrt as trt
    logger=trt.Logger(trt.Logger.WARNING)
    builder=trt.Builder(logger);network=builder.create_network(0)
    parser=trt.OnnxParser(network,logger)
    if not parser.parse(Path(onnx).read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if network.num_inputs!=1:raise ValueError('Expected one image input')
    inp=network.get_input(0);shape=list(inp.shape)
    if shape[1:] not in ([3,64,128],[64,128,3]):raise ValueError(f'Unsupported shape {shape}')
    cfg=builder.create_builder_config();cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,1<<30)
    cfg.clear_flag(trt.BuilderFlag.TF32)
    if fp16:cfg.set_flag(trt.BuilderFlag.FP16)
    if shape[0]<0:
        profile=builder.create_optimization_profile()
        profile.set_shape(inp.name,[1]+shape[1:],[min(8,max_batch)]+shape[1:],[max_batch]+shape[1:]);cfg.add_optimization_profile(profile)
    elif shape[0]!=1:raise ValueError('Only static B1 or dynamic batch supported')
    serialized=builder.build_serialized_network(network,cfg)
    if serialized is None:raise RuntimeError('TensorRT build failed')
    Path(engine).parent.mkdir(parents=True,exist_ok=True);Path(engine).write_bytes(bytes(serialized))

class Runtime:
    def __init__(self,path):
        import tensorrt as trt
        import pycuda.driver as cuda
        self.cuda=cuda;self.trt=trt;cuda.init();self.device_context=cuda.Device(0).make_context()
        self.allocations={}
        try:
            self.runtime=trt.Runtime(trt.Logger(trt.Logger.WARNING));self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None:raise RuntimeError('Invalid engine')
            self.context=self.engine.create_execution_context();self.stream=cuda.Stream()
            self.names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_name=next(n for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT)
            self.shape=tuple(self.engine.get_tensor_shape(self.input_name));self.dtype=np.dtype(trt.nptype(self.engine.get_tensor_dtype(self.input_name)))
        except Exception:
            self.close();raise
    def run(self,x):
        x=np.ascontiguousarray(x,dtype=self.dtype)
        if not self.context.set_input_shape(self.input_name,x.shape):raise ValueError(f'Invalid shape {x.shape}')
        outputs={}
        for n in self.names:
            shape=tuple(self.context.get_tensor_shape(n));dtype=np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(n)))
            if any(v<0 for v in shape):raise ValueError(f'Unresolved shape {n}: {shape}')
            size=int(np.prod(shape))*dtype.itemsize
            if n not in self.allocations or self.allocations[n][1]<size:
                if n in self.allocations:self.allocations[n][0].free()
                self.allocations[n]=(self.cuda.mem_alloc(size),size)
            self.context.set_tensor_address(n,int(self.allocations[n][0]))
            if n!=self.input_name:outputs[n]=np.empty(shape,dtype)
        self.cuda.memcpy_htod(self.allocations[self.input_name][0],x)
        if not self.context.execute_async_v3(self.stream.handle):raise RuntimeError('Execution failed')
        self.stream.synchronize()
        for n,out in outputs.items():self.cuda.memcpy_dtoh(out,self.allocations[n][0])
        return list(outputs.values())
    def close(self):
        for allocation,_ in getattr(self,'allocations',{}).values():allocation.free()
        self.allocations={}
        if getattr(self,'device_context',None):self.device_context.pop();self.device_context.detach();self.device_context=None
