"""CPU-only package/configuration check; no training or GPU qualification."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
import argparse, importlib.metadata, json, sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='artifacts/environment_check.json');a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    versions={}
    for name in ['fast-plate-ocr','tensorflow','keras','numpy','Pillow','opencv-python','opencv-python-headless','onnx','onnxruntime','onnxruntime-gpu','tf2onnx','onnxslim','tensorrt','pycuda']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)
    from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
    from fast_plate_ocr.train.model.model_schema import load_model_config_from_yaml
    from fast_plate_ocr.train.model.model_builders import build_model
    plate=load_plate_config_from_yaml(root/'config/cct_s_v2_global_plate_config.yaml')
    config=load_model_config_from_yaml(root/'config/cct_s_v2_global_model_config.yaml')
    model=build_model(config,plate,enable_region_head=False)
    import numpy as np
    trainable=sum(int(np.prod(v.shape)) for v in model.trainable_weights)
    fixed=sum(int(np.prod(v.shape)) for v in model.non_trainable_weights)
    assert trainable==977510 and fixed==432,(trainable,fixed)
    assert tuple(model.input_shape[1:])==(64,128,3)
    assert tuple(model.output_shape[1:])==(15,37)
    assert not tf.config.get_visible_devices('GPU')
    result={'pass':True,'scope':'CPU package imports, complete upstream config parsing and scratch architecture construction only; no optimizer/export/GPU qualification','python':sys.version.split()[0],'versions':versions,'input_shape':model.input_shape,'output_shape':model.output_shape,'learned_parameters':trainable,'fixed_elements':fixed,'tensorflow_visible_gpu_count':0,'historical_lock_is_not_this_environment':versions['tensorflow']!='2.21.0'}
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
