import argparse
from trt_runtime import build
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--onnx',required=True);p.add_argument('--engine',required=True);p.add_argument('--max-batch',type=int,default=32);p.add_argument('--fp16',action='store_true',help='Opt-in; requires fresh parity validation');a=p.parse_args()
    build(a.onnx,a.engine,a.max_batch,a.fp16)
