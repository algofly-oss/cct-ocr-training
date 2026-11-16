"""Native/ORT/TRT evaluation and timed RGB-to-string inference, with no label filtering."""
import argparse,csv,json,time
from pathlib import Path
import numpy as np
import cv2
ALPHABET='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
def decode(outputs):
    values=list(outputs.values()) if isinstance(outputs,dict) else outputs if isinstance(outputs,(list,tuple)) else [outputs]
    candidates=[np.asarray(v) for v in values if len(v.shape)==3 and v.shape[-1]==37]
    if len(candidates)!=1:raise ValueError('Expected exactly one [B,slots,37] plate output')
    logits=candidates[0]
    return [''.join(ALPHABET[i] for i in row).split('_',1)[0] for row in logits.argmax(-1)],logits
class Backend:
    def __init__(self,kind,path):
        self.kind=kind
        if kind=='onnx':
            import onnxruntime as ort
            self.model=ort.InferenceSession(path,providers=['CPUExecutionProvider']); inp=self.model.get_inputs()[0]
            self.shape=inp.shape;self.dtype=np.uint8 if inp.type=='tensor(uint8)' else np.float32
        elif kind=='native':
            from fast_plate_ocr.train.utilities.utils import load_keras_model
            from fast_plate_ocr.train.model.config import load_plate_config_from_yaml
            config=load_plate_config_from_yaml(Path(__file__).resolve().parents[1]/'config/cct_s_v2_global_plate_config.yaml')
            self.model=load_keras_model(Path(path),config);self.shape=self.model.input_shape;self.dtype=np.float32
        else:
            from trt_runtime import Runtime
            self.model=Runtime(path);self.shape=self.model.shape;self.dtype=self.model.dtype
    def infer(self,images):
        x=np.stack([cv2.resize(im,(128,64),interpolation=cv2.INTER_LINEAR) for im in images]).astype(self.dtype)
        if self.shape[1]==3:x=x.transpose(0,3,1,2).copy()
        if self.kind=='onnx':out=self.model.run(None,{self.model.get_inputs()[0].name:x})
        elif self.kind=='native':out=self.model(x,training=False)
        else:out=self.model.run(x)
        return decode(out)
    def close(self):
        if self.kind=='trt':self.model.close()
def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['native','onnx','trt'],required=True);p.add_argument('--model',required=True);p.add_argument('--annotations',default='dummy_data/val_annotations.csv');p.add_argument('--output',default='artifacts/evaluation.json');p.add_argument('--batch-size',type=int,default=1);p.add_argument('--repeats',type=int,default=1);p.add_argument('--warmups',type=int,default=5);p.add_argument('--compare-onnx');a=p.parse_args()
    if min(a.batch_size,a.repeats)<1:raise ValueError('Positive batch/repeats required')
    f=Path(a.annotations);rows=list(csv.DictReader(f.open()));images=[]
    for r in rows:
        im=cv2.imread(str(f.parent/r['image_path']))
        if im is None:raise ValueError(r['image_path'])
        images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
    b=Backend(a.backend,a.model);ref=Backend('onnx',a.compare_onnx) if a.compare_onnx else None
    try:
        for _ in range(a.warmups):b.infer(images[:a.batch_size])
        times=[];records=[];max_error=0.;argmax_diffs=0
        for repeat in range(a.repeats):
            for start in range(0,len(rows),a.batch_size):
                batch=images[start:start+a.batch_size];t=time.perf_counter();pred,logits=b.infer(batch);times.append((time.perf_counter()-t)*1000)
                if ref:
                    rp,rl=ref.infer(batch);max_error=max(max_error,float(np.max(np.abs(logits-rl))));argmax_diffs+=int(np.count_nonzero(logits.argmax(-1)!=rl.argmax(-1)))
                    if pred!=rp:raise AssertionError('TRT/ORT decoded strings differ')
                for r,s in zip(rows[start:start+a.batch_size],pred):records.append({'repeat':repeat,'image':r['image_path'],'gt':r['plate_text'],'prediction':s})
        result={'scope':'predecoded RGB through resize, host transfer, inference, full output copy and decoding; disk decoding excluded','backend':a.backend,'batch_size':a.batch_size,'warmups':a.warmups,'repeats':a.repeats,'examples':len(rows),'exact':sum(r['gt']==r['prediction'] for r in records),'exposures':len(records),'batch_ms_mean':float(np.mean(times)),'batch_ms_p95':float(np.percentile(times,95)),'images_per_second':len(records)/(sum(times)/1000),'records':records,'dummy_data_is_not_quality_evidence':True,'memory_measurement':'not measured by this runner; use isolated PID NVML monitoring','parity':None if not ref else {'max_absolute_logit_error':max_error,'frame_argmax_differences':argmax_diffs,'strings_equal':True,'pass':max_error<=.05 and argmax_diffs==0}}
        Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(result,indent=2)+'\n')
        if ref and not result['parity']['pass']:raise AssertionError('Numerical/argmax parity failed; result preserved')
    finally:
        b.close()
        if ref:ref.close()
if __name__=='__main__':main()
