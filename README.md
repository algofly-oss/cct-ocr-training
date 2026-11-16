# CCT-S license plate OCR: experiments and reproducible workflows

A compact TensorFlow/Keras plate recognizer, with training, ONNX export, and TensorRT evaluation recipes. This repository records measured reference results and provides fictional image/text data to exercise the workflow. Dummy data is not an OCR benchmark.

## Measured reference results

The task model has **977,510 learned parameters**, 432 fixed blur elements, RGB input **64 × 128**, **15 character slots**, and **37 classes** (`0–9`, `A–Z`, padding `_`). These results describe an existing task checkpoint, not a newly trained dummy model.

| Batch | Batch latency, mean | Throughput | Sampled process GPU memory |
|---:|---:|---:|---:|
| 1 | 0.4474 ms | 2,235 images/s | 742 MiB |
| 32 | 2.5137 ms | 12,730 images/s | 742 MiB |

Scope: 256 reference plate crops, five warmups and three passes; predecoded RGB through preprocessing, transfer, model execution and decoding. The engine and allocations supported batch 32. GPU memory includes the CUDA context and is a sampled process peak. Reference-suite hardware context: NVIDIA GeForce RTX 5090, driver 595.58.03. The individual historical batch artifacts omit the GPU name and measurement date, so their date is unknown. These are not disk-to-text timings or full-test accuracy claims. See [the complete batch measurements](results/reference_metrics.json). Reproduce on your hardware before comparing numbers; new exports do not inherit this qualification.

The upstream generic `cct_s_v2_global.keras` is a **different configuration**: 1,060,907 parameters, 10 plate slots and a region branch. Compatible pretrained weights can initialize the 15-slot model, but this is not an exact whole-model restore. Upstream training uses `skip_mismatch=True`; inspect loading warnings. The 977,736-parameter price model is also distinct and is not the model documented here.

See [capabilities across layouts, rotations and domains](docs/capabilities.md) for measured limits and comparison scopes.

## Workflow

Run the clean notebooks in order:

0. [Environment](notebooks/00_environment.ipynb)
1. [Dummy data and image/TXT conversion](notebooks/01_dummy_dataset.ipynb)
2. [Native CCT-S training](notebooks/02_train.ipynb)
3. [Native Keras evaluation](notebooks/03_native_inference.ipynb)
4. [ONNX export](notebooks/04_export_onnx.ipynb)
5. [ONNX Runtime evaluation](notebooks/05_onnx_evaluation.ipynb)
6. [TensorRT build and parity](notebooks/06_tensorrt.ipynb)
7. [Benchmarking](notebooks/07_benchmark.ipynb)

Use a Python environment compatible with `fast-plate-ocr==1.1.0`. `requirements.lock` records the original environment rather than a portable CUDA installation. Install TensorRT and PyCUDA for your own driver/CUDA environment. GPU notebooks are runnable recipes; their results have not been asserted for every environment.

```bash
python -m pip install -r requirements.txt
python scripts/dataset.py dummy dummy_data
python scripts/dataset.py convert dummy_data/train
```

Each image has a same-stem UTF-8 `.txt` containing one expected plate string. The converter requires 1–15 uppercase letters or digits, rejects unsupported labels rather than silently filtering them, and writes the upstream CSV format `image_path,plate_text`. CSV image paths are relative to the CSV directory. Dummy generation is manifest-guarded: repeat runs verify every generated file and make no changes. Existing folders without a valid `DUMMY_DATA.json`, or modified fixtures, are refused without overwrite. No bounding boxes or region annotations are required. The supplied 16 training and four validation images contain fictional plates.

The training notebook defaults to one smoke epoch and scratch initialization. Set your own CSV paths, epochs, and optional pretrained weights explicitly. Replace `REPLACE_RUN` with the actual training output directory; nothing deletes existing weights. Separate training, validation, and final test images before training.

The native model contains its input rescaling. Evaluation feeds RGB values in 0–255 with linear resizing; it does not divide inputs by 255 again. ONNX export requests dynamic batches, float32 input and NCHW explicitly. Native and ONNX evaluation also support the exported NHWC layout. Decoding stops at the first padding class and retains the original alphabet.

TensorRT defaults to FP32 with TF32 disabled. FP16 is an explicit option and requires fresh parity. The parity runner checks logits, frame argmax and decoded strings against ONNX Runtime; a failed numerical check remains a failed qualification even if strings agree. Every output is copied back in the measured scope. The simple benchmark runner does not measure GPU memory; use an isolated process and external NVML monitoring for memory comparisons. With four dummy validation images, requested batch 8/32 is only a partial batch: use a larger cohort for meaningful throughput.

## Model scope and provenance

Only the checked-in CCT-S configuration is provided; there are no invented Tiny/Small/Large variants. Its convolutional tokenizer, five spatial transformer blocks, query reducer, three post-reducer blocks and DyT normalization are implemented by upstream [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr). Training and export remain upstream APIs, preserving Keras compatibility. Generalization to different countries, layouts, rotated crops or unseen imaging conditions must be measured; small size and fast inference do not establish that capability.

Private datasets, model weights, engines, local archives and generated run outputs are excluded by `.gitignore`. The original local notebooks are archived outside the publication set; the root notebook names are compatibility links to the clean workflow. The existing remote commit history is being preserved and contains two plate-image outputs in older notebooks. The current notebooks are cleared and use dummy-data defaults; clearing the current revision does not erase historical images. No new private images are included. Never commit credentials. Upstream dependency licenses remain applicable; see [source provenance](results/source_provenance.json). No pretrained weights or private benchmark images are redistributed here.


## Docker and local label editor

```bash
# NVIDIA Container Toolkit and a compatible host driver are required for the GPU service.
JUPYTER_TOKEN=choose-a-local-password docker compose up --build
```

JupyterLab is available at `http://127.0.0.1:8780` and the image/text label editor at `http://127.0.0.1:8000`. Both ports bind to localhost. The default token `cct-local` is for local setup; set your own token before sharing access. The notebook image uses TensorFlow 2.19.0 and Keras 3.12.1, with no PyTorch dependency. The complete configuration was checked with these TensorFlow/Keras versions on CPU; the Docker build itself has not been executed in this preparation environment because Docker is unavailable. TensorRT/PyCUDA are separate target-machine installations, not supplied or qualified by this image.

Place your own cropped plate images in `annotator/static/images/`. Only `.gitkeep` is included in the release. The editor writes same-stem UTF-8 TXT labels without changing or deleting the original images. Use 1–15 uppercase letters/digits, then run the strict CSV converter on your chosen train and validation folders. The editor is a local annotation utility, not an OCR detector or a bounding-box labeling requirement.

The repository is mounted into the container; private datasets and weights are not copied into the Docker image. For a CPU-only host, remove the NVIDIA device reservation from a local Compose override before starting Jupyter. No CPU training-performance claim is made.

## License status

The historical remote repository did not include a root license. `NOTICE` records the upstream MIT dependency and the Apache-2.0 shared annotator (`annotator/LICENSE`). It does not retroactively license the historical notebooks. A root license for the remaining workflow files is an owner publication decision; public visibility alone does not grant unrestricted reuse rights.
