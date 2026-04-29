### License Plate OCR Training (+Tenosrrt Inference)

This repository contains code for training Lightweight & fast OCR models for license plate text recognition.
Please make sure that you have docker and nvidia-docker-toolkit installed to mount gpu inside the container.

Steps to Run:

```
docker compose up
```

After running the containers follow http://localhost:8780 for jupyter lab interface and http://localhost:8000 for Annotator interface  
please make sure that you have added your images at this path: `./annotator/static/images` before running the container.

Ref: https://github.com/ankandrew/fast-plate-ocr
