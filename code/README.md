# Code

This is the actual code behind the results in `../Model Reports/`: custom
architecture modules for YOLOv5/v7/v9, the configs that wire them into each
model's backbone/neck, and the Colab notebook used to drive training.

## Layout

```
code/
├── notebooks/
│   └── training_notebook.ipynb   # the original Colab driver (repo cloning, dataset
│                                  # prep, per-model training/resume commands)
├── yolov5/
│   ├── configs/yolov5l_C3HB.yaml
│   └── modules/                  # C3HB + its HorNet-attention internals
├── yolov7/
│   ├── configs/
│   │   ├── yolov7-Transformer-HorNet-attnetion.yaml
│   │   ├── yolov7-C3C2-SPPFCSPC.yaml
│   │   └── yolov7-MobileOne.yaml
│   └── modules/                  # C3HB/C3TR, CNeB/C3C2/MobileOne/SPPFCSPC + deps
└── yolov9/
    └── configs/yolov9-Transformer-HorNet-attnetion.yaml   # experimental, not in the
                                                             # published results table
```

The base YOLOv5/v7/v8/v9 runs use the stock model configs from each
project's upstream repo — no custom code needed for those.

## How the custom modules attach

Each architecture variant is a stock [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
or [WongKinYiu YOLOv7](https://github.com/WongKinYiu/yolov7) clone with a
handful of extra `nn.Module` classes registered, then a `--cfg` YAML that
references them by name in the backbone/neck.

To reproduce a variant:

1. Clone the matching upstream repo (pin close to a mid-2024 commit —
   that's when this project's training ran).
2. Copy this project's `modules/*.py` files for that architecture into the
   clone (e.g. `yolov7/models/custom/`), and import their classes into
   `models/common.py` (or register them directly in `models/yolo.py`'s
   `parse_model`, which is what actually resolves the module names in the
   `--cfg` YAML at build time).
3. Copy the matching `configs/*.yaml` file into the clone's `configs/` or
   `cfg/training/` directory.
4. Train:
   ```bash
   python train.py --batch 16 --epochs 100 \
     --data <your_data.yaml> \
     --cfg configs/yolov7-C3C2-SPPFCSPC.yaml \
     --device 0 --project <out_dir> --name <run_name>
   ```

The custom `nn.Module`s themselves only depend on classes already present
in a stock `models/common.py` (`Conv`, `Bottleneck`, `C3`, `TransformerBlock`,
`autopad`) — nothing else needs to be vendored.

## Provenance

The `C3HB` / HorNet-attention, `CNeB` (ConvNeXt), `C3C2`, and `SPPFCSPC`
blocks are adapted from [iscyy/yoloair](https://github.com/iscyy/yoloair),
a module zoo for YOLOv5/v7. `MobileOne` follows
[Apple's MobileOne](https://arxiv.org/abs/2206.04040) (unofficial PyTorch
port). Each module file credits its source in a header comment.

## Dataset

The 272-image satellite dataset used for training isn't included here.
`training_notebook.ipynb` expects a YOLO-format dataset with a `data.yaml`
(10 classes: airliner, boat, bus, car, long vehicle, other, pushback truck,
stair truck, truck, van) mounted at a `sat-1/` path.
