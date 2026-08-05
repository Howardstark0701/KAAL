# KAAL — CLI Reference

Full command reference for `kaal` CLI. See Spec 15.1 for complete specification.

## kaal audit

Run a full adversarial vulnerability audit.

```
kaal audit \
  --model    PATH     Model file (.h5 / .pt / .onnx / .tflite)  [required]
  --dataset  PATH     Image directory                            [required]
  --attacks  TEXT     fgsm,pgd,patch,blackbox,physical           [default: fgsm,pgd,patch,physical]
  --epsilon  FLOAT    Perturbation strength                      [default: 0.03]
  --steps    INT      PGD steps                                  [default: 40]
  --output   PATH     Output directory                           [default: ./kaal_output/]
  --report   TEXT     pdf,json,html,all                          [default: pdf,json]
  --no-gradcam        Skip GradCAM (faster)
  --quiet             Suppress progress output
```

## kaal serve

Launch the KAAL web UI.

```
kaal serve \
  --port  INT   Port number    [default: 8080]
  --host  TEXT  Host address   [default: 127.0.0.1]
```

## kaal patch

Generate an adversarial patch only.

```
kaal patch \
  --model     PATH    Model file              [required]
  --dataset   PATH    Image directory         [required]
  --target    INT     Target class index      [required]
  --size      FLOAT   Patch size (fraction)   [default: 0.05]
  --print-cm  FLOAT   Physical print size cm  [default: 15.0]
  --output    PATH    Output directory        [default: ./kaal_output/]
```

## kaal compare

Compare two audit JSON reports.

```
kaal compare \
  --before  PATH   First audit JSON    [required]
  --after   PATH   Second audit JSON   [required]
  --output  PATH   Output directory
```
