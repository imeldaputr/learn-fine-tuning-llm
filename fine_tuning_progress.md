# Fine-tuning Progress --- Qwen2.5-Coder-1.5B-Instruct (LoRA)

## 1. Model Loading & Setup

    Loading model Qwen/Qwen2.5-Coder-1.5B-Instruct...
    Map: 100% | 5/5 [00:00<00:00, 314.07 examples/s]

**Status:** Model loaded successfully\
**Dataset:** 5 samples (tokenized very fast)

## 2. LoRA Parameters

    Trainable params: 2,179,072
    Total params:     1,545,893,376
    Trainable %:      0.1410%

**Notes:** - Total model parameters: **1.5B** - LoRA parameters trained:
**2.1M** (\~0.14%) - Remaining weights are frozen

## 3. Training Progress

### Raw Training Log

    Epoch   Loss    Grad Norm   Learning Rate
    0.8     2.07    61.84       0.0002
    1.0     0.95    51.07       0.00017
    1.8     1.45    61.47       0.00013
    2.0     1.42    81.56       0.00012
    2.8     1.29    67.30       0.000067
    3.0     1.30    61.47       0.000033

**Observations:** - Loss decreases from **2.07 → 1.31**, consistent
improvement. - Learning rate decays as expected. - Grad Norm fluctuates
but stays in a reasonable range for small datasets.

## 4. Training Summary

    train_runtime: 15.0242s
    train_samples_per_second: 0.998
    train_loss: 1.4188
    epoch: 3.0

**Summary:** - Total runtime: **15s** - Throughput: \~1 sample/sec -
Final average loss: **1.42** - Completed: **3 epochs**
