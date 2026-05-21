
# Polyp Segmentation Using U-Net Architecture
TR-SE-Net (Triple-Receptive Squeeze-and-Excitation Network) also known as => Parallel Multi-Scale Depthwise Separable Squeeze-and-Excitation (PMS-DSSE) Block
![model predection result](./public/prediction_result.png)


![Training History](./public/training_history.png)


For Activate GPU: 
```sh
# Final Fix:
export VENV_PACKAGES="/mnt/c/development/Thesis/Gastrovision/linux-venv/lib/python3.12/site-packages"

# Set Important Libraries:
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$VENV_PACKAGES/nvidia/cudnn/lib:$VENV_PACKAGES/nvidia/cublas/lib:$VENV_PACKAGES/nvidia/cuda_runtime/lib:$VENV_PACKAGES/nvidia/cusolver/lib:$VENV_PACKAGES/nvidia/cusparse/lib:$LD_LIBRARY_PATH


# Inform the System
sudo ldconfig
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

Datasets:
1. PolypGen:  Download Link: https://www.synapse.org/Synapse:syn26376615/datasets/
```sh
    import synapseclient
    import synapseutils
    # to create token: Login(username:ruhulamin)-->Account Settings --> Create Access Token (Select Download tic mark)
    syn = synapseclient.Synapse()
    syn.login(authToken="your_auth_token")
    synapseutils.syncFromSynapse(syn, "syn26376615")
```
2. Kvasir-SEG: Download Link: https://datasets.simula.no/downloads/hyper-kvasir/hyper-kvasir-segmented-images.zip

```sh
                                      ┌──> UNET (Decoder) ──> Segmented Image (Mask)
                                      │
Input ➔ SE+PR-CNN ➔ UNET (Encoder) ───┤
                                      │
                                      └──> PD-CNN ➔ PCC ➔ Standardization ──> Classification (0/1)


```

Important Keys:
1. Multi-Task Learning (MTL) with Shared Representation
2. Alternate Training / Decoupled Multi-Task Training।



Full workflow:

```sh
[ Input Image: 256 × 256 × 3 ]
                                │
                                ▼
                       ┌─────────────────┐
                       │   SE+PR-CNN     │  (Lightweight Stem with
                       │   Stem Block    │   Pointwise Residual)
                       └─────────────────┘
                                │
                                ▼
         ┌──────────────────────────────────────────────┐
         │              SHARED ENCODER                  │
         │  (Multi-scale Depthwise-Separable Convolutions│
         │   with Squeeze-and-Excitation / SE-Blocks)   │
         └──────────────────────────────────────────────┘
            │          │               │             │
          (c1)        (c2)           (c3)          (c4)  <── [Skip Connections]
            │          │               │             │
            ▼          ▼               ▼             ▼
         ┌──────────────────────────────────────────────┐
         │          Bottleneck: 16 × 16 × 512           │
         └──────────────────────────────────────────────┘
                                │
         ┌──────────────────────┴──────────────────────┐
         │                                             │
         ▼ (Branch A: Segmentation)                    ▼ (Branch B: Classification)
┌────────────────────────────────┐            ┌────────────────────────────────┐
│         UNet Decoder           │            │           PD-CNN Head          │
│ (Conv2DTranspose + Concatenate │            │  (Polyp-Discriminative CNN     │
│    with Skip Connections)      │            │   - 2x Separable Conv Blocks)  │
└────────────────────────────────┘            └────────────────────────────────┘
         │                                             │
         ▼                                             ▼
┌────────────────────────────────┐            ┌────────────────────────────────┐
│      Output: mask_output       │            │    Global Average Pooling      │
│  (1 Channel Sigmoid - Float32) │            └────────────────────────────────┘
└────────────────────────────────┘                             │
                                                               ▼
                                              ┌────────────────────────────────┐
                                              │           PCC Layer            │
                                              │ (Pearson Correlation Coeff.   │
                                              │  Inter-channel Signal Math)    │
                                              └────────────────────────────────┘
                                                               │
                                                               ▼
                                              ┌────────────────────────────────┐
                                              │      Batch Normalization       │
                                              │      (Z-Score Standard)        │
                                              └────────────────────────────────┘
                                                               │
                                                               ▼
                                              ┌────────────────────────────────┐
                                              │      Dense + Dropout Layers    │
                                              └────────────────────────────────┘
                                                               │
                                                               ▼
                                              ┌────────────────────────────────┐
                                              │     Output: class_output       │
                                              │    (1 Value Sigmoid - Float32) │
                                              └────────────────────────────────┘

```