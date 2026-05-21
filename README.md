
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
                                              │ (Pearson Correlation Coeff.    │
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


## 🛠️ How the Core Components of the Architecture Work

### 1. Shared Feature Extraction (Root Directory)

#### **SE + PR-CNN Stem**

At the beginning, the model receives the raw input image and passes it through this stem block. Along with standard 3×3 convolutions, a 1×1 pointwise residual connection is used to accelerate feature extraction and improve information flow.

#### **Shared Encoder**

Next, the image passes through four downsampling stages. Each stage contains your specialized **Multi-scale Depthwise-Separable Convolution** module. This module simultaneously applies:

* 3×3 convolution
* 5×5 convolution
* Dilated 3×3 convolution

As a result, the network can effectively capture both small and large spatial variations of polyps. Finally, the **SE-Block (Squeeze-and-Excitation Block)** applies channel-wise feature attention to emphasize the most informative channels.

---

### 2. The Decision Point (The Bottleneck)

At the end of the encoder, the feature map size becomes **16 × 16 × 512**. This is where the main “magic” of your thesis happens.

These 512 channels contain the core latent representations required for both:

* Segmentation
* Classification

The bottleneck acts as the shared semantic knowledge hub for the two tasks.

---

### 3. Branch-A: Segmentation Mask Generator (U-Net Decoder)

This branch takes the bottleneck feature map and gradually restores the spatial resolution using **Conv2DTranspose** (transposed convolution) or upsampling operations.

To preserve precise positional details during upsampling, **Skip Connections** from encoder stages (`c1, c2, c3, c4`) are concatenated with decoder features.

The final output layer, `mask_output`, generates a single-channel `float32` segmentation mask that accurately identifies the polyp boundary or region.

---

### 4. Branch-B: Disease Diagnostic Head (PD-CNN → PCC → CLF)

#### **PD-CNN Head**

This head directly receives bottleneck features and further refines them into more discriminative representations for classification.

#### **Global Average Pooling (GAP)**

The 2D feature maps are converted into a compact 1D feature vector of size `(128,)`.

#### **PCCLayer (Pearson Correlation Coefficient Layer)**

This is one of the most innovative parts of your thesis.

The 128-dimensional feature vector is divided into two equal groups. The layer then computes the **inter-channel linear correlation** between them using the Pearson Correlation Coefficient.

This mathematical relationship acts as a powerful signal for detecting the presence or absence of polyps.

#### **Classification Top**

The correlated features are then standardized using Batch Normalization and passed through two Dense layers with Dropout regularization.

Finally, the `class_output` layer produces a confidence score indicating whether the image is:

* Polyp Positive (1.0)
* Polyp Negative (0.0)

---

## 📊 How the Loss Engine Controls the System

In your custom training engine, `tf.GradientTape` is used to jointly optimize both branches by combining their losses:

0.5 \times loss_{seg} + 0.5 \times loss_{clf}

During backpropagation, the model updates its weights in such a way that it learns to:

* Generate accurate segmentation masks
* Perform correct polyp classification simultaneously

This can be defended in your thesis as a form of **Multi-Objective Topology Optimization**, where the network is optimized for two interdependent medical imaging objectives at the same time.
