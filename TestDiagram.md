```sh
[ inputs ] (Raw Image Input: 256x256x3)
           |
           v
==================================================================================================
 1. SHARED MULTI-SCALE ENCODER PATHWAY (With Integrated TR-SE Blocks)
==================================================================================================
  
  [c1 Block] ──> |──> [3x3 SeparableConv] ──> [BN+ReLU] ──| ──> [Concatenate] ──> [se_block] ──> [Identity Add Check] ──> (256x256x32) ───+
                 |──> [5x5 SeparableConv] ──> [BN+ReLU] ──|        (f4 = 8)         (Ratio=16)                                           |
                 |──> [7x7 SeparableConv] ──> [BN+ReLU] ──|                                                                              |
                 |──> [1x1 SeparableConv] ──> [BN+ReLU] ──|                                                                              |
                   |                                                                                                                     |
         [p1]      v MaxPooling2D (128x128x32)                                                                                           |
                   |                                                                                                                     |
  [c2 Block] ──> [Parallel 3x3, 5x5, 7x7, 1x1 Branches] ──> [Concat] ──> [SE Attention] ──> Residual Add ──> (128x128x64) ────────────+ |
                   |                                                                                                               | |
         [p2]      v MaxPooling2D (64x64x64)                                                                                       | |
                   |                                                                                                               | |
  [c3 Block] ──> [Parallel 3x3, 5x5, 7x7, 1x1 Branches] ──> [Concat] ──> [SE Attention] ──> SpatialDropout ──> (64x64x128) ────────+ | |
                   |                                                                                                             | | |
         [p3]      v MaxPooling2D (32x32x128)                                                                                    | | |
                   |                                                                                                             | | |
  [c4 Block] ──> [Parallel 3x3, 5x5, 7x7, 1x1 Branches] ──> [Concat] ──> [SE Attention] ──> SpatialDropout ──> (32x32x256) ──────+ | | |
                   |                                                                                                           | | | |
         [p4]      v MaxPooling2D (16x16x256)                                                                                  | | | |
                                                                                                                               | | | |
==================================================================================================                             | | | |
 2. SHARED LATENT BOTTLENECK                                                                                                   | | | |
==================================================================================================                             | | | |
  [ b Block] ──> [Parallel 3x3, 5x5, 7x7, 1x1 Branches] ──> [Concat] ──> [SE Attention] ──> SpatialDropout ──> (16x16x512)     | | | |
                   |                                                                                                           | | | |
         +---------+------------+                                                                                              | | | |
         |                      |                                                                                              | | | |
         | (To Seg Decoder)     | (To Clf Head Branch B)                                                                       | | | |
         v                      v                                                                                              | | | |
=======================================================        ========================================================        | | | |
  BRANCH A: DECODER WITH DEEP SUPERVISION                        BRANCH B: SEGMENTATION-GUIDED CLASSIFICATION HEAD             | | | |
=======================================================        ========================================================        | | | |
  [u4] Conv2DTranspose (32x32x256)                                      |                                                      | | | |
         |                                                              v                                                      | | | |
         v                                                       [pd_cnn] SeparableConv2D (16x16x32)                           | | | |
  [dec_concat_4] <──────────────────────────────────────────────────────+ (Skip c4 Feature Map Link)                           | | | |
  (Concat Shape: 32x32x512)                                             |                                                      | | | |
         |                                                              v                                                      | | | |
         v                                                       [p_pool] MaxPooling2D (8x8x32)                                | | | |
  [c5] TR-SE Block (32x32x256)                                          |                                                      | | | |
         |───────> [ds1_conv] Conv2D (1x1)                              v                                                      | | | |
         |             |                                         [clf_gap] GlobalAveragePooling2D                              | | | |
         |             v [ds1_output] UpSampling2D (8x8)                |                                                      | | | |
         |         (256x256x1 Target Aux Loss 1)                        v                                                      | | | |
         v                                                       [clf_pcc_layer] KerasPCCLayer                                 | | | |
  [u3] Conv2DTranspose (64x64x128)                                      |                                                      | | | |
         |                                                              v                                                      | | | |
         v                                                       [clf_zscore_bn] BatchNormalization                            | | | |
  [dec_concat_3] <──────────────────────────────────────────────────────+ (Skip c3 Feature Map Link)                             | | | |
  (Concat Shape: 64x64x256)                                             |                                                      | | | |
         |                                                              v                                                      | | | |
         v                                                       [clf_dense1] Dense Layer (64 Neurons)                         | | | |
  [c6] TR-SE Block (64x64x128)                                          |                                                      | | | |
         |───────> [ds2_conv] Conv2D (1x1)                              v                                                      | | | |
         |             |                                         [clf_dropout] Dropout (Rate=0.3)                              | | | |
         |             v [ds2_output] UpSampling2D (4x4)                |                                                      | | | |
         |         (256x256x1 Target Aux Loss 2)                        v (Sigmoid)                                            | | | |
         v                                                       [clf_output] Final Disease Probability                        | | | |
  [u2] Conv2DTranspose (128x128x64)                                                                                            | | |
         |                                                                                                                     | | |
         v                                                                                                                     | | |
  [dec_concat_2] <─────────────────────────────────────────────────────────────────────────────────────────────────────────────+ | |
  (Concat Shape: 128x128x128)                                                                                                    | |
         |                                                                                                                       | |
         v                                                                                                                       | |
  [c7] TR-SE Block (128x128x64)                                                                                                  | |
         |───────> [ds3_conv] Conv2D (1x1)                                                                                       | |
         |             |                                                                                                         | |
         |             v [ds3_output] UpSampling2D (2x2)                                                                         | |
         |         (256x256x1 Target Aux Loss 3)                                                                                 | |
         v                                                                                                                       | |
  [u1] Conv2DTranspose (256x256x32)                                                                                              | |
         |                                                                                                                       | |
         v                                                                                                                       | |
  [dec_concat_1] <───────────────────────────────────────────────────────────────────────────────────────────────────────────────+ |
  (Concat Shape: 256x256x64)                                                                                                       |
         |                                                                                                                         |
         v                                                                                                                         |
  [c8] TR-SE Block (256x256x32)                                                                                                    |
         |                                                                                                                         |
         v (Sigmoid)                                                                                                               v
=======================================================                                                                 ===========
  [seg_output] Final Binary Mask (256x256x1)                                                                             Outputs Graph
=======================================================                                                                 ===========

```