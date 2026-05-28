```sh

                                [ Input Image (256 x 256 x 3) ]
                                                 │
                                                 ▼
                                     [ SE + PR-CNN Block Hub ]
                                                 │
                                                 ▼
                                   [ UNET Encoder Backbone ]
                                                 │
                  ┌──────────────────────────────┴──────────────────────────────┐
                  │ (Shared Feature Map Extract)                                │ (Bottleneck Pass)
                  ▼                                                             ▼
         [ UNET Decoder Path ]                                         [ PD-CNN Network ]
                  │                                                             │
                  ├──► (Block 1: 32x32)                                         ▼
                  │         │                                          [ PCC (Feature Pooling) ]
                  │         ├──► [Intermediate Pred 32x32]                      │
                  │         │               │                                   ▼
                  │         │               ▼ (Calculate Error)            [ Standardization ]
                  │         │       [ FFN 1 + Dropout ]                         │
                  │         │               │                                   ▼
                  │         ▼ (Inject Matrix)                                   ▼
                  ├──► [ Refined Block 1 ]                             [ Classification Dense ]
                  │         │                                                   │
                  ├──► (Block 2: 64x64)                                         ▼
                  │         │                                          [ clf_output ] (🎯 0/1 Diagnosis)
                  │         ├──► [Intermediate Pred 64x64]
                  │         │               │
                  │         │               ▼ (Calculate Error)
                  │         │       [ FFN 2 + Dropout ]
                  │         │               │
                  │         ▼ (Inject Matrix)
                  ├──► [ Refined Block 2 ]
                  │         │
                  ├──► (Block 3: 128x128) ──► [ FFN 3 + Dropout Loop ] ──► [ Refined Block 3 ]
                  │                                                               │
                  ▼                                                               ▼
         [ Final Output Conv (256x256) ] ─────────────────────────────────────────┘
                  │
                  ▼
         [ seg_output ] (🎯 Outperforming Pure Mask)

```