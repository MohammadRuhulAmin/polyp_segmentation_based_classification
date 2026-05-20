
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
                                      └──> PD-CNN ➔ PCC ➔ Standardization ──> Classification (0/1


```