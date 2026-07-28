import sys
import os
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input, mixed_precision
from sklearn.model_selection import train_test_split
import cv2
import keras
print("library loaded successfully")
print("*"*100)
import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
import gc
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
import tensorflow as tf

# 🟢 1. GPU চেক এবং মেমোরি গ্রোথ সেট করা (ফাইলের একদম উপরে থাকবে)
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print(f"✅ GPU Active: {physical_devices[0]}", file=sys.stderr)
    except RuntimeError as e:
        print(e, file=sys.stderr)
else:
    print("⚠️ Warning: No GPU found! Running on CPU.", file=sys.stderr)
    
@tf.keras.utils.register_keras_serializable()
class KerasPCCLayer(layers.Layer):
    """Natively computes spatial Pearson Correlation Coefficients within the network graph."""
    def __init__(self, **kwargs):
        super(KerasPCCLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # We used epsilon sothat standard deviation value cannot be zero, otherwise code will crash
        epsilon = 1e-8
        # finds the average standard deviaiton
        mean = tf.reduce_mean(inputs, axis=-1, keepdims=True)
        std = tf.math.reduce_std(inputs, axis=-1, keepdims=True) + epsilon
        # data normalization or (Z-Score)
        norm_inputs = (inputs - mean) / std
        pcc_matrix = tf.matmul(tf.expand_dims(norm_inputs, -1), tf.expand_dims(norm_inputs, 1))
        return layers.Flatten()(pcc_matrix)
@keras.saving.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)
def bce_dice_loss(y_true, y_pred):
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    return bce + dice_loss(y_true, y_pred)
def focal_tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Focal Tversky Loss for imbalanced medical image segmentation.
    alpha: controls penalty for False Negatives (FN) -> keeping it higher (0.7) focuses on small polyps.
    beta: controls penalty for False Positives (FP).
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # Calculate True Positives, False Negatives, and False Positives
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    
    # Tversky Index
    smooth = 1e-6
    tversky_index = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    
    # Focal Tversky Loss
    return tf.math.pow((1.0 - tversky_index), gamma)
@keras.saving.register_keras_serializable()
def iou_metric(y_true, y_pred):
    y_true_f = tf.cast(y_true, tf.float32)
    y_pred_f = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
    return (intersection + 1.0) / (union + 1.0)
@keras.saving.register_keras_serializable()
def tnr_metric(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    true_negatives = tf.reduce_sum(tf.cast((1.0 - y_true_f) * (1.0 - y_pred_f), tf.float32))
    false_positives = tf.reduce_sum(tf.cast((1.0 - y_true_f) * y_pred_f, tf.float32))
    
    # TNR = TN / (TN + FP)
    return (true_negatives + 1e-7) / (true_negatives + false_positives + 1e-7)
@keras.saving.register_keras_serializable()
def mean_iou_metric(y_true, y_pred):
    # Flatten the matrices to 1D tensors to calculate pixel-level metrics
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # Apply hard thresholding (Same as your F2 and Dice implementation)
    y_pred_f = tf.cast(y_pred_f > 0.5, tf.float32)
    
    # Calculate basic confusion matrix components for the positive class (Polyp)
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))
    
    # Calculate True Negatives (TN) which represents correctly classified background pixels
    tn = tf.reduce_sum((1.0 - y_true_f) * (1.0 - y_pred_f))

    # ─── 🎯 STEP 1: CALCULATE IoU FOR POLYP CLASS (Positive Class) ───
    polyp_iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
    
    # ─── 🌍 STEP 2: CALCULATE IoU FOR BACKGROUND CLASS (Negative Class) ───
    # For background: TN is True Positive, FN becomes False Positive, and FP becomes False Negative
    bg_iou = (tn + 1e-7) / (tn + fn + fp + 1e-7)
    
    # ─── 📊 STEP 3: COMPUTE THE MEAN (AVERAGE) IoU OF BOTH CLASSES ───
    mIoU = (polyp_iou + bg_iou) / 2.0
    
    return mIoU
@keras.saving.register_keras_serializable()
def f2_segmentation(y_true, y_pred):
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    y_pred_f = tf.cast(y_pred_f > 0.5, tf.float32)
    
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1.0 - y_pred_f))

    numerator = 5.0 * tp + 1e-7
    denominator = 5.0 * tp + 4.0 * fn + fp + 1e-7
    
    return numerator / denominator
@keras.saving.register_keras_serializable()
def precision_segmentation(y_true, y_pred):
    # Flatten the matrices to 1D tensors to calculate pixel-level metrics
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    # Apply hard thresholding (Same as your F2 and mIoU implementation)
    y_pred_f = tf.cast(y_pred_f > 0.5, tf.float32)
    
    # Calculate True Positives (TP) and False Positives (FP)
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1.0 - y_true_f) * y_pred_f)
    
    # Precision Formula: TP / (TP + FP)
    # Adding a tiny epsilon (1e-7) to completely avoid zero division errors
    numerator = tp + 1e-7
    denominator = tp + fp + 1e-7
    
    return numerator / denominator
# ─────────────────────────────────────────────────────────────────────────────
# ⚙️ GLOBAL INFERENCE CONFIGURATIONS & ELM ARTIFACTS LINKING
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE = 256

class NumpyFeatureSelector:
    """Stand-in for sklearn's SelectKBest at inference time — all we need is
    the column indices it kept, not the fitted statistics."""
    def __init__(self, indices):
        self.indices = np.asarray(indices, dtype=np.int32)

    def transform(self, X):
        return X[:, self.indices]


def load_elm_artifacts(path):
    data = np.load(path)
    ensemble_weights = [data['ensemble_weights'][i] for i in range(data['ensemble_weights'].shape[0])]

    scaler = RobustScaler()
    scaler.center_ = data['scaler_center_']
    scaler.scale_ = data['scaler_scale_']
    scaler.n_features_in_ = scaler.center_.shape[0]

    selector = NumpyFeatureSelector(data['selector_support'])

    return data['W_input'], data['b_input'], ensemble_weights, scaler, selector
# ─────────────────────────────────────────────────────────────────────────────
# 🛡️ DEPENDENCY PLACEHOLDERS
# ─────────────────────────────────────────────────────────────────────────────
def tnr_metric(y_true, y_pred): return tf.constant(1.0)
def f2_segmentation(y_true, y_pred): return tf.constant(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 🔮 SINGLE IMAGE SEGMENTATION VISUALIZATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
def generate_single_image_segmentation(
    model_path,
    image_path,
    img_size=256,
    output_path="./single_segmentation_comparison.png"
):
    """
    নির্দিষ্ট একটি ইমেজ পাথ এবং মডেল পাথ ইনপুট নিয়ে ৩-কলামের কমপ্যারেটিভ গ্রিড তৈরি করে।
    """
    
    # পাথগুলো চেক করা
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path not found: {image_path}")

    # ১. মডেল লোড করা
    print("⏳ Loading Multi-Task Keras Model Engine...")
    loaded_model = tf.keras.models.load_model(
        model_path,
        custom_objects={'Custom>tnr_metric': tnr_metric, 'Custom>f2_segmentation': f2_segmentation},
        compile=False
    )
    
    # Isolate the 1st output channel dedicated to raw segmentation probabilities
    seg_model = tf.keras.models.Model(inputs=loaded_model.input, outputs=loaded_model.outputs[0])
    print("✅ Segmentation Sub-Graph Node Successfully Locked.")

    # ২. নির্দিষ্ট ইমেজ লোড এবং প্রিপ্রসেস করা
    print(f"🚀 Processing image: {os.path.basename(image_path)}...")
    img = cv2.imread(image_path)
    if img is None: 
        raise ValueError(f"Could not read image from {image_path}")
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_res = cv2.resize(img_rgb, (img_size, img_size))
    inp = np.expand_dims(img_res / 255.0, axis=0).astype(np.float32)

    # ৩. মডেল বাইনারি প্রেডিকশন মাস্ক তৈরি করা
    pred_mask_tensor = seg_model.predict(inp, verbose=0)[0]
    pred_spatial = pred_mask_tensor[:, :, 0] if len(pred_mask_tensor.shape) == 3 else pred_mask_tensor
    pred_bin = (pred_spatial > 0.5).astype(np.uint8) * 255

    # ৪. ট্রান্সপারেন্ট সেগমেন্টেশন ওভারলে তৈরি (গ্রিন হাইলাইট ইফেক্ট)
    overlay_img = img_res.copy()
    green_mask = np.zeros_like(img_res)
    green_mask[:, :, 1] = pred_bin  # শুধু গ্রিন চ্যানেলে পলিফ সিগন্যাল বসানো
    cv2.addWeighted(green_mask, 0.4, overlay_img, 1.0, 0, overlay_img)

    # ৫. প্লটিং ক্যানভাস তৈরি (Single Row, 3 Columns)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    column_titles = ['Original Image', 'Predicted Mask', 'Segmentation Overlay']
    display_elements = [img_res, pred_bin, overlay_img]

    for col_idx in range(3):
        ax = axes[col_idx]
        
        if col_idx == 1:
            ax.imshow(display_elements[col_idx], cmap='gray')
        else:
            ax.imshow(display_elements[col_idx])
        
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(column_titles[col_idx], fontsize=12, fontweight='bold', pad=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=250, bbox_inches='tight')
    print(f"\n🏆 Successfully generated and saved Single Image Segmentation grid to: {output_path}")
    
    plt.show()
# 🟢 2. মডেল রান করার ব্লক (ফাইলের একদম নিচে থাকবে)
if __name__ == "__main__":
    import argparse
    import sys
    
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run Polyp Segmentation")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", required=True, help="Path to save output image")
    args = parser.parse_args()

    MODEL_PATH = "/mnt/c/development/Thesis/PolypSegmentationBasedClassification/models/SC/tr-elm/thesis_v3_sequential_v_eval_data_driven_v2.keras"
    
    try:
        # Run the function using the arguments passed by FastAPI
        generate_single_image_segmentation(
            model_path=MODEL_PATH,
            image_path=args.input,
            output_path=args.output, 
            img_size=256
        )
        print("Success")
        plt.show()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)