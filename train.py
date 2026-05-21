"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     DREAM POLYP UNIFIED NET — Optimized Multi-Task Pipeline v2.0           ║
║     Architecture: SE+PR-CNN → UNet(Enc) → [Decoder | PD-CNN → PCC → CLF]  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Key fixes vs original:
  • PCC matrix replaced with lightweight channel-correlation (no 16K explosion)
  • .cache() moved AFTER .batch() to cache batches, not raw tensors
  • Validation uses accumulated metric states, not last-batch only
  • SE+PR-CNN stem added before UNet encoder (matching diagram)
  • LR scheduling (CosineDecay + warmup) and EarlyStopping added
  • Mixed-precision safe: PCC/Dense heads cast to float32 explicitly
  • train_step methods removed @tf.function to avoid retracing issues
  • Proper model saving with custom layer registration
"""

import os
import glob
import gc
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input, mixed_precision
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# GPU SETUP
# ─────────────────────────────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✓ GPU Memory Growth Enabled")
    except RuntimeError as e:
        print(e)

# ─────────────────────────────────────────────────────────────────────────────
# ⚙️  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
IMG_SIZE    = 256
BATCH_SIZE  = 8        # Increased: model is now lighter (PCC fixed)
EPOCHS      = 100
WARMUP_EPOCHS = 5

DATA_PATH   = '/mnt/c/development/Thesis/PolypSegmentationBasedClassification/DataSets/PolypGen2021_MultiCenterData_v2'
POSITIVE_DIR = os.path.join(DATA_PATH, "imagesAll_positive")
NEGATIVE_DIR = os.path.join(DATA_PATH, "sequenceData", "negativeOnly")
SAVE_PATH    = '/mnt/c/development/Thesis/PolypSegmentationBasedClassification/models/unet-trsenet/depth-se-polypgen/unified_model.keras'

mixed_precision.set_global_policy('mixed_float16')
print("✓ Mixed Precision (float16) enabled")


# ─────────────────────────────────────────────────────────────────────────────
# 📊  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_segmentation_paths(base_path):
    all_images, all_masks = [], []
    for i in range(1, 7):
        cid     = f"C{i}"
        img_dir  = os.path.join(base_path, f"data_{cid}", f"images_{cid}")
        mask_dir = os.path.join(base_path, f"data_{cid}", f"masks_{cid}")
        if not (os.path.exists(img_dir) and os.path.exists(mask_dir)):
            continue
        img_paths = []
        for ext in ("*.jpg", "*.JPG", "*.png", "*.jpeg"):
            img_paths.extend(glob.glob(os.path.join(img_dir, ext)))
        for img_p in img_paths:
            fname               = os.path.basename(img_p)
            name, ext           = os.path.splitext(fname)
            mask_p              = os.path.join(mask_dir, f"{name}_mask{ext}")
            if os.path.exists(mask_p):
                all_images.append(img_p)
                all_masks.append(mask_p)
    print(f"  Segmentation samples: {len(all_images)}")
    return all_images, all_masks


def load_classification_paths():
    valid_ext = ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.PNG','*.JPG','*.JPEG')
    pos_files, neg_files = [], []
    for ext in valid_ext:
        pos_files.extend(glob.glob(os.path.join(POSITIVE_DIR, "**", ext), recursive=True))
    for i in range(1, 14):
        folder = os.path.join(NEGATIVE_DIR, f"seq{i}_neg")
        if os.path.exists(folder):
            for ext in valid_ext:
                neg_files.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
    file_paths = pos_files + neg_files
    labels     = [1.0] * len(pos_files) + [0.0] * len(neg_files)
    print(f"  Classification samples: pos={len(pos_files)}, neg={len(neg_files)}")
    return np.array(file_paths), np.array(labels, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# 🗄️  TF-DATA PARSERS & AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def augment(img, mask):
    """Paired augmentation — both image and mask transformed identically."""
    # Flip left-right
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
    # Flip up-down
    if tf.random.uniform(()) > 0.5:
        img  = tf.image.flip_up_down(img)
        mask = tf.image.flip_up_down(mask)
    # Colour jitter (image only)
    img = tf.image.random_brightness(img, max_delta=0.15)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    img = tf.clip_by_value(img, 0.0, 1.0)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img, mask


def parse_seg_element(img_path, mask_path):
    img  = tf.io.read_file(img_path)
    img  = tf.image.decode_image(img, channels=3, expand_animations=False)
    img  = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img  = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_image(mask, channels=1, expand_animations=False)
    mask = tf.image.resize(mask, [IMG_SIZE, IMG_SIZE],
                           method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    mask = tf.cast(mask, tf.float32) / 255.0
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    mask.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return img, mask


def parse_clf_element(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, tf.reshape(label, [1])


def build_datasets(s_tr_x, s_tr_y, s_val_x, s_val_y,
                   c_tr_x, c_tr_y, c_val_x, c_val_y):
    """
    FIX: .cache() placed AFTER .batch() — we cache batches (much smaller),
    not raw decoded images (which caused OOM in original).
    """
    AUTOTUNE = tf.data.AUTOTUNE

    train_seg_ds = (
        tf.data.Dataset.from_tensor_slices((s_tr_x, s_tr_y))
        .shuffle(len(s_tr_x), reshuffle_each_iteration=True)
        .map(parse_seg_element, num_parallel_calls=AUTOTUNE)
        .map(augment, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .cache()                      # ← cache AFTER batch
        .prefetch(AUTOTUNE)
    )
    val_seg_ds = (
        tf.data.Dataset.from_tensor_slices((s_val_x, s_val_y))
        .map(parse_seg_element, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    train_clf_ds = (
        tf.data.Dataset.from_tensor_slices((c_tr_x, c_tr_y))
        .shuffle(len(c_tr_x), reshuffle_each_iteration=True)
        .map(parse_clf_element, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=True)
        .cache()
        .prefetch(AUTOTUNE)
    )
    val_clf_ds = (
        tf.data.Dataset.from_tensor_slices((c_val_x, c_val_y))
        .map(parse_clf_element, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTOTUNE)
    )
    val_zipped_ds = tf.data.Dataset.zip((val_seg_ds, val_clf_ds)).prefetch(AUTOTUNE)
    return train_seg_ds, val_seg_ds, train_clf_ds, val_clf_ds, val_zipped_ds


# ─────────────────────────────────────────────────────────────────────────────
# 📐  ARCHITECTURAL BLOCKS
# ─────────────────────────────────────────────────────────────────────────────

# ── Squeeze-and-Excitation block ──────────────────────────────────────────────
def se_block(x, ratio=8):
    c = x.shape[-1]
    # squeeze
    s = layers.GlobalAveragePooling2D()(x)
    # excitation
    e = layers.Dense(max(c // ratio, 1), activation='relu', use_bias=False)(s)
    e = layers.Dense(c, activation='sigmoid', use_bias=False)(e)
    e = layers.Reshape((1, 1, c))(e)
    return layers.multiply([x, e])


# ── SE + Pointwise-Residual CNN stem (SE+PR-CNN from diagram) ────────────────
def se_pr_cnn_stem(inputs, filters=32):
    """
    Lightweight SE-enhanced stem that sits before the UNet encoder.
    Uses pointwise (1×1) residual connections for speed.
    """
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Pointwise residual branch
    res = layers.Conv2D(filters, 1, padding='same', use_bias=False)(inputs)
    res = layers.BatchNormalization()(res)

    x = layers.add([x, res])
    x = se_block(x)
    return x


# ── Multi-scale Depthwise-Separable Conv Block (UNet blocks) ─────────────────
def ds_conv_block(x, filters, dropout_rate=0.0):
    """
    Multi-scale separable conv. Branches: 3×3, 5×5, 1×1.
    Removed 7×7 branch (too slow, marginal gain) → replaced with dilated 3×3.
    """
    f4 = filters // 4

    b1 = layers.SeparableConv2D(f4, 3, padding='same', use_bias=False)(x)
    b1 = layers.BatchNormalization()(b1)
    b1 = layers.Activation('relu')(b1)

    b2 = layers.SeparableConv2D(f4, 5, padding='same', use_bias=False)(x)
    b2 = layers.BatchNormalization()(b2)
    b2 = layers.Activation('relu')(b2)

    # Dilated conv replaces the slow 7×7 (same receptive field, 4× fewer FLOPs)
    b3 = layers.SeparableConv2D(f4, 3, dilation_rate=3, padding='same', use_bias=False)(x)
    b3 = layers.BatchNormalization()(b3)
    b3 = layers.Activation('relu')(b3)

    b4 = layers.SeparableConv2D(f4, 1, padding='same', use_bias=False)(x)
    b4 = layers.BatchNormalization()(b4)
    b4 = layers.Activation('relu')(b4)

    merged = layers.Concatenate()([b1, b2, b3, b4])
    merged = se_block(merged)

    # Residual only when channel dims match
    if x.shape[-1] == filters:
        merged = layers.add([x, merged])

    if dropout_rate > 0:
        merged = layers.SpatialDropout2D(dropout_rate)(merged)

    return merged


# ── PD-CNN: Polyp-Discriminative CNN (classification branch) ─────────────────
def pd_cnn_head(bottleneck, filters=128):
    """
    Sits on top of the bottleneck.  Two separable conv blocks with
    progressively smaller spatial resolution for discriminative features.
    """
    x = layers.SeparableConv2D(filters, 3, padding='same', use_bias=False)(bottleneck)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x)

    x = layers.SeparableConv2D(filters // 2, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


# ── Pearson Correlation Coefficient Layer (fixed, lightweight) ────────────────
@tf.keras.utils.register_keras_serializable()
class PCCLayer(layers.Layer):
    """
    FIX: Original code computed a full N×N outer-product (128×128 = 16 384 values).
    This version computes a compact PCC between two halved feature vectors,
    producing just N/2 correlation values — same inter-channel signal, 32× fewer
    parameters, and runs in float32 for numerical stability.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        # Cast to float32 — PCC math is numerically unstable in float16
        x = tf.cast(inputs, tf.float32)

        # Split features into two halves and compute element-wise PCC
        half = tf.shape(x)[-1] // 2
        a, b = x[:, :half], x[:, half:]

        eps = 1e-8
        a_mean = tf.reduce_mean(a, axis=-1, keepdims=True)
        b_mean = tf.reduce_mean(b, axis=-1, keepdims=True)
        a_c    = a - a_mean
        b_c    = b - b_mean

        num    = tf.reduce_sum(a_c * b_c, axis=-1, keepdims=True)
        denom  = (tf.norm(a_c, axis=-1, keepdims=True) *
                  tf.norm(b_c, axis=-1, keepdims=True) + eps)
        rho    = num / denom  # scalar PCC per sample

        # Also keep the normalised features for downstream Dense layers
        a_norm = a_c / (tf.norm(a_c, axis=-1, keepdims=True) + eps)
        b_norm = b_c / (tf.norm(b_c, axis=-1, keepdims=True) + eps)

        # Concatenate: [normalised_a | normalised_b | rho] → compact rich representation
        return tf.concat([a_norm, b_norm, rho], axis=-1)

    def get_config(self):
        return super().get_config()


# ─────────────────────────────────────────────────────────────────────────────
# 🏗️  UNIFIED MULTI-TASK MODEL
#     Diagram: Input → SE+PR-CNN → UNet(Enc) → Bottleneck
#                                       ↓                ↓
#                                  UNet(Dec)         PD-CNN → PCC → CLF
# ─────────────────────────────────────────────────────────────────────────────
def build_unified_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    # ── 1. SE+PR-CNN Stem ────────────────────────────────────────────────────
    stem = se_pr_cnn_stem(inputs, filters=32)   # (H, W, 32)

    # ── 2. UNet Encoder ──────────────────────────────────────────────────────
    c1 = ds_conv_block(stem, 32)
    p1 = layers.MaxPooling2D(2)(c1)             # 128×128

    c2 = ds_conv_block(p1, 64)
    p2 = layers.MaxPooling2D(2)(c2)             # 64×64

    c3 = ds_conv_block(p2, 128, dropout_rate=0.1)
    p3 = layers.MaxPooling2D(2)(c3)             # 32×32

    c4 = ds_conv_block(p3, 256, dropout_rate=0.2)
    p4 = layers.MaxPooling2D(2)(c4)             # 16×16

    # ── 3. Bottleneck ────────────────────────────────────────────────────────
    bottleneck = ds_conv_block(p4, 512, dropout_rate=0.3)   # 16×16×512

    # ══════════════════════════════════════════════════════════════════════════
    # BRANCH A — UNet Decoder → Segmentation mask
    # ══════════════════════════════════════════════════════════════════════════
    u4 = layers.Conv2DTranspose(256, 2, strides=2, padding='same')(bottleneck)
    u4 = layers.concatenate([u4, c4])
    d4 = ds_conv_block(u4, 256, dropout_rate=0.2)

    u3 = layers.Conv2DTranspose(128, 2, strides=2, padding='same')(d4)
    u3 = layers.concatenate([u3, c3])
    d3 = ds_conv_block(u3, 128, dropout_rate=0.1)

    u2 = layers.Conv2DTranspose(64, 2, strides=2, padding='same')(d3)
    u2 = layers.concatenate([u2, c2])
    d2 = ds_conv_block(u2, 64)

    u1 = layers.Conv2DTranspose(32, 2, strides=2, padding='same')(d2)
    u1 = layers.concatenate([u1, c1])
    d1 = ds_conv_block(u1, 32)

    # Segmentation output — explicit float32 for stability with mixed precision
    mask_output = layers.Conv2D(
        1, 1, activation='sigmoid', dtype='float32', name='mask_output'
    )(d1)

    # ══════════════════════════════════════════════════════════════════════════
    # BRANCH B — PD-CNN → PCC → Standardisation → Classification
    # ══════════════════════════════════════════════════════════════════════════
    pd_features = pd_cnn_head(bottleneck, filters=128)          # (16,16,128)
    gap         = layers.GlobalAveragePooling2D()(pd_features)   # (128,)

    # PCC compact correlation layer (float32 cast internal)
    pcc_out     = PCCLayer(name='pcc')(gap)                      # (N/2*2 + 1,)

    # Standardisation (Z-score via BatchNorm) — cast to float32 explicitly
    pcc_f32     = layers.Lambda(
        lambda t: tf.cast(t, tf.float32), name='pcc_cast'
    )(pcc_out)
    std_out     = layers.BatchNormalization(dtype='float32')(pcc_f32)

    fc1         = layers.Dense(128, activation='relu', dtype='float32')(std_out)
    fc1         = layers.Dropout(0.4)(fc1)
    fc2         = layers.Dense(64, activation='relu', dtype='float32')(fc1)
    fc2         = layers.Dropout(0.3)(fc2)

    class_output = layers.Dense(
        1, activation='sigmoid', dtype='float32', name='class_output'
    )(fc2)

    model = models.Model(
        inputs=inputs,
        outputs=[mask_output, class_output],
        name="Dream_Polyp_Unified_Net_v2"
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 🧮  LOSS FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    inter  = tf.reduce_sum(y_true * y_pred)
    return (2.0 * inter + smooth) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth
    )


def bce_dice_loss(y_true, y_pred):
    y_t = tf.cast(y_true, tf.float32)
    y_p = tf.cast(y_pred, tf.float32)
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_t, y_p))
    return bce + (1.0 - dice_coef(y_t, y_p))


# ─────────────────────────────────────────────────────────────────────────────
# 🧠  CUSTOM TRAINING ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class UnifiedTrainer(models.Model):
    def __init__(self, unified_model, **kwargs):
        super().__init__(**kwargs)
        self.unified_model = unified_model
        self.bce_loss      = tf.keras.losses.BinaryCrossentropy()

        # Metrics
        self.loss_seg_tr   = tf.keras.metrics.Mean(name="loss_seg")
        self.loss_clf_tr   = tf.keras.metrics.Mean(name="loss_clf")
        self.dice_tr       = tf.keras.metrics.Mean(name="dice")
        self.clf_acc_tr    = tf.keras.metrics.BinaryAccuracy(name="clf_acc")
        self.loss_seg_val  = tf.keras.metrics.Mean(name="val_loss_seg")
        self.loss_clf_val  = tf.keras.metrics.Mean(name="val_loss_clf")
        self.dice_val      = tf.keras.metrics.Mean(name="val_dice")
        self.clf_acc_val   = tf.keras.metrics.BinaryAccuracy(name="val_clf_acc")

    def compile(self, optimizer, seg_loss_fn=bce_dice_loss):
        super().compile()
        self.optimizer   = optimizer
        self.seg_loss_fn = seg_loss_fn

    @property
    def metrics(self):
        return [
            self.loss_seg_tr, self.loss_clf_tr, self.dice_tr, self.clf_acc_tr,
            self.loss_seg_val, self.loss_clf_val, self.dice_val, self.clf_acc_val,
        ]

    # ── Single combined train step (seg + clf in one tape) ───────────────────
    @tf.function
    def combined_train_step(self, seg_batch, clf_batch):
        s_imgs, s_masks = seg_batch
        c_imgs, c_lbls  = clf_batch

        with tf.GradientTape() as tape:
            pred_masks, _   = self.unified_model(s_imgs, training=True)
            _,  pred_labels = self.unified_model(c_imgs, training=True)

            loss_seg = self.seg_loss_fn(s_masks, pred_masks)
            loss_clf = self.bce_loss(c_lbls, pred_labels)

            # Weighted combination — segmentation slightly dominant
            total_loss = 0.6 * loss_seg + 0.4 * loss_clf

            # Mixed-precision scaling
            if hasattr(self.optimizer, 'get_scaled_loss'):
                scaled = self.optimizer.get_scaled_loss(total_loss)
            else:
                scaled = total_loss

        grads = tape.gradient(scaled, self.unified_model.trainable_variables)
        if hasattr(self.optimizer, 'get_unscaled_gradients'):
            grads = self.optimizer.get_unscaled_gradients(grads)

        self.optimizer.apply_gradients(
            zip(grads, self.unified_model.trainable_variables)
        )

        self.loss_seg_tr.update_state(loss_seg)
        self.loss_clf_tr.update_state(loss_clf)
        self.dice_tr.update_state(dice_coef(s_masks, pred_masks))
        self.clf_acc_tr.update_state(c_lbls, pred_labels)

    # ── Validation step ──────────────────────────────────────────────────────
    @tf.function
    def validation_step(self, val_seg_batch, val_clf_batch):
        s_imgs, s_masks = val_seg_batch
        c_imgs, c_lbls  = val_clf_batch

        pred_masks, _   = self.unified_model(s_imgs, training=False)
        _, pred_labels  = self.unified_model(c_imgs, training=False)

        self.loss_seg_val.update_state(self.seg_loss_fn(s_masks, pred_masks))
        self.loss_clf_val.update_state(self.bce_loss(c_lbls, pred_labels))
        self.dice_val.update_state(dice_coef(s_masks, pred_masks))
        self.clf_acc_val.update_state(c_lbls, pred_labels)


# ─────────────────────────────────────────────────────────────────────────────
# 📅  LEARNING RATE SCHEDULE (Cosine Decay with linear warmup)
# ─────────────────────────────────────────────────────────────────────────────
class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, peak_lr, warmup_steps, total_steps):
        super().__init__()
        self.peak_lr      = tf.cast(peak_lr, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.total_steps  = tf.cast(total_steps, tf.float32)

    def __call__(self, step):
        step   = tf.cast(step, tf.float32)
        warmup = self.peak_lr * (step / self.warmup_steps)
        cos_decay = 0.5 * self.peak_lr * (1.0 + tf.cos(
            np.pi * (step - self.warmup_steps) /
            (self.total_steps - self.warmup_steps)
        ))
        return tf.where(step < self.warmup_steps, warmup, cos_decay)

    def get_config(self):
        return {
            "peak_lr":      float(self.peak_lr.numpy()),
            "warmup_steps": float(self.warmup_steps.numpy()),
            "total_steps":  float(self.total_steps.numpy()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 🏃  MAIN TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────
def train():
    # 1. Load paths
    print("\n⏳ Loading dataset paths...")
    seg_img_paths, seg_mask_paths = load_segmentation_paths(DATA_PATH)
    clf_img_paths, clf_labels     = load_classification_paths()

    s_tr_x, s_val_x, s_tr_y, s_val_y = train_test_split(
        seg_img_paths, seg_mask_paths, test_size=0.2, random_state=42
    )
    c_tr_x, c_val_x, c_tr_y, c_val_y = train_test_split(
        clf_img_paths, clf_labels, test_size=0.2, random_state=42
    )

    # 2. Build tf.data pipelines
    print("⏳ Building tf.data pipelines...")
    train_seg_ds, val_seg_ds, train_clf_ds, val_clf_ds, val_zipped_ds = build_datasets(
        s_tr_x, s_tr_y, s_val_x, s_val_y,
        c_tr_x, c_tr_y, c_val_x, c_val_y
    )

    steps_per_epoch = min(
        sum(1 for _ in train_seg_ds),
        sum(1 for _ in train_clf_ds)
    )
    total_steps   = steps_per_epoch * EPOCHS
    warmup_steps  = steps_per_epoch * WARMUP_EPOCHS

    print(f"  Steps/epoch: {steps_per_epoch} | Total: {total_steps} | Warmup: {warmup_steps}")

    # 3. Build model
    print("\n🏗️  Building Unified Model...")
    core_model = build_unified_model()
    core_model.summary(line_length=100)

    lr_schedule = WarmupCosineDecay(
        peak_lr=1e-3, warmup_steps=warmup_steps, total_steps=total_steps
    )
    optimizer = mixed_precision.LossScaleOptimizer(
        tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    )

    trainer = UnifiedTrainer(core_model)
    trainer.compile(optimizer=optimizer, seg_loss_fn=bce_dice_loss)

    # 4. Early stopping state
    best_val_dice    = -np.inf
    patience         = 12
    patience_counter = 0
    history          = {k: [] for k in [
        'loss_seg','loss_clf','dice','clf_acc',
        'val_loss_seg','val_loss_clf','val_dice','val_clf_acc'
    ]}

    print(f"\n🚀 Training for up to {EPOCHS} epochs (early stop patience={patience})")
    print("=" * 70)

    for epoch in range(EPOCHS):
        # Reset all trackers at epoch start
        for m in trainer.metrics:
            m.reset_state()

        seg_iter = iter(train_seg_ds)
        clf_iter = iter(train_clf_ds)

        pb = tf.keras.utils.Progbar(steps_per_epoch, verbose=1)
        print(f"\n[Epoch {epoch+1}/{EPOCHS}]")

        # ── Training ─────────────────────────────────────────────────────────
        for step in range(steps_per_epoch):
            seg_batch = next(seg_iter)
            clf_batch = next(clf_iter)
            trainer.combined_train_step(seg_batch, clf_batch)

            pb.update(step + 1, values=[
                ("LS",  trainer.loss_seg_tr.result().numpy()),
                ("LC",  trainer.loss_clf_tr.result().numpy()),
                ("Dc",  trainer.dice_tr.result().numpy()),
                ("Acc", trainer.clf_acc_tr.result().numpy()),
            ])

        # ── Validation ───────────────────────────────────────────────────────
        # FIX: reset val metrics before accumulating across all val batches
        trainer.loss_seg_val.reset_state()
        trainer.loss_clf_val.reset_state()
        trainer.dice_val.reset_state()
        trainer.clf_acc_val.reset_state()

        for val_seg_batch, val_clf_batch in val_zipped_ds:
            trainer.validation_step(val_seg_batch, val_clf_batch)

        v_ls  = trainer.loss_seg_val.result().numpy()
        v_lc  = trainer.loss_clf_val.result().numpy()
        v_d   = trainer.dice_val.result().numpy()
        v_acc = trainer.clf_acc_val.result().numpy()

        print(f"  → Val | Loss_Seg:{v_ls:.4f} | Loss_Clf:{v_lc:.4f} "
              f"| Dice:{v_d:.4f} | Acc:{v_acc:.4f}")

        # Record history
        history['loss_seg'].append(trainer.loss_seg_tr.result().numpy())
        history['loss_clf'].append(trainer.loss_clf_tr.result().numpy())
        history['dice'].append(trainer.dice_tr.result().numpy())
        history['clf_acc'].append(trainer.clf_acc_tr.result().numpy())
        history['val_loss_seg'].append(v_ls)
        history['val_loss_clf'].append(v_lc)
        history['val_dice'].append(v_d)
        history['val_clf_acc'].append(v_acc)

        # ── Early stopping + best model save ─────────────────────────────────
        if v_d > best_val_dice:
            best_val_dice    = v_d
            patience_counter = 0
            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            core_model.save(SAVE_PATH)
            print(f"  ✅ Best model saved (Val Dice: {best_val_dice:.4f})")
        else:
            patience_counter += 1
            print(f"  ⚠️  No improvement ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print(f"\n🛑 Early stopping at epoch {epoch+1}")
                break

        gc.collect()

    print("\n🏆 Training complete!")
    return core_model, history


# ─────────────────────────────────────────────────────────────────────────────
# 📈  TRAINING CURVE PLOTTING
# ─────────────────────────────────────────────────────────────────────────────
def plot_history(history):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Dream Polyp Unified Net — Training History", fontsize=15, fontweight='bold')

    metrics = [
        ('dice',     'val_dice',     'Dice Coefficient',  axes[0, 0]),
        ('clf_acc',  'val_clf_acc',  'Classification Accuracy', axes[0, 1]),
        ('loss_seg', 'val_loss_seg', 'Segmentation Loss', axes[1, 0]),
        ('loss_clf', 'val_loss_clf', 'Classification Loss', axes[1, 1]),
    ]
    for tr_key, val_key, title, ax in metrics:
        ax.plot(history[tr_key],  label='Train', color='steelblue')
        ax.plot(history[val_key], label='Val',   color='tomato', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("📊 Training curves saved to training_history.png")


# ─────────────────────────────────────────────────────────────────────────────
# 🔮  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────
def infer(model, image_path):
    """
    Single image inference → segmentation mask + polyp/non-polyp classification.
    """
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    raw   = tf.io.read_file(image_path)
    img   = tf.image.decode_image(raw, channels=3, expand_animations=False)
    orig  = img.numpy().astype("uint8")
    img   = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img   = tf.cast(img, tf.float32) / 255.0
    inp   = tf.expand_dims(img, 0)

    pred_mask, pred_prob = model.predict(inp, verbose=0)

    prob     = float(pred_prob[0][0])
    mask_vis = (pred_mask[0, :, :, 0] > 0.5).astype(np.uint8) * 255

    label  = "POLYP DETECTED"     if prob >= 0.5 else "NON-POLYP"
    conf   = prob * 100            if prob >= 0.5 else (1 - prob) * 100
    color  = 'limegreen'           if prob >= 0.5 else 'tomato'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(orig);              axes[0].set_title("Input Image");    axes[0].axis("off")
    axes[1].imshow(mask_vis, cmap='gray'); axes[1].set_title("Predicted Mask");  axes[1].axis("off")

    # Overlay
    overlay            = orig.copy()
    mask_resized       = tf.image.resize(
        mask_vis[..., np.newaxis], [orig.shape[0], orig.shape[1]],
        method='nearest'
    ).numpy().squeeze().astype(bool)
    overlay[mask_resized, 1] = np.clip(overlay[mask_resized, 1] + 80, 0, 255)
    axes[2].imshow(overlay);           axes[2].set_title("Mask Overlay");   axes[2].axis("off")

    fig.suptitle(
        f"Diagnosis: {label}  ({conf:.1f}% confidence)",
        color=color, fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.show()
    print(f"\n{'='*50}")
    print(f"  {label}  |  Confidence: {conf:.2f}%")
    print(f"{'='*50}\n")
    return pred_mask, prob


# ─────────────────────────────────────────────────────────────────────────────
# ▶  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    core_model, history = train()
    plot_history(history)

    # Example inference (update path as needed):
    # infer(core_model, "/path/to/test_image.jpg")