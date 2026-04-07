import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from scipy.ndimage import zoom
from scipy import ndimage
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import streamlit as st

# ── Page config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="MS Lesion Detection — CareBot",
    page_icon="🧠",
    layout="centered",
)


# ── Build model & load weights (cached) ────────────────────────────
@st.cache_resource
def load_model():
    def dice_coefficient(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        return (2.0 * intersection + smooth) / (
            tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth
        )

    def dice_loss(y_true, y_pred):
        return 1 - dice_coefficient(y_true, y_pred)

    def iou_metric(y_true, y_pred, smooth=1e-6):
        y_true_f = tf.keras.backend.flatten(y_true)
        y_pred_f = tf.keras.backend.flatten(y_pred)
        intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
        union = tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)

    def combined_loss(y_true, y_pred):
        return dice_loss(y_true, y_pred) + tf.keras.losses.binary_crossentropy(y_true, y_pred)

    # Build architecture
    inputs = layers.Input(shape=(256, 256, 3))
    backbone = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = True
    skip1 = backbone.get_layer("conv1_relu").output
    skip2 = backbone.get_layer("conv2_block3_out").output
    skip3 = backbone.get_layer("conv3_block4_out").output
    skip4 = backbone.get_layer("conv4_block6_out").output
    bridge = backbone.output

    up1 = layers.Conv2DTranspose(512, 2, strides=2, padding="same")(bridge)
    up1 = layers.concatenate([up1, skip4])
    up1 = layers.Conv2D(512, 3, activation="relu", padding="same")(up1)
    up1 = layers.Conv2D(512, 3, activation="relu", padding="same")(up1)

    up2 = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(up1)
    up2 = layers.concatenate([up2, skip3])
    up2 = layers.Conv2D(256, 3, activation="relu", padding="same")(up2)
    up2 = layers.Conv2D(256, 3, activation="relu", padding="same")(up2)

    up3 = layers.Conv2DTranspose(128, 2, strides=2, padding="same")(up2)
    up3 = layers.concatenate([up3, skip2])
    up3 = layers.Conv2D(128, 3, activation="relu", padding="same")(up3)
    up3 = layers.Conv2D(128, 3, activation="relu", padding="same")(up3)

    up4 = layers.Conv2DTranspose(64, 2, strides=2, padding="same")(up3)
    up4 = layers.concatenate([up4, skip1])
    up4 = layers.Conv2D(64, 3, activation="relu", padding="same")(up4)
    up4 = layers.Conv2D(64, 3, activation="relu", padding="same")(up4)

    up5 = layers.Conv2DTranspose(32, 2, strides=2, padding="same")(up4)
    up5 = layers.Conv2D(32, 3, activation="relu", padding="same")(up5)
    outputs = layers.Conv2D(1, 1, activation="sigmoid")(up5)

    model = models.Model(inputs=inputs, outputs=outputs, name="UNet-ResNet50")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss=combined_loss,
        metrics=[dice_coefficient, iou_metric, "binary_accuracy"],
    )

    model.load_weights("best_unet_resnet50.weights.h5")
    return model


model = load_model()


# ── Helper functions ────────────────────────────────────────────────
def normalize_image(image):
    image = image.astype(np.float32)
    mn, mx = image.min(), image.max()
    if mx > mn:
        image = (image - mn) / (mx - mn)
    return image


def resize_slice(slice_2d, target_shape=(256, 256)):
    if slice_2d.shape == target_shape:
        return slice_2d
    return zoom(
        slice_2d,
        [target_shape[0] / slice_2d.shape[0], target_shape[1] / slice_2d.shape[1]],
        order=1,
    )


def post_process_mask(mask, min_size=10):
    labeled_mask, num_features = ndimage.label(mask)
    for lbl in range(1, num_features + 1):
        if np.sum(labeled_mask == lbl) < min_size:
            labeled_mask[labeled_mask == lbl] = 0
    return (labeled_mask > 0).astype(np.float32)


def calculate_lesion_volume(mask):
    return float(np.sum(mask > 0)), int(np.sum(mask > 0))


def calculate_brain_volume(image):
    return float(np.sum(image > 0.1)), int(np.sum(image > 0.1))


def calculate_lesion_load(lesion_vol, brain_vol):
    if brain_vol == 0:
        return 0.0
    return (lesion_vol / brain_vol) * 100


def classify_severity(lesion_load):
    if lesion_load < 0.1:
        return "Minimal"
    elif lesion_load < 0.5:
        return "Mild"
    elif lesion_load < 1.0:
        return "Moderate"
    elif lesion_load < 2.0:
        return "Severe"
    return "Very Severe"


def create_heatmap_overlay(image, prediction_probs, alpha=0.6):
    image_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
    image_rgb = np.stack([image_norm] * 3, axis=-1)
    heatmap = cm.hot(prediction_probs)[:, :, :3]
    overlay = (1 - alpha) * image_rgb + alpha * heatmap
    return np.clip(overlay, 0, 1)


# ── Inference ───────────────────────────────────────────────────────
def run_inference(image_array):
    img = normalize_image(image_array)
    img = resize_slice(img, (256, 256))
    img_batch = np.expand_dims(
        np.repeat(np.expand_dims(img, -1), 3, axis=-1), axis=0
    )

    prediction = model.predict(img_batch, verbose=0)[0, :, :, 0]
    pred_clean = post_process_mask((prediction > 0.5).astype(np.float32))

    lesion_vol, _ = calculate_lesion_volume(pred_clean)
    brain_vol, _ = calculate_brain_volume(img)
    lesion_load = calculate_lesion_load(lesion_vol, brain_vol)
    severity = classify_severity(lesion_load)
    is_ms = bool(lesion_load >= 0.1)

    masked_prediction = np.where(prediction >= 0.5, prediction, 0.0)
    overlay = create_heatmap_overlay(img, masked_prediction, alpha=0.6)

    # Generate visualization
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(overlay)
    mask_display = np.ma.masked_where(pred_clean < 0.5, pred_clean)
    ax.imshow(mask_display, cmap="Reds", alpha=0.35)
    ax.set_axis_off()

    sm = cm.ScalarMappable(cmap="hot", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Lesion Probability", fontsize=10)

    ax.set_title(
        f"Lesion Load: {lesion_load:.2f}%  |  Severity: {severity}  |  MS: {is_ms}",
        fontsize=11,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)

    return {
        "image_buf": buf,
        "lesion_load": round(lesion_load, 4),
        "severity": severity,
        "is_MS": is_ms,
    }


# ── UI ──────────────────────────────────────────────────────────────
st.title("🧠 MS Lesion Detection")
st.markdown("**UNet-ResNet50** model for Multiple Sclerosis lesion segmentation on brain MRI")

st.divider()

uploaded = st.file_uploader(
    "Upload a brain MRI slice (grayscale image)",
    type=["png", "jpg", "jpeg"],
)

if uploaded:
    st.image(uploaded, caption="Uploaded MRI slice", use_container_width=True)

    if st.button("🔬 Analyze MRI", type="primary", use_container_width=True):
        try:
            img = Image.open(uploaded).convert("L")
            img_array = np.array(img, dtype=np.float32)

            with st.spinner("Running segmentation model... this may take a moment."):
                result = run_inference(img_array)

            st.divider()

            # Show heatmap
            st.image(result["image_buf"], caption="Lesion Heatmap Overlay", use_container_width=True)

            # Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Lesion Load", f"{result['lesion_load']:.2f}%")
            col2.metric("Severity", result["severity"])
            col3.metric("MS Detected", "Yes" if result["is_MS"] else "No")

            if result["is_MS"]:
                st.warning(
                    f"⚠️ Potential MS lesions detected — Severity: **{result['severity']}**. "
                    "Please consult a neurologist for clinical evaluation."
                )
            else:
                st.success("✅ No significant MS lesions detected in this slice.")

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
