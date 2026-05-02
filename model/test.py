import cv2
import numpy as np
import tensorflow as tf
from train import extract_frames, TransformerLayer

CLASSES    = sorted(["HighJump", "PlayingPiano", "Skiing"])
MODEL_PATH = "model/efficient_net_model_full.h5"
N_FRAMES   = 10
IMG_SIZE   = 224

def predict_video(video_path):
    # Load full model (architecture + weights)
    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"TransformerLayer": TransformerLayer}
    )

    frames = extract_frames(video_path, n_frames=N_FRAMES, img_size=IMG_SIZE)
    if frames is None:
        print("❌ Could not read video.")
        return

    input_tensor = np.expand_dims(frames, axis=0)  # (1, 10, 224, 224, 3)
    predictions  = model.predict(input_tensor)[0]   # (3,)

    top_idx      = np.argmax(predictions)
    print(f"\n✅ Predicted : {CLASSES[top_idx]}")
    print(f"   Confidence: {predictions[top_idx]*100:.1f}%")
    print(f"\n   Top 3:")
    for i in np.argsort(predictions)[::-1]:
        print(f"     {CLASSES[i]:<20} {predictions[i]*100:.1f}%")

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) < 2:
    #     print("Usage: python test.py path/to/video.avi")
    # else:
    VIDEO_PATH   = r"D:\Video Classification\UCF-101\UCF-101\HighJump\v_HighJump_g22_c03.avi"
    predict_video(VIDEO_PATH)

