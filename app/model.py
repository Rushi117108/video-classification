import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (
    Layer, MultiHeadAttention, Dense, LayerNormalization,
    Flatten, LSTM, Bidirectional, TimeDistributed,
    Dropout, Rescaling, Input
)
from tensorflow.keras import Sequential, Model
from app.utils import extract_frames

# ── Classes must match exact order used during training ───────────────────────
CLASSES = sorted(["HighJump", "PlayingPiano", "Skiing"])  # ['HighJump', 'PlayingPiano', 'Skiing']

# ── Same TransformerLayer from train.py ───────────────────────────────────────
class TransformerLayer(Layer):
    def __init__(self, num_heads, feed_forward_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_heads        = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate     = dropout_rate

    def build(self, input_shape):
        self.attention    = MultiHeadAttention(num_heads=self.num_heads, key_dim=input_shape[-1])
        self.attn_dropout = Dropout(self.dropout_rate)
        self.attn_norm    = LayerNormalization(epsilon=1e-6)
        self.feed_forward = Sequential([
            Dense(self.feed_forward_dim, activation="relu"),
            Dense(input_shape[-1])
        ])
        self.ff_dropout   = Dropout(self.dropout_rate)
        self.ff_norm      = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, inputs, training=True):
        attn_out = self.attention(inputs, inputs)
        attn_out = self.attn_dropout(attn_out, training=training)
        attn_out = self.attn_norm(inputs + attn_out)
        ff_out   = self.feed_forward(attn_out)
        ff_out   = self.ff_dropout(ff_out, training=training)
        return self.ff_norm(attn_out + ff_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "feed_forward_dim": self.feed_forward_dim,
            "dropout_rate": self.dropout_rate
        })
        return config

# ── Singleton model loader ────────────────────────────────────────────────────
_model = None

def load_model():
    global _model
    if _model is None:
        print("Loading model...")
        _model = tf.keras.models.load_model(
            "model/efficient_net_model_full.h5",
            custom_objects={"TransformerLayer": TransformerLayer}
        )
        print("Model loaded successfully.")
    return _model

def predict_video(video_path: str) -> dict:
    model  = load_model()
    frames = extract_frames(video_path, n_frames=10, img_size=224)

    if frames is None:
        raise ValueError("Could not extract frames from video.")

    input_tensor = np.expand_dims(frames, axis=0)   # (1, 10, 224, 224, 3)
    predictions  = model.predict(input_tensor)[0]   # (3,)

    top_idx    = int(np.argmax(predictions))
    top3_idx   = np.argsort(predictions)[::-1][:3]
    top3       = [
        {"label": CLASSES[i], "confidence": round(float(predictions[i]), 4)}
        for i in top3_idx
    ]

    return {
        "predicted_class": CLASSES[top_idx],
        "confidence": round(float(predictions[top_idx]), 4),
        "top_3": top3
    }