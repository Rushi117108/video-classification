import os
import gc
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import (
    Layer, MultiHeadAttention, Dense, LayerNormalization,
    Flatten, LSTM, Bidirectional, TimeDistributed,
    Dropout, Rescaling, Input
)
from tensorflow.keras import Sequential, Model

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR   = r"D:\Video Classification\UCF-101\UCF-101"
MODEL_PATH = "model/efficient_net_model_full.h5"
CLASSES    = sorted(["HighJump", "PlayingPiano", "Skiing"])  # sorted = consistent label order
N_FRAMES   = 10
IMG_SIZE   = 224
BATCH_SIZE = 4    # Small batch — safe for 8GB RAM
EPOCHS     = 10

print(f"Classes: {CLASSES}")
print(f"Label mapping: { {cls: i for i, cls in enumerate(CLASSES)} }")

# ── FRAME EXTRACTOR ───────────────────────────────────────────────────────────
def extract_frames(video_path, n_frames=N_FRAMES, img_size=IMG_SIZE):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return None

    indices = np.linspace(0, total - 1, n_frames, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            frame = frames[-1] if frames else np.zeros((img_size, img_size, 3), dtype=np.uint8)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (img_size, img_size))
        frames.append(frame)
    cap.release()

    frames = np.array(frames, dtype=np.float32) / 255.0
    return frames  # shape: (N_FRAMES, IMG_SIZE, IMG_SIZE, 3)

# ── LOAD FILE PATHS & LABELS (not videos, just paths) ────────────────────────
def get_file_paths_and_labels(data_dir, classes):
    file_paths, labels = [], []
    for label_idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.exists(cls_dir):
            print(f"WARNING: folder not found → {cls_dir}")
            continue
        for fname in os.listdir(cls_dir):
            if fname.endswith(('.avi', '.mp4')):
                file_paths.append(os.path.join(cls_dir, fname))
                labels.append(label_idx)
    print(f"Total videos found: {len(file_paths)}")
    return file_paths, labels

# ── DATA GENERATOR (memory safe) ─────────────────────────────────────────────
class VideoGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size=BATCH_SIZE, shuffle=True):
        self.file_paths = file_paths
        self.labels     = labels
        self.batch_size = batch_size
        self.shuffle    = shuffle
        self.indices    = np.arange(len(file_paths))
        if shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.file_paths) // self.batch_size)

    def __getitem__(self, batch_idx):
        batch_indices = self.indices[batch_idx * self.batch_size:(batch_idx + 1) * self.batch_size]
        X, y = [], []
        for i in batch_indices:
            frames = extract_frames(self.file_paths[i])
            if frames is not None:
                X.append(frames)
                y.append(self.labels[i])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# ── CUSTOM TRANSFORMER LAYER ──────────────────────────────────────────────────
class TransformerLayer(Layer):
    def __init__(self, num_heads, feed_forward_dim, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.num_heads       = num_heads
        self.feed_forward_dim = feed_forward_dim
        self.dropout_rate    = dropout_rate

    def build(self, input_shape):
        self.attention              = MultiHeadAttention(num_heads=self.num_heads, key_dim=input_shape[-1])
        self.attention_dropout      = Dropout(self.dropout_rate)
        self.attention_norm         = LayerNormalization(epsilon=1e-6)
        self.feed_forward           = Sequential([
            Dense(self.feed_forward_dim, activation="relu"),
            Dense(input_shape[-1])
        ])
        self.ff_dropout             = Dropout(self.dropout_rate)
        self.ff_norm                = LayerNormalization(epsilon=1e-6)
        super().build(input_shape)

    def call(self, inputs, training=True):
        attn_out = self.attention(inputs, inputs)
        attn_out = self.attention_dropout(attn_out, training=training)
        attn_out = self.attention_norm(inputs + attn_out)
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

# ── BUILD MODEL ───────────────────────────────────────────────────────────────
def build_model(num_classes=3):
    net = tf.keras.applications.EfficientNetB0(include_top=False)
    net.trainable = False  # Freeze backbone — saves memory & trains faster

    inputs  = Input(shape=(N_FRAMES, IMG_SIZE, IMG_SIZE, 3))
    x       = Rescaling(255.0)(inputs)
    x       = TimeDistributed(net)(x)
    x       = TimeDistributed(Flatten())(x)
    x       = Bidirectional(LSTM(64, return_sequences=True))(x)
    x       = TransformerLayer(num_heads=4, feed_forward_dim=128)(x)
    x       = Flatten()(x)
    x       = Dense(256, activation='relu')(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1. Load file paths
    file_paths, labels = get_file_paths_and_labels(DATA_DIR, CLASSES)

    # 2. Train/val split
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        file_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")

    # 3. Create generators
    train_gen = VideoGenerator(train_paths, train_labels, batch_size=BATCH_SIZE, shuffle=True)
    val_gen   = VideoGenerator(val_paths,   val_labels,   batch_size=BATCH_SIZE, shuffle=False)

    # 4. Build model
    model = build_model(num_classes=len(CLASSES))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 5. Train
    os.makedirs("model", exist_ok=True)
    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss', restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                MODEL_PATH,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                save_weights_only=False  # ← saves full model (architecture + weights)
            ),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)
        ]
    )

    print(f"\n✅ Training complete! Model saved to → {MODEL_PATH}")
    gc.collect()