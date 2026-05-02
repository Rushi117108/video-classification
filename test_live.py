import requests

# ── Replace with your actual HuggingFace username ────────────────────────────
USERNAME   = "rushi-117"
BASE_URL   = f"https://{USERNAME}-video-action-classifier.hf.space"

# ── Replace with any test video path on your machine ─────────────────────────
VIDEO_PATH = r"D:\Video Classification\UCF-101\UCF-101\HighJump\v_HighJump_g22_c03.avi"

# ── 1. Health Check ───────────────────────────────────────────────────────────
print("--- Health Check ---")
r = requests.get(f"{BASE_URL}/health")
print(r.json())

# ── 2. Prediction ─────────────────────────────────────────────────────────────
print("\n--- Prediction ---")
with open(VIDEO_PATH, "rb") as f:
    response = requests.post(
        f"{BASE_URL}/predict",
        files={"file": ("test.avi", f, "video/x-msvideo")}
    )

if response.status_code == 200:
    result = response.json()
    print(f"✅ Predicted Class : {result['predicted_class']}")
    print(f"   Confidence      : {result['confidence'] * 100:.1f}%")
    print(f"\n   Top 3:")
    for item in result['top_3']:
        print(f"     - {item['label']:<20} {item['confidence']*100:.1f}%")
else:
    print(f"❌ Error {response.status_code}: {response.text}")