# autolabel_predict.py (1080p-friendly)
from glob import glob
from ultralytics import YOLO

# --- Choose a preset for 1080p (1920x1080) ---
# Balanced (good accuracy, much faster than 1536):
IMGSZ = 1080          # square letterbox, multiple of 32
CONF  = 0.70         # low-ish for recall during autolabel
IOU   = 0.6
AUG   = False        # set True for test-time aug (slower, small recall bump)
MAX_DET = 800        # 1080p usually doesn't need 1500

# High-recall (busier scenes, tiny objects; slower):
# IMGSZ, CONF, IOU, AUG, MAX_DET = 1280, 0.15, 0.60, True, 1200

# If you only want certain classes prefilled (ids per your data.yaml/class order), uncomment:
# CLASSES = [0, 1, 2, 3]  # e.g., Pedestrian, Bicycle, Motorcycle, Passenger Car
CLASSES = None

# --- Pick latest trained model ---
candidates = sorted(glob(r"runs/detect/traffic_v8s*/weights/best.pt"), key=str.lower)
model_path = candidates[-1] if candidates else r"runs/detect/traffic_v8s6/weights/best.pt"
model = YOLO(model_path)

# Use your ROI-masked images if you created them; otherwise original images
SOURCE = "data/miovision/images"       # or "data/side-angle/images_masked"
RUN_NAME = f"miovision_{IMGSZ}{'_aug' if AUG else ''}"

model.predict(
    source=SOURCE,
    imgsz=IMGSZ,
    conf=CONF,
    iou=IOU,
    half=True,          # Ampere/Lovelace GPUs: good speedup
    device=0,
    max_det=MAX_DET,
    augment=AUG,        # TTA via Python API
    classes=CLASSES,
    save_txt=True,
    save_conf=False,
    project="runs/autolabel",
    name=RUN_NAME,
)
