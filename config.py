from pathlib import Path

ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = ROOT / "weights"
ARTIFACTS_DIR = ROOT / "artifacts"   # small JSON/CSV outputs the app reads (committed)
DATASETS_DIR = ROOT / "datasets"     # downloaded data (git-ignored)
BEST_WEIGHTS = WEIGHTS_DIR / "best.pt"

RF_WORKSPACE = "joseph-nelson"
RF_PROJECT = "hard-hat-workers"
RF_VERSION = 5
RF_FORMAT = "yolov11"
POSITIVE_CLASS = "head"

# --- Training -----------------------------------------------------------------
BASE_MODEL = "yolo11n.pt"
EPOCHS = 40
IMGSZ = 640
BATCH = 16
PATIENCE = 10               
SEED = 0

# --- Business assumptions (euros, per camera snapshot) ------------------------
COST_FN = 60.0
COST_FP = 8.0       
COST_TP = 8.0      
COST_TN = 0.0       
COST_REVIEW = 0.50  

BASE_RATE = 0.02    
# --- Volume and staffing ------------------------------------------------------
CAMERAS = 20
SNAPSHOTS_PER_CAMERA_PER_HOUR = 12  
SHIFT_HOURS = 10
SECONDS_PER_REVIEW = 30
REVIEWER_PRODUCTIVE_HOURS = 6        
REVIEWERS_AVAILABLE = 1           
CASES_PER_DAY = CAMERAS * SNAPSHOTS_PER_CAMERA_PER_HOUR * SHIFT_HOURS
REVIEWS_PER_REVIEWER_PER_DAY = REVIEWER_PRODUCTIVE_HOURS * 3600 // SECONDS_PER_REVIEW
REVIEW_CAPACITY_PER_DAY = REVIEWERS_AVAILABLE * REVIEWS_PER_REVIEWER_PER_DAY


def cost_params() -> dict:
    return dict(
        c_fn=COST_FN, c_fp=COST_FP, c_tp=COST_TP, c_tn=COST_TN,
        c_review=COST_REVIEW, base_rate=BASE_RATE,
    )
