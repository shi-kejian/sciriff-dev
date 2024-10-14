"""
Store paths to shared directories.
"""


from pathlib import Path


project_root = Path(__file__).parent.parent


TASK_DIR = project_root / "tasks"
TASK_SCHEMA_FILE = TASK_DIR / "task_schema.yaml"
TEMPLATE_DIR = project_root / "tasks/templates"
DATA_DIR = project_root / "tasks/data"
INSTANCE_DIR = project_root / "data/instances"
EVAL_DIR = project_root / "eval"
PREDICTION_DIR = project_root / "eval/results/prediction"
METRICS_DIR = project_root / "eval/results/metrics"
