from pathlib import Path
from typing import Any, Dict

TESTS_ROOT = Path(__file__).parent
FIXTURES_ROOT = TESTS_ROOT / "fixtures"


def _config_to_str(cfg: Dict[str, Any]) -> str:
    result = "-".join([f"{k}={cfg[k]}" for k in sorted(cfg)])
    return result
