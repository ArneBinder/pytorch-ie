import logging
from typing import Any, Dict, Optional

# kept for backward compatibility
from pie_core import AutoTaskModule

from pytorch_ie.pipeline import Pipeline

# kept for backward compatibility
from pytorch_ie.pytorch_model import AutoModel

logger = logging.getLogger(__name__)


# kept for backward compatibility
class AutoPipeline:
    @staticmethod
    def from_pretrained(
        pretrained_model_name_or_path: str,
        force_download: bool = False,
        resume_download: bool = False,
        proxies: Optional[Dict] = None,
        use_auth_token: Optional[str] = None,
        cache_dir: Optional[str] = None,
        local_files_only: bool = False,
        taskmodule_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: int = -1,
        binary_output: bool = False,
        **kwargs,
    ) -> Pipeline:
        logger.warning(
            "pytorch_ie.AutoPipeline is deprecated. Use pytorch_ie.Pipeline.from_pretrained instead."
        )

        return Pipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            use_auth_token=use_auth_token,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            taskmodule_kwargs=taskmodule_kwargs or {},
            model_kwargs=model_kwargs or {},
            device=device,
            binary_output=binary_output,
            **kwargs,
        )
