"""统一异常与退出码。"""

from __future__ import annotations


class PlannerError(Exception):
    """所有可预期错误的基类。"""

    exit_code = 3

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ConfigError(PlannerError):
    """输入配置不合法。"""

    exit_code = 2


class LibraryError(PlannerError):
    """数据集库读写失败。"""

    exit_code = 3


class InsufficientDataError(PlannerError):
    """缺少必要的历史数据 / 估算参数。"""

    exit_code = 4


class ProbeError(PlannerError):
    """吞吐探测失败。"""

    exit_code = 5
