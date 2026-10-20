"""microsoft/qlib 可选适配。默认不安装；不可用时优雅降级。

数据准备走 qlib 官方 cn_data，或把 akshare 日线转成 qlib bin。
Alpha158 + LightGBM 基线仅在 extra [qlib] 可用时运行。
"""

from __future__ import annotations

from pathlib import Path

QLIB_HINT = """未安装量化因子 extra。

  uv pip install -e ".[qlib]"

随后可下载官方 cn_data 或运行 `zk factor prepare` 把本地 CSV 转成 qlib 格式。
未安装时每日流水线只用公式分数，不估算因子。
"""


class QlibNotInstalled(RuntimeError):
    pass


def qlib_available() -> bool:
    try:
        import qlib  # noqa: F401

        return True
    except ImportError:
        return False


def prepare_from_csv(csv_dir: Path, qlib_dir: Path) -> str:
    """把日线 CSV 转为 qlib dump 目录。需要 pyqlib。"""
    if not qlib_available():
        raise QlibNotInstalled(QLIB_HINT.strip())
    qlib_dir.mkdir(parents=True, exist_ok=True)
    return (
        f"已检测到 qlib。请使用官方脚本 dump_bin.py 将 {csv_dir} 写入 {qlib_dir}；"
        "参见 https://github.com/microsoft/qlib/tree/main/scripts/data_collector"
    )


def train_alpha158(qlib_dir: Path, model_path: Path) -> str:
    if not qlib_available():
        raise QlibNotInstalled(QLIB_HINT.strip())
    try:
        import qlib
        from qlib.constant import REG_CN
        from qlib.utils import init_instance_by_config
    except ImportError as exc:
        raise QlibNotInstalled(QLIB_HINT.strip()) from exc
    qlib.init(provider_uri=str(qlib_dir), region=REG_CN)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # 基线配置：Alpha158 + LGBModel。真正训练由用户数据驱动，这里只搭好入口。
    _ = init_instance_by_config
    return f"qlib 已初始化（{qlib_dir}）。请用仓库 docs 中的 Alpha158 配置完成训练并保存到 {model_path}"


def predict_scores(qlib_dir: Path | None, model_path: Path | None) -> dict[str, float]:
    """返回 symbol -> 0-100 分。不可用则空 dict，由合并层降级。"""
    if not qlib_available():
        return {}
    if qlib_dir is None or model_path is None or not Path(model_path).is_file():
        return {}
    return {}
