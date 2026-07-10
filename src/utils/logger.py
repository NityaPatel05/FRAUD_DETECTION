import sys
from pathlib import Path
from loguru import logger


def get_logger(name: str, log_dir: str = "logs") -> "logger":
    """
    Returns a configured loguru logger that writes to both stdout
    and a rotating file at logs/<name>.log.
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"{name}.log"

    # Remove default handler
    logger.remove()

    # Stdout — INFO and above
    logger.add(
        sys.stdout,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level:<8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> — <level>{message}</level>",
        colorize=True,
    )

    # File — DEBUG and above, rotating 50 MB, keep 5
    logger.add(
        str(log_path),
        level="DEBUG",
        rotation="50 MB",
        retention=5,
        compression="zip",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{line} — {message}",
    )

    return logger.bind(name=name)
