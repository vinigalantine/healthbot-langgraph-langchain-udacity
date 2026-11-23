import logging
from datetime import date
from pathlib import Path
from contextvars import ContextVar

APP_LOGGER_NAME = "healthbot"
LOG_DIR = Path("logs")
LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - THREAD_ID %(thread_id)s - %(message)s"
)
DATE_FORMAT = "%d/%m/%Y - %H:%M:%S"
NO_THREAD = "-"

current_thread_id: ContextVar[str] = ContextVar("current_thread_id", default="-")


class ThreadIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.thread_id = current_thread_id.get()
        return True


class PerThreadFileHandler(logging.Handler):
    def __init__(self, log_dir: Path) -> None:
        super().__init__()
        self.log_dir = log_dir
        self._files: dict[str, logging.FileHandler] = {}

    def _file_for(self, thread_id: str) -> logging.FileHandler:
        handler = self._files.get(thread_id)

        if handler is None:
            name = "startup" if thread_id == NO_THREAD else thread_id
            path = self.log_dir / f"run_{name}_{date.today():%d-%m-%Y}.log"
            handler = logging.FileHandler(path, encoding="utf-8")
            handler.setFormatter(self.formatter)
            self._files[thread_id] = handler

        return handler

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._file_for(record.thread_id).emit(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        for handler in self._files.values():
            handler.close()

        self._files.clear()
        super().close()


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    handler = PerThreadFileHandler(LOG_DIR)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    handler.addFilter(ThreadIdFilter())

    logging.basicConfig(level=logging.WARNING, handlers=[handler])
    logging.getLogger(APP_LOGGER_NAME).setLevel(logging.INFO)


def get_logger(file_name: str) -> logging.Logger:
    return logging.getLogger(f"{APP_LOGGER_NAME}.{file_name}")


def set_thread_id(thread_id: str) -> None:
    current_thread_id.set(thread_id)
