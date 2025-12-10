import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, Optional


class LogManager:
    """
    Logging helper for:
      - A long-running mediator process (global log)
      - Per-job logs (server/clients/etc) under logs/{timestamp}_{job_id}/

    Usage examples
    --------------
    lm = LogManager(capacity=20)

    mediator_logger = lm.get_mediator_logger()
    mediator_logger.info("Mediator started")

    # For a job with id/pid "12345":
    server_logger = lm.get_job_logger(job_id="12345", name="server")
    client0_logger = lm.get_job_logger(job_id="12345", name="client_0")

    server_logger.info("Server job booted")
    client0_logger.info("Client 0 connected")
    """

    def __init__(self, root_dir: Optional[str] = None, capacity: int = 20):
        """
        :param root_dir: root logs directory (default: <repo_root>/logs)
        :param capacity: max number of job dirs (logs/{timestamp}_{job_id}) kept
        """
        self.root_dir = (
            Path(__file__).resolve().parent.parent / "server_logs"  
            if root_dir is None
            else Path(root_dir).resolve()
        )
        self.capacity = max(int(capacity), 1)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        # Cache loggers so we don't re-add handlers
        self._mediator_logger: Optional[logging.Logger] = None
        self._job_loggers: Dict[Tuple[str, str], logging.Logger] = {}  # (job_dir_name, name) -> logger

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _create_job_dir(self, job_id: str) -> Path:
        """
        Create logs/{timestamp}_{job_id} and return its Path.
        Also enforces capacity on number of job dirs.
        """
        self.enforce_capacity()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # we use job_id (or pid) in the folder name so we can recover it
        safe_job_id = str(job_id)
        job_dir = self.root_dir / f"{timestamp}_{safe_job_id}"
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def _iter_job_dirs(self):
        """Return all per-job directories under root_dir, sorted by mtime (oldest first)."""
        dirs = [
            d
            for d in self.root_dir.iterdir()
            if d.is_dir() and d.name != "global"
        ]
        return sorted(dirs, key=lambda p: p.stat().st_mtime)

    # -------------------------------------------------------------------------
    # Capacity management
    # -------------------------------------------------------------------------

    def enforce_capacity(self) -> None:
        """
        Keep at most `capacity - 1` existing job directories, since we are about
        to create a new one.

        This does NOT touch the global mediator.log.
        """
        dirs = self._iter_job_dirs()
        if len(dirs) >= self.capacity:
            num_to_delete = len(dirs) - (self.capacity - 1)
            for d in dirs[:num_to_delete]:
                try:
                    shutil.rmtree(d)
                except Exception as e:
                    print(f"[LogManager] Could not delete {d}: {e}")

    # -------------------------------------------------------------------------
    # Global mediator logger
    # -------------------------------------------------------------------------

    def get_mediator_logger(
        self,
        name: str = "mediator",
        log_level: int = logging.INFO,
    ) -> logging.Logger:
        """
        Global logger for the long-running mediator process.

        - File: logs/mediator.log
        - Also logs to console (stdout)
        """
        if self._mediator_logger is not None:
            return self._mediator_logger

        log_path = self.root_dir / "mediator.log"

        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        logger.propagate = False  # don't bubble up to root

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # File handler (overwrite on each run)
        file_handler = logging.FileHandler(str(log_path), mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        self._mediator_logger = logger
        return logger

    # -------------------------------------------------------------------------
    # Per-job loggers (server/client/ns3/fl/nerf/etc)
    # -------------------------------------------------------------------------

    def get_job_logger(
        self,
        job_id: str,
        name: str,
        log_level: int = logging.INFO,
    ) -> logging.Logger:
        """
        Logger for an individual job component, e.g.:
          - job_id="12345", name="nerf"

        Log file path:
          logs/{timestamp}_{job_id}/{name}.log

        The {timestamp}_{job_id} directory is created on first logger
        request for this job_id.
        """
        # Find or create job directory for this job_id
        job_dir = self.get_job_log_dir(job_id)
        if job_dir is None:
            job_dir = self._create_job_dir(job_id)

        key = (job_dir.name, name)
        if key in self._job_loggers:
            return self._job_loggers[key]

        logger = logging.getLogger(f"{name}_{job_dir.name}")
        logger.setLevel(log_level)
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        log_path = job_dir / f"{name}.log"
        file_handler = logging.FileHandler(str(log_path), mode="w")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        self._job_loggers[key] = logger
        return logger

    # -------------------------------------------------------------------------
    # Utility: resolve job log dir
    # -------------------------------------------------------------------------

    def get_job_log_dir(self, job_id: str) -> Optional[Path]:
        """
        Return the latest matching job directory for a given job_id
        """
        suffix = f"_{job_id}"
        candidates = [
            d for d in self._iter_job_dirs() if d.name.endswith(suffix)
        ]
        if not candidates:
            return None
        # newest first
        return candidates[-1]
