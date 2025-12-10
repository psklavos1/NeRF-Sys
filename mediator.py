import argparse
import os
import subprocess
import tempfile
import uuid
import json
import traceback
import threading
import logging
from pathlib import Path

from common.kafka import KafkaConsumer
from common.kafka.topic_manager import KafkaTopicManager

NERF_PROC = "nerf_runner.py"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger("mediator")


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_logging():
    # Use the same logs/ directory convention as your internal Logger
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "mediator.log"

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # file handler: logs/mediator.log
    file_handler = logging.FileHandler(str(log_path), mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # optional console output
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    logger.info(f"Mediator logging to {log_path}")


# ---------------------------------------------------------------------
# Kafka listening
# ---------------------------------------------------------------------
def listen_for_config(broker: str, topic: str, group_id: str):
    consumer_config = {
        "bootstrap.servers": broker,
        "group.id": group_id,
        "auto.offset.reset": "latest",
        "enable.auto.commit": True,
    }
    kafka_consumer = KafkaConsumer(topic, consumer_config)
    logger.info(f"Listening for NeRF configs on topic '{topic}'...")

    try:
        while True:
            message = kafka_consumer.receive()
            if message is None:
                continue
            logger.info("Received new NeRF configuration")
            yield message
    finally:
        kafka_consumer.close()


# ---------------------------------------------------------------------
# Process launching
# ---------------------------------------------------------------------
def launch_process(script, config_path=None, cwd=None):
    """
    Launch nerf_runner.py with --configPath <config_path>.

    stdout/stderr are sent to DEVNULL so the mediator terminal stays clean.
    The NeRF code logs to its own files via your internal Logger.
    """
    if isinstance(script, list):
        cmd = script
    elif isinstance(script, str):
        cmd = ["python", script]
        if config_path:
            cmd.extend(["--configPath", config_path])
    else:
        raise TypeError("script must be a str or list")

    env = None
    if cwd is not None:
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.abspath(cwd)

    return subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        cwd=cwd,
        env=env,
    )


def write_temp_config(cfg: dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp, indent=4)
        return tmp.name


# ---------------------------------------------------------------------
# Job handling
# ---------------------------------------------------------------------
def handle_config(config: dict):
    """
    Expected config shape (example):
      {
        "job_id": "nerf-001",   # optional, otherwise auto-assigned
        "op": "train" | "eval" | "view" | "video",
        ... other nerf_runner args ...
      }
    """
    job_id = config.get("job_id", uuid.uuid4().hex[:12])

    thread = threading.Thread(
        target=run_nerf_thread,
        args=(config, job_id),
        daemon=True,
    )
    thread.start()


def run_nerf_thread(cfg: dict, job_id: str) -> int:
    """
    Launch a NeRF runner process and wait for it in a background thread.

    Mediator only logs start + exit code. All detailed logs are handled
    inside the NeRF pipeline (your internal Logger).
    """
    tmp_path = None
    op = cfg.get("op", "train")
    try:
        cfg = dict(cfg)
        cfg["job_id"] = job_id

        tmp_path = write_temp_config(cfg)
        logger.info(f"[job_id={job_id}] Launching NeRF job with op='{op}'")

        process = launch_process(
            script=NERF_PROC,
            config_path=tmp_path,
            cwd=ROOT_DIR,
        )
        process.wait()

        if process.returncode != 0:
            logger.error(
                f"[job_id={job_id}] NeRF job FAILED (op='{op}', code={process.returncode})"
            )
        else:
            logger.info(
                f"[job_id={job_id}] NeRF job COMPLETED (op='{op}', code=0)"
            )

        return process.returncode

    except Exception as e:
        logger.error(f"[job_id={job_id}] NeRF job ERROR (op='{op}'): {e}")
        logger.error(traceback.format_exc())
        return -1

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                logger.warning(
                    f"[job_id={job_id}] Could not remove temp config: {tmp_path}"
                )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------
def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="NeRF Job Mediator")
    parser.add_argument("--broker", type=str, default="localhost:9092")
    parser.add_argument("--topic", type=str, default="nerf_jobs")
    parser.add_argument("--group_id", type=str, default="nerf-mediator")
    args = parser.parse_args()

    logger.info("NeRF Mediator starting...")
    logger.info(
        f"Kafka Broker: {args.broker} | Topic: {args.topic} | Group: {args.group_id}"
    )

    topic_mgr = KafkaTopicManager(args.broker, logger=logger)
    topic_mgr.create_topic(args.topic)

    for config in listen_for_config(args.broker, args.topic, args.group_id):
        if not isinstance(config, dict):
            logger.error(f"Ignoring non-dict message: {config}")
            continue
        handle_config(config)


if __name__ == "__main__":
    main()
