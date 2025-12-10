import argparse
from datetime import datetime
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

logger = logging.getLogger("Mediator")


# ---------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------
def setup_logging():
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / "mediator.log"

    logger.setLevel(logging.INFO)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(str(log_path), mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

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
    kafka_consumer = KafkaConsumer(topic, consumer_config, logger=logger)
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
def launch_process(script, config_path=None, cwd=None, devices: str | None = None):
    """
    Launch nerf_runner.py with --configPath <config_path>.

    If `devices` is not None and not "all", it is passed via CUDA_VISIBLE_DEVICES.
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

    env = os.environ.copy()
    if cwd is not None:
        env["PYTHONPATH"] = os.path.abspath(cwd)

    # Device selection:
    # - devices is None or "all" -> do not touch CUDA_VISIBLE_DEVICES (all GPUs visible)
    # - otherwise set CUDA_VISIBLE_DEVICES to the provided string, e.g. "0" or "0,1"
    if devices is not None and devices.lower() != "all":
        env["CUDA_VISIBLE_DEVICES"] = devices

    return subprocess.Popen(
        cmd,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
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
def handle_config(config: dict, devices: str | None):
    """
    Expected config shape (example):
      {
        "job_id": "nerf-001",   # optional, otherwise auto-assigned
        "op": "train" | "eval" | "view" | "video",
        ... other NeRF config parameters ...
      }

    `devices` comes from the CLI and is shared across all jobs started by this mediator.
    """
    job_id = config.get("job_id", uuid.uuid4().hex[:12])
    op = config.get("op", "train")

    thread = threading.Thread(
        target=run_nerf_thread,
        args=(config, job_id, devices),
        daemon=True,
    )
    thread.start()


def run_nerf_thread(cfg: dict, job_id: str, devices: str | None) -> int:
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

        # Also inject devices into the config if you want nerf_runner to see it explicitly
        if devices is not None:
            cfg.setdefault("devices", devices)

        date_str = datetime.now().strftime("%y%m%d")
        run_tag = f"{date_str}_{op}"
        cfg["fname"] = f"{job_id}/{run_tag}"  # => logs/{job_id}/{tag}/

        tmp_path = write_temp_config(cfg)
        logger.info(
            f"[job_id={job_id}] Launching NeRF job with op='{op}' "
        )

        process = launch_process(
            script=NERF_PROC,
            config_path=tmp_path,
            cwd=ROOT_DIR,
            devices=devices,
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
    parser.add_argument("--topic", type=str, default="nerfConfigs")
    parser.add_argument("--group_id", type=str, default="nerf-mediator")
    parser.add_argument(
        "--devices",
        type=str,
        default=None,
        help=(
            "CUDA_VISIBLE_DEVICES setting for all launched NeRF jobs. "
            'Examples: "0", "0,1,2". '
            'If omitted or set to "all", all GPUs remain visible.'
        ),
    )
    args = parser.parse_args()

    logger.info("NeRF Mediator starting...")
    logger.info(
        f"Kafka Broker: {args.broker} | Topic: {args.topic} | "
        f"Group: {args.group_id} | Device: {'cuda:' + args.devices or 'cuda'}"
    )

    topic_mgr = KafkaTopicManager(args.broker, logger=logger)
    topic_mgr.create_topic(args.topic)

    for config in listen_for_config(args.broker, args.topic, args.group_id):
        if not isinstance(config, dict):
            logger.error(f"Ignoring non-dict message: {config}")
            continue
        handle_config(config, args.devices)


if __name__ == "__main__":
    main()
