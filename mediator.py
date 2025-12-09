import argparse
import os
import subprocess
import tempfile
import uuid
import json
import traceback
import threading

from server.kafka import KafkaConsumer
from server.kafka.topic_manager import KafkaTopicManager
from server.log_manager import LogManager

# ---------------------------------------------------------------------
# Constants / Globals
# ---------------------------------------------------------------------

NERF_PROC = "nerf_runner.py"
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

log_manager: LogManager | None = None
mediator_logger = None


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
    mediator_logger.info(f"Listening for NeRF configs on topic '{topic}'...")

    try:
        while True:
            message = kafka_consumer.receive()
            if message is None:
                continue

            mediator_logger.info("Received new NeRF configuration")
            log_manager.enforce_capacity()
            yield message
            mediator_logger.info("Waiting for new configurations...")
    finally:
        kafka_consumer.close()


# ---------------------------------------------------------------------
# Job handling
# ---------------------------------------------------------------------
def launch_process(script, config_path=None, cwd=None, text_logs=False):
    try:
        if isinstance(script, list):
            cmd = script  # custom command like ./waf ...
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
            text=text_logs,
            bufsize=1 if text_logs else -1,
            stdout=subprocess.PIPE if text_logs else None,
            stderr=subprocess.STDOUT if text_logs else None,
            cwd=cwd,
            env=env,
        )
    except Exception as e:
        print(f"Exception Launching process: {e}")


def write_temp_config(cfg: dict) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp, indent=4)
        return tmp.name


def handle_config(config: dict):
    """
    Handle a single NeRF job configuration.

    Expected shape (example):
      {
        "job_id": "nerf-001",   # optional, otherwise auto-assigned
        "op": "train" | "eval" | "view" | "video",
        ... other nerf_runner args ...
      }
    """
    job_id = config.get("job_id", uuid.uuid4().hex[:12])
    op = config.get("op", "train")

    mediator_logger.info(f"[job_id={job_id}] Launching NeRF job with op='{op}'")

    thread = threading.Thread(
        target=run_nerf_thread,
        args=(config, job_id),
        daemon=True,
    )
    thread.start()

    mediator_logger.info(f"[job_id={job_id}] NeRF job thread started.")


def run_nerf_thread(cfg: dict, job_id: str) -> int:
    """
    Launch a NeRF runner process (nerf_runner.py --config <tmp>) and wait for it.
    """
    tmp_path = None
    job_logger = log_manager.get_job_logger(job_id=job_id, name="nerf")

    try:
        cfg = dict(cfg)  # shallow copy so we don't mutate Kafka message
        cfg["job_id"] = job_id

        # write temp config JSON for nerf_runner.py
        tmp_path = write_temp_config(cfg)

        job_logger.info(f"[job_id={job_id}] Starting nerf_runner.py with config:\n{cfg}")
        
        # input()
        process = launch_process(
            script=NERF_PROC,
            config_path=tmp_path,
            cwd=ROOT_DIR,
            text_logs=False,  # adjust if your launch_process supports this
        )

        process.wait()

        if process.returncode != 0:
            job_logger.error(
                f"[job_id={job_id}] ❌ nerf_runner exited with code: {process.returncode}"
            )
        else:
            job_logger.info(f"[job_id={job_id}] ✅ NeRF job completed successfully.")

        return process.returncode

    except Exception as e:
        job_logger.error(f"[job_id={job_id}] ❌ Exception in NeRF job: {e}")
        job_logger.error(traceback.format_exc())
        return -1

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                job_logger.warning(
                    f"[job_id={job_id}] Could not remove temp config: {tmp_path}"
                )


# ---------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------


def main():
    global log_manager, mediator_logger

    parser = argparse.ArgumentParser(description="NeRF Job Mediator")
    parser.add_argument("--broker", type=str, default="localhost:9092")
    parser.add_argument("--topic", type=str, default="nerf_jobs")
    parser.add_argument("--group_id", type=str, default="nerf-mediator")
    parser.add_argument("--log_capacity", type=int, default=10)

    args = parser.parse_args()

    log_manager = LogManager(capacity=args.log_capacity)
    mediator_logger = log_manager.get_mediator_logger()

    mediator_logger.info("NeRF Mediator starting...")
    mediator_logger.info(
        f"Kafka Broker: {args.broker} | Topic: {args.topic} | "
        f"Group: {args.group_id} | Log capacity: {args.log_capacity}"
    )
    
     # 🔹 ensure topic exists
    topic_mgr = KafkaTopicManager(args.broker, logger=mediator_logger)
    topic_mgr.create_topic(args.topic)
    
    for config in listen_for_config(args.broker, args.topic, args.group_id):
        # 'config' is the JSON payload consumed from Kafka
        if not isinstance(config, dict):
            mediator_logger.error(f"Ignoring non-dict message: {config}")
            continue
        handle_config(config)


if __name__ == "__main__":
    main()
