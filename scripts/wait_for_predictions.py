"""Wait until the Delta predictions table has enough rows, or as many as the topic has messages.

Used by the pipeline check. Fails early if the scoring job has died.
"""

import argparse
import os
import subprocess
import sys
import time

from deltalake import DeltaTable


def rows() -> int:
    try:
        return len(
            DeltaTable("data/predictions").to_pyarrow_dataset().to_table(columns=["kafka_offset"])
        )
    except Exception:
        return 0  # the table does not exist yet


def topic_messages() -> int:
    from confluent_kafka import Consumer, TopicPartition

    c = Consumer({"bootstrap.servers": "localhost:19092", "group.id": "wait"})
    parts = c.list_topics("transactions", timeout=10).topics["transactions"].partitions
    total = sum(c.get_watermark_offsets(TopicPartition("transactions", n))[1] for n in parts)
    c.close()
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-rows", type=int, default=0)
    parser.add_argument("--match-topic", action="store_true")
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    start = time.time()
    while time.time() - start < args.timeout:
        target = topic_messages() if args.match_topic else args.min_rows
        n = rows()
        if n >= target:
            print(f"{n} predictions (target {target}) after {int(time.time() - start)}s")
            return 0
        if subprocess.run(["pgrep", "-f", "score_stream.py"], capture_output=True).returncode:
            print(f"::error::the scoring job is not running ({n} predictions so far)")
            return 1
        time.sleep(5)
    print(f"::error::{rows()} predictions after {args.timeout}s (target {target})")
    return 1


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)  # deltalake's runtime threads can abort a normal interpreter exit
