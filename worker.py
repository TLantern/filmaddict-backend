#!/usr/bin/env python3
"""
RQ Worker script that uses the configured Redis connection with SSL support.
"""
import os
import sys
import multiprocessing

# Disable system proxy detection to avoid fork crash on macOS
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'

# Fix for Python 3.14 + macOS: use 'spawn' instead of 'fork' to avoid asyncio issues
if sys.platform == "darwin":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        # Already set, ignore
        pass

# Add the backend directory to the path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rq import Worker
from utils.jobs import redis_conn, video_processing_queue

if __name__ == "__main__":
    # Create a worker that uses our configured Redis connection
    # Use job_class to avoid multiprocessing issues
    worker = Worker(
        [video_processing_queue], 
        connection=redis_conn,
        job_class=None  # Use default job class
    )
    worker.work()

