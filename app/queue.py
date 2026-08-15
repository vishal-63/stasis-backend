import asyncio
import logging
from dataclasses import dataclass

from app import db
from app.config import get_settings
from app.pipeline.runner import run_pipeline

logger = logging.getLogger(__name__)

MAX_CONCURRENT_JOBS = 2


@dataclass
class Job:
    note_id:     str
    job_id:      str
    user_id:     str
    source_url:  str
    resume_from: str | None


class JobQueue:
    def __init__(self):
        self._queue: asyncio.Queue[Job] = asyncio.Queue(maxsize=100)
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
        self._running = False

    async def enqueue(self, job: Job) -> None:
        logger.info(f"Enqueueing job {job.job_id}")
        """Add a job to the queue. Raises QueueFull if at capacity."""
        try:
            self._queue.put_nowait(job)
            logger.info(f"Job {job.job_id} enqueued successfully — queue size: {self._queue.qsize()}")
        except asyncio.QueueFull:
            logger.warning(f"Queue full — job {job.job_id} rejected")
            raise RuntimeError("Job queue is full")

    async def start_worker(self) -> None:
        """Background worker that processes jobs from the queue."""
        self._running = True
        logger.info("Job queue worker started")

        while self._running:
            try:
                # Block until a job is available
                job = await self._queue.get()

                # Run within semaphore to cap concurrency
                asyncio.create_task(self._process(job))

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception(f"Queue worker error: {e}")

    async def _process(self, job: Job) -> None:
        async with self._semaphore:
            try:
                await run_pipeline(
                    note_id=job.note_id,
                    job_id=job.job_id,
                    user_id=job.user_id,
                    source_url=job.source_url,
                    resume_from=job.resume_from,  # ← add
                )
            except Exception as e:
                logger.exception(f"Unhandled error in job {job.job_id}: {e}")
            finally:
                self._queue.task_done()

    async def stop(self) -> None:
        self._running = False
        logger.info("Job queue worker stopped")

    @property
    def size(self) -> int:
        return self._queue.qsize()


job_queue = JobQueue()