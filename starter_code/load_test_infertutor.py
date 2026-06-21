```python
"""
Fixed workload load tester for InferTutor Arena.

This script simulates concurrent users sending streaming chat-completion
requests to an OpenAI-compatible server.

It supports four workload modes:
    1. text   - short text prompts only
    2. long   - longer text prompts only
    3. image  - multimodal prompts with a small generated PNG
    4. mixed  - product-like mix of text, long, and image prompts

The script measures:
    - total requests
    - total errors
    - error rate
    - total streamed chunks
    - time to first token, also called TTFT
    - inter-token latency, also called ITL
    - total request latency
    - aggregate streaming throughput

At the end, it writes a JSON results file into:
    results_infertutor/
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import random
import statistics
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import httpx
from rich.console import Console
from rich.live import Live
from rich.table import Table


# Rich console used for nicely formatted terminal output.
console = Console()

# Directory containing this script.
# This makes file paths relative to the script location instead of the shell's
# current working directory.
ROOT = Path(__file__).parent

# Load the official prompt set once at startup.
# Expected file:
#     prompts.json
#
# Expected structure:
#     {
#         "system_prompt": "...",
#         "text": [...],
#         "long": [...],
#         "image": [...]
#     }
PROMPTS = json.loads((ROOT / "prompts.json").read_text())


def make_png_data_url(width: int = 256, height: int = 192) -> str:
    """
    Create a small deterministic PNG and return it as a data URL.

    Why this exists:
        Image workloads need an image input. Instead of depending on an
        external image file or URL, this function generates a tiny image
        programmatically.

    Why deterministic:
        The image is always the same for the same width and height. This keeps
        benchmark inputs stable across runs.

    Args:
        width:
            Width of the generated PNG in pixels.

        height:
            Height of the generated PNG in pixels.

    Returns:
        str:
            A data URL that can be passed to an OpenAI-compatible multimodal
            chat-completion API.

            Example:
                data:image/png;base64,...
    """

    # Simple RGB palette used to draw a synthetic diagram-like image.
    # Each tuple is an RGB color.
    palette = [
        (245, 247, 250),  # light background
        (38, 92, 135),    # blue bar
        (228, 111, 71),   # orange bar
        (81, 168, 129),   # green blocks
    ]

    # Raw image data for PNG scanlines.
    #
    # PNG scanlines begin with one filter byte per row.
    # We use filter type 0, which means "no filter".
    raw = bytearray()

    # Build the image row by row.
    for y in range(height):
        # PNG filter byte for this row.
        raw.append(0)

        for x in range(width):
            # Create a checkerboard-like base pattern.
            idx = ((x // 24) + (y // 24)) % len(palette)

            # Add a blue horizontal bar to make the image diagram-like.
            if 48 < x < 208 and 70 < y < 92:
                idx = 1

            # Add an orange horizontal bar to make the image diagram-like.
            if 48 < x < 208 and 132 < y < 154:
                idx = 2

            # Append the selected RGB color to the raw byte buffer.
            raw.extend(palette[idx])

    def chunk(kind: bytes, data: bytes) -> bytes:
        """
        Build one PNG chunk.

        PNG files are made of chunks. Each chunk has:
            - length
            - chunk type
            - chunk data
            - CRC checksum

        Args:
            kind:
                Four-byte PNG chunk type, such as b"IHDR", b"IDAT", or b"IEND".

            data:
                Raw chunk payload.

        Returns:
            bytes:
                Encoded PNG chunk.
        """

        import struct

        return (
            # Chunk payload length, encoded as big-endian unsigned int.
            struct.pack(">I", len(data))

            # Chunk type.
            + kind

            # Chunk payload.
            + data

            # CRC checksum over chunk type and data.
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    import struct

    # PNG IHDR header fields:
    #   width, height
    #   bit depth = 8
    #   color type = 2, meaning truecolor RGB
    #   compression method = 0
    #   filter method = 0
    #   interlace method = 0
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)

    # Assemble the full PNG:
    #   signature
    #   IHDR chunk
    #   IDAT chunk containing compressed pixel data
    #   IEND chunk
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )

    # Convert binary PNG bytes into a base64 data URL.
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


# Generate the synthetic image once and reuse it across all image requests.
# This avoids doing PNG generation inside every user request loop.
IMAGE_URL = make_png_data_url()


@dataclass
class Stats:
    """
    Thread-safe-ish async statistics accumulator for the load test.

    One Stats object is shared by all simulated users.

    Because many async tasks update this object concurrently, updates are
    protected by an asyncio.Lock.
    """

    # Number of completed requests, including both successes and errors.
    total_requests: int = 0

    # Number of failed requests.
    total_errors: int = 0

    # Total number of streamed content chunks received across successful calls.
    total_chunks: int = 0

    # Time to first token in milliseconds for each successful request.
    ttft_ms: list[float] = field(default_factory=list)

    # Inter-token latency in milliseconds for each successful request.
    itl_ms: list[float] = field(default_factory=list)

    # End-to-end request latency in milliseconds for each successful request.
    latency_ms: list[float] = field(default_factory=list)

    # Per-request throughput in chunks per second.
    per_request_tps: list[float] = field(default_factory=list)

    # Wall-clock start time for the test.
    started_at: float = 0.0

    # Number of simulated users currently started.
    active_users: int = 0

    # Async lock used to protect concurrent writes to the stats fields.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def success(self, ttft: float, itl: float, latency: float, chunks: int):
        """
        Record one successful streaming request.

        Args:
            ttft:
                Time to first token in milliseconds.

            itl:
                Average inter-token latency in milliseconds.

            latency:
                End-to-end request latency in milliseconds.

            chunks:
                Number of streamed content chunks received.
        """

        # Use the lock because multiple user_loop tasks may call this at once.
        async with self.lock:
            self.total_requests += 1
            self.total_chunks += chunks
            self.ttft_ms.append(ttft)
            self.itl_ms.append(itl)
            self.latency_ms.append(latency)

            # Convert request latency from milliseconds to seconds before
            # computing chunks per second.
            self.per_request_tps.append(
                chunks / (latency / 1000) if latency > 0 else 0
            )

    async def error(self):
        """
        Record one failed request.

        A failed request still counts toward total_requests because it consumed
        load-test capacity and should affect the error rate.
        """

        async with self.lock:
            self.total_requests += 1
            self.total_errors += 1

    @staticmethod
    def percentile(values: list[float], p: float) -> float:
        """
        Compute an approximate percentile from a list of values.

        Args:
            values:
                Numeric values.

            p:
                Percentile to compute. For example:
                    50 for p50
                    95 for p95
                    99 for p99

        Returns:
            float:
                The selected percentile value, or 0.0 if values is empty.

        Note:
            This uses a simple nearest-rank style index. It is good enough for
            report-level load-test summaries, but it is not an interpolated
            percentile calculation.
        """

        if not values:
            return 0.0

        ordered = sorted(values)

        # Convert percentile into an index.
        # min(...) protects against indexing past the end of the list.
        index = min(int(len(ordered) * p / 100), len(ordered) - 1)

        return ordered[index]

    def elapsed(self) -> float:
        """
        Return elapsed wall-clock time in seconds.

        Returns 0.0 if the test has not started yet.
        """

        return time.time() - self.started_at if self.started_at else 0.0

    def results(self) -> dict:
        """
        Return the current statistics as a JSON-serializable dictionary.

        This is used both for:
            - live terminal display
            - final JSON result file
        """

        elapsed = self.elapsed()

        return {
            # Raw counts.
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,

            # Error rate is protected from division by zero.
            "error_rate": self.total_errors / max(self.total_requests, 1),

            # Total streamed output volume.
            "total_stream_chunks": self.total_chunks,

            # Time-to-first-token percentiles.
            "ttft_p50_ms": self.percentile(self.ttft_ms, 50),
            "ttft_p95_ms": self.percentile(self.ttft_ms, 95),
            "ttft_p99_ms": self.percentile(self.ttft_ms, 99),

            # Inter-token latency percentiles.
            "itl_p50_ms": self.percentile(self.itl_ms, 50),
            "itl_p95_ms": self.percentile(self.itl_ms, 95),

            # End-to-end latency percentiles.
            "latency_p50_ms": self.percentile(self.latency_ms, 50),
            "latency_p95_ms": self.percentile(self.latency_ms, 95),

            # Average per-request streamed chunks per second.
            "per_request_tps_mean": (
                statistics.mean(self.per_request_tps)
                if self.per_request_tps
                else 0
            ),

            # Global throughput across the whole test.
            "aggregate_stream_chunks_per_s": (
                self.total_chunks / elapsed if elapsed else 0
            ),

            # Completed requests per second, including errors.
            "requests_per_s": self.total_requests / elapsed if elapsed else 0,
        }

    def table(self) -> Table:
        """
        Build a Rich table for live terminal display.

        Returns:
            rich.table.Table:
                A table containing the most important live metrics.
        """

        # Get a snapshot of the current metrics.
        r = self.results()

        # Table title includes elapsed time so the user can see progress.
        table = Table(title=f"InferTutor Load Test - {self.elapsed():.0f}s")

        # First column contains metric names.
        table.add_column("Metric", style="cyan")

        # Second column contains right-aligned metric values.
        table.add_column("Value", justify="right", style="green")

        # Add rows in rough order of operational importance.
        table.add_row("Active users", str(self.active_users))
        table.add_row("Requests", str(r["total_requests"]))
        table.add_row("Errors", str(r["total_errors"]))
        table.add_row("TTFT p95", f'{r["ttft_p95_ms"]:.1f} ms')
        table.add_row("ITL p95", f'{r["itl_p95_ms"]:.1f} ms')
        table.add_row("Latency p95", f'{r["latency_p95_ms"]:.1f} ms')
        table.add_row("Throughput", f'{r["aggregate_stream_chunks_per_s"]:.1f} chunks/s')
        table.add_row("Req/s", f'{r["requests_per_s"]:.2f}')

        return table


def choose_messages(mode: str) -> list[dict]:
    """
    Build one OpenAI-compatible chat request message list.

    Args:
        mode:
            Workload mode. Supported values:
                "text"
                "long"
                "image"
                "mixed"

    Returns:
        list[dict]:
            Messages suitable for /v1/chat/completions.
    """

    # Every request uses the same system prompt from prompts.json.
    system = {
        "role": "system",
        "content": PROMPTS["system_prompt"],
    }

    # Text-only short prompt.
    if mode == "text":
        return [
            system,
            {
                "role": "user",
                "content": random.choice(PROMPTS["text"]),
            },
        ]

    # Text-only long prompt.
    if mode == "long":
        return [
            system,
            {
                "role": "user",
                "content": random.choice(PROMPTS["long"]),
            },
        ]

    # Multimodal prompt with one image and one text instruction.
    if mode == "image":
        content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": IMAGE_URL,
                },
            },
            {
                "type": "text",
                "text": random.choice(PROMPTS["image"]),
            },
        ]

        return [
            system,
            {
                "role": "user",
                "content": content,
            },
        ]

    # Mixed mode approximates the official product workload.
    #
    # Distribution:
    #   25% image
    #   20% long text
    #   55% short text
    roll = random.random()

    if roll < 0.25:
        return choose_messages("image")

    if roll < 0.45:
        return choose_messages("long")

    return choose_messages("text")


async def user_loop(
    user_id: int,
    args,
    stats: Stats,
    stop_event: asyncio.Event,
):
    """
    Simulate one user repeatedly sending streaming requests.

    Each simulated user:
        1. Builds a request payload.
        2. Sends it to /v1/chat/completions.
        3. Reads streamed response lines.
        4. Measures TTFT, ITL, latency, and chunks.
        5. Sleeps for a random pause.
        6. Repeats until stop_event is set.

    Args:
        user_id:
            Integer identifier for this simulated user.
            Currently unused, but useful for debugging.

        args:
            Parsed command-line arguments.

        stats:
            Shared Stats object.

        stop_event:
            Async event used to stop all user loops.
    """

    # One HTTP client per simulated user.
    # Reusing the client allows connection pooling across requests from the
    # same user.
    async with httpx.AsyncClient(timeout=args.request_timeout) as client:

        # Keep sending requests until the main runner signals shutdown.
        while not stop_event.is_set():

            # OpenAI-compatible chat-completions request body.
            payload = {
                "model": args.model,
                "messages": choose_messages(args.mode),
                "max_tokens": args.max_tokens,
                "temperature": 0.2,
                "stream": True,
            }

            # High-resolution start time for latency measurement.
            request_start = time.perf_counter()

            # Time when the first content chunk arrives.
            # None means we have not received any content yet.
            first_chunk_at = None

            # Timestamp of each streamed content chunk.
            chunk_times = []

            # Number of content chunks received for this request.
            chunks = 0

            try:
                # Send streaming POST request to the OpenAI-compatible endpoint.
                async with client.stream(
                    "POST",
                    f"{args.url.rstrip('/')}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                ) as resp:

                    # Treat non-200 HTTP responses as request errors.
                    if resp.status_code != 200:
                        await stats.error()

                        # Drain response body before continuing.
                        # This helps httpx cleanly reuse or close the connection.
                        await resp.aread()

                        continue

                    # Iterate through server-sent-event style response lines.
                    async for line in resp.aiter_lines():

                        # Skip empty lines and stream terminator.
                        if not line or line == "data: [DONE]":
                            continue

                        # OpenAI streaming responses usually prefix JSON with
                        # "data: ". Strip it before JSON parsing.
                        if line.startswith("data: "):
                            line = line[6:]

                        try:
                            # Parse one streaming chunk.
                            chunk = json.loads(line)

                            # Extract token content from the delta.
                            # Some chunks may contain role metadata or empty
                            # deltas, so default to empty string.
                            content = chunk["choices"][0]["delta"].get("content", "")

                        except Exception:
                            # Ignore malformed or unexpected stream lines.
                            # This keeps the load test running even if one line
                            # cannot be parsed.
                            continue

                        # Only count chunks that contain actual generated text.
                        if content:
                            now = time.perf_counter()

                            # First content chunk determines TTFT.
                            first_chunk_at = first_chunk_at or now

                            # Store chunk timestamp for ITL calculation.
                            chunk_times.append(now)

                            # Count this streamed content chunk.
                            chunks += 1

                # Request is complete once the stream closes.
                request_end = time.perf_counter()

                # If the request returned no content, count it as an error.
                if first_chunk_at is None or chunks == 0:
                    await stats.error()
                    continue

                # Calculate gaps between consecutive chunk arrival times.
                #
                # Example:
                #   chunk_times = [t1, t2, t3]
                #   gaps = [t2 - t1, t3 - t2]
                gaps = [
                    b - a
                    for a, b in zip(chunk_times, chunk_times[1:])
                ]

                # Time to first token in milliseconds.
                ttft = (first_chunk_at - request_start) * 1000

                # Average inter-token latency in milliseconds.
                #
                # If only one content chunk was emitted, there are no gaps.
                # In that case, ITL is set to 0.0.
                itl = (
                    sum(gaps) / len(gaps) * 1000
                    if gaps
                    else 0.0
                )

                # End-to-end request latency in milliseconds.
                latency = (request_end - request_start) * 1000

                # Record successful request metrics.
                await stats.success(ttft, itl, latency, chunks)

            except Exception:
                # Count any request-level exception as an error.
                #
                # Examples:
                #   connection refused
                #   timeout
                #   server disconnect
                #   DNS failure
                await stats.error()

            # Sleep between requests to avoid perfectly synchronized traffic.
            #
            # This makes the load more realistic than every user hammering the
            # server in lockstep.
            await asyncio.sleep(
                random.uniform(args.min_pause, args.max_pause)
            )


async def run(args):
    """
    Run the full load test.

    High-level flow:
        1. Create shared Stats object.
        2. Start users gradually according to ramp-up.
        3. Display live metrics while the test runs.
        4. Stop all users.
        5. Save final JSON results.
    """

    # Initialize shared metrics and set the start time.
    stats = Stats(started_at=time.time())

    # Event used to ask user tasks to stop.
    stop_event = asyncio.Event()

    # Track all simulated user tasks so they can be cancelled at shutdown.
    tasks = []

    async def ramp_users():
        """
        Gradually start simulated users.

        If ramp_up is 15 seconds and users is 50, this starts one user every
        15 / 50 = 0.3 seconds.

        If ramp_up is 0, all users start immediately.
        """

        # Delay between starting each user.
        delay = args.ramp_up / max(args.users, 1) if args.ramp_up else 0

        for i in range(args.users):
            # Stop ramping if the test is already ending.
            if stop_event.is_set():
                return

            # Start one simulated user as an asyncio task.
            tasks.append(
                asyncio.create_task(
                    user_loop(i, args, stats, stop_event)
                )
            )

            # Update visible active-user count.
            stats.active_users = i + 1

            # Wait before starting the next user.
            if delay:
                await asyncio.sleep(delay)

    # Start the ramp-up task.
    ramp_task = asyncio.create_task(ramp_users())

    # Use Rich Live to refresh the terminal table while the test runs.
    with Live(
        stats.table(),
        refresh_per_second=0.5,
        console=console,
    ) as live:

        # Absolute end time for the test.
        end = time.time() + args.duration

        # Refresh the display every two seconds until duration expires.
        while time.time() < end:
            await asyncio.sleep(2)
            live.update(stats.table())

    # Signal all users to stop.
    stop_event.set()

    # Cancel ramp task in case it is still adding users.
    ramp_task.cancel()

    # Cancel every active user task.
    for task in tasks:
        task.cancel()

    # Wait for cancelled tasks to finish.
    # return_exceptions=True prevents cancellation exceptions from crashing
    # the shutdown path.
    await asyncio.gather(*tasks, return_exceptions=True)

    # Final result payload written to disk.
    result = {
        "config": vars(args),
        "results": stats.results(),
    }

    # Make sure result directory exists.
    out_dir = ROOT / "results_infertutor"
    out_dir.mkdir(exist_ok=True)

    # Include label, mode, users, and timestamp in the filename.
    # This avoids overwriting prior runs.
    out_file = (
        out_dir
        / f"{args.label}_{args.mode}_{args.users}u_{int(time.time())}.json"
    )

    # Write pretty JSON for easier inspection.
    out_file.write_text(json.dumps(result, indent=2))

    # Print final metrics table and saved-file path.
    console.print(stats.table())
    console.print(f"[green]Saved {out_file}[/green]")


def main():
    """
    Parse command-line arguments and start the async load test.
    """

    parser = argparse.ArgumentParser()

    # Base URL of the OpenAI-compatible server.
    #
    # Example:
    #   --url http://localhost:8000
    #
    # The script will call:
    #   {url}/v1/chat/completions
    parser.add_argument("--url", required=True)

    # Model name sent in the request payload.
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-4B-Instruct",
    )

    # Workload mode.
    parser.add_argument(
        "--mode",
        choices=["text", "long", "image", "mixed"],
        default="mixed",
    )

    # Number of concurrent simulated users.
    parser.add_argument("--users", type=int, default=50)

    # Total test duration in seconds.
    parser.add_argument("--duration", type=int, default=60)

    # Number of seconds over which users are gradually started.
    #
    # Example:
    #   users = 50, ramp_up = 15
    #   one new user starts every 0.3 seconds.
    parser.add_argument("--ramp-up", type=int, default=15)

    # Max generated tokens per request.
    parser.add_argument("--max-tokens", type=int, default=96)

    # Per-request timeout in seconds.
    parser.add_argument("--request-timeout", type=int, default=180)

    # Minimum random pause between requests for each simulated user.
    parser.add_argument("--min-pause", type=float, default=0.2)

    # Maximum random pause between requests for each simulated user.
    parser.add_argument("--max-pause", type=float, default=1.2)

    # Human-readable run label included in the output filename and config.
    parser.add_argument("--label", default="manual")

    # GPU count used later by reporting scripts when computing score.
    #
    # This load tester does not use the value directly during request sending.
    parser.add_argument("--total-gpus", type=int, default=1)

    # Parse command-line arguments.
    args = parser.parse_args()

    # Start the async runner.
    asyncio.run(run(args))


# Standard Python entry-point guard.
#
# This allows the file to be imported without immediately running a load test.
if __name__ == "__main__":
    main()
```
