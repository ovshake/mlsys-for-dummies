#!/usr/bin/env python3
"""
Minimal LLM Inference Server

This script demonstrates the core components of an inference server:
- Request management
- Simple batching
- Token generation loop

This is educational, not production-ready. Real servers like vLLM and
SGLang have much more sophisticated implementations.

Usage:
    python minimal_inference_server.py
    python minimal_inference_server.py --num-requests 10
"""

import argparse
import asyncio
import time
from dataclasses import dataclass, field
from typing import List, Optional, AsyncIterator
from collections import deque
import random


@dataclass
class GenerateRequest:
    """A request to generate text."""
    id: int
    prompt: str
    prompt_tokens: List[int]
    max_tokens: int = 50
    temperature: float = 1.0
    created_at: float = field(default_factory=time.time)

    # Tracking
    generated_tokens: List[int] = field(default_factory=list)
    is_finished: bool = False
    prefill_done: bool = False


@dataclass
class Batch:
    """A batch of requests to process together."""
    requests: List[GenerateRequest]
    is_prefill: bool  # True for prefill, False for decode


class SimpleTokenizer:
    """
    A simplified tokenizer for demonstration.

    Real tokenizers (like SentencePiece or tiktoken) are more complex.
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        # Simple word-based tokenization
        self.token_to_id = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
        self.id_to_token = {0: "<pad>", 1: "<eos>", 2: "<unk>"}

    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs."""
        # Simplified: assign random IDs to words
        words = text.lower().split()
        tokens = []
        for word in words:
            # Hash word to get consistent token ID
            token_id = hash(word) % (self.vocab_size - 3) + 3
            tokens.append(token_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Convert token IDs back to text."""
        # Simplified: just return placeholder
        return f"[Generated {len(token_ids)} tokens]"


class SimpleModelRunner:
    """
    A simplified model runner for demonstration.

    Real model runners load actual neural networks and run GPU inference.
    """

    def __init__(self, vocab_size: int = 1000, latency_ms: float = 10):
        self.vocab_size = vocab_size
        self.latency_ms = latency_ms

    async def prefill(self, request: GenerateRequest) -> int:
        """
        Process prompt and return first generated token.

        Real prefill:
        1. Run all prompt tokens through model in parallel
        2. Build KV cache for all tokens
        3. Sample first output token
        """
        # Simulate compute time (proportional to prompt length)
        prompt_len = len(request.prompt_tokens)
        await asyncio.sleep(self.latency_ms * prompt_len / 100)

        # "Generate" first token
        first_token = random.randint(3, self.vocab_size - 1)
        return first_token

    async def decode(self, batch: List[GenerateRequest]) -> List[int]:
        """
        Generate next token for each request in batch.

        Real decode:
        1. Run single token through model for each request
        2. Update KV cache with new KV pairs
        3. Sample next token for each request
        """
        # Simulate compute time (roughly constant per batch)
        await asyncio.sleep(self.latency_ms)

        # "Generate" next tokens
        next_tokens = []
        for req in batch:
            # 10% chance of generating EOS
            if random.random() < 0.1:
                next_tokens.append(1)  # EOS
            else:
                next_tokens.append(random.randint(3, self.vocab_size - 1))
        return next_tokens


class Scheduler:
    """
    Manages request queue and batching decisions.

    Key responsibilities:
    1. Accept new requests
    2. Decide which requests to process together
    3. Manage prefill vs decode scheduling
    """

    def __init__(self, max_batch_size: int = 4):
        self.max_batch_size = max_batch_size
        self.waiting_queue: deque = deque()  # Requests waiting for prefill
        self.running_batch: List[GenerateRequest] = []  # Requests in decode phase
        self.completed: List[GenerateRequest] = []

    def add_request(self, request: GenerateRequest):
        """Add a new request to the waiting queue."""
        self.waiting_queue.append(request)
        print(f"[Scheduler] Added request {request.id} to queue "
              f"(queue size: {len(self.waiting_queue)})")

    def get_next_batch(self) -> Optional[Batch]:
        """
        Decide what to process next.

        Strategy (simplified):
        1. If we have requests waiting AND room in running batch, do prefill
        2. If running batch has requests, do decode
        """
        # Check for finished requests first
        self.running_batch = [r for r in self.running_batch if not r.is_finished]

        # Prefill new requests if we have capacity
        while (self.waiting_queue and
               len(self.running_batch) < self.max_batch_size):
            request = self.waiting_queue.popleft()
            return Batch(requests=[request], is_prefill=True)

        # Decode existing requests
        if self.running_batch:
            return Batch(requests=self.running_batch, is_prefill=False)

        return None

    def process_prefill_result(self, request: GenerateRequest, token: int):
        """Handle result from prefill."""
        request.prefill_done = True
        request.generated_tokens.append(token)
        self.running_batch.append(request)
        print(f"[Scheduler] Request {request.id} finished prefill, "
              f"added to running batch (size: {len(self.running_batch)})")

    def process_decode_result(self, request: GenerateRequest, token: int):
        """Handle result from decode."""
        request.generated_tokens.append(token)

        # Check if finished
        if token == 1 or len(request.generated_tokens) >= request.max_tokens:
            request.is_finished = True
            self.completed.append(request)
            print(f"[Scheduler] Request {request.id} finished "
                  f"({len(request.generated_tokens)} tokens)")

    def has_work(self) -> bool:
        """Check if there's more work to do."""
        return bool(self.waiting_queue or self.running_batch)


class InferenceServer:
    """
    Main inference server orchestrating all components.
    """

    def __init__(self, max_batch_size: int = 4):
        self.tokenizer = SimpleTokenizer()
        self.model_runner = SimpleModelRunner()
        self.scheduler = Scheduler(max_batch_size)
        self.request_counter = 0

    async def generate(self, prompt: str, max_tokens: int = 50) -> GenerateRequest:
        """Submit a generation request."""
        # Tokenize
        tokens = self.tokenizer.encode(prompt)

        # Create request
        request = GenerateRequest(
            id=self.request_counter,
            prompt=prompt,
            prompt_tokens=tokens,
            max_tokens=max_tokens,
        )
        self.request_counter += 1

        # Submit to scheduler
        self.scheduler.add_request(request)

        return request

    async def run_step(self) -> bool:
        """Run one step of inference."""
        batch = self.scheduler.get_next_batch()
        if batch is None:
            return False

        if batch.is_prefill:
            # Prefill phase
            request = batch.requests[0]
            print(f"[Server] Prefill request {request.id} "
                  f"({len(request.prompt_tokens)} prompt tokens)")

            token = await self.model_runner.prefill(request)
            self.scheduler.process_prefill_result(request, token)

        else:
            # Decode phase
            print(f"[Server] Decode batch of {len(batch.requests)} requests")

            tokens = await self.model_runner.decode(batch.requests)
            for request, token in zip(batch.requests, tokens):
                self.scheduler.process_decode_result(request, token)

        return True

    async def run_until_complete(self):
        """Run until all requests are complete."""
        while self.scheduler.has_work():
            await self.run_step()


async def run_demo(num_requests: int, max_batch_size: int):
    """Run a demonstration of the inference server."""
    print("=" * 60)
    print(" MINIMAL INFERENCE SERVER DEMO")
    print("=" * 60)

    server = InferenceServer(max_batch_size=max_batch_size)

    # Sample prompts
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a haiku about programming.",
        "What is machine learning?",
        "Tell me a joke.",
        "How does the internet work?",
        "What is the meaning of life?",
        "Describe a beautiful sunset.",
    ]

    print(f"\nConfiguration:")
    print(f"  Max batch size: {max_batch_size}")
    print(f"  Number of requests: {num_requests}")
    print(f"\n{'─' * 60}\n")

    # Submit requests
    requests = []
    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        request = await server.generate(prompt, max_tokens=20)
        requests.append(request)

    print(f"\n{'─' * 60}\n")
    print("Processing requests...\n")

    # Process all requests
    start_time = time.time()
    await server.run_until_complete()
    total_time = time.time() - start_time

    # Print results
    print(f"\n{'─' * 60}")
    print(" RESULTS")
    print(f"{'─' * 60}\n")

    total_tokens = 0
    for req in server.scheduler.completed:
        latency = time.time() - req.created_at
        print(f"Request {req.id}: {len(req.generated_tokens)} tokens, "
              f"{latency:.3f}s latency")
        total_tokens += len(req.generated_tokens)

    print(f"\n{'─' * 60}")
    print(" SUMMARY")
    print(f"{'─' * 60}")
    print(f"Total requests: {num_requests}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Throughput: {total_tokens / total_time:.1f} tokens/second")

    # Explain what's happening
    print(f"\n{'─' * 60}")
    print(" WHAT THIS DEMONSTRATES")
    print(f"{'─' * 60}")
    print("""
1. REQUEST FLOW:
   Prompt → Tokenizer → Scheduler → Model Runner → Response

2. PREFILL vs DECODE:
   - Prefill: Process entire prompt (one request at a time here)
   - Decode: Generate tokens in batches

3. BATCHING:
   - Multiple requests share GPU compute during decode
   - Higher batch size → higher throughput but higher latency

4. CONTINUOUS BATCHING (simplified):
   - New requests can start prefill while others decode
   - Finished requests exit, making room for new ones

5. LIMITATIONS OF THIS DEMO:
   - No actual model (just simulated delays)
   - No KV cache management
   - No memory management
   - No streaming output
   - Simplified scheduling logic
""")


def main():
    parser = argparse.ArgumentParser(description="Minimal Inference Server Demo")
    parser.add_argument("--num-requests", "-n", type=int, default=5,
                        help="Number of requests to process")
    parser.add_argument("--batch-size", "-b", type=int, default=4,
                        help="Maximum batch size")
    args = parser.parse_args()

    asyncio.run(run_demo(args.num_requests, args.batch_size))


if __name__ == "__main__":
    main()
