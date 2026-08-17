"""Standalone OpenPI-compatible WebSocket mock policy server."""

import argparse
import asyncio
import time

import numpy as np


class MockPolicy:
    def __init__(self, mode="noop", chunk_size=16, random_scale=0.05, seed=7):
        if mode not in ("noop", "random"):
            raise ValueError("mode must be noop or random")
        if chunk_size <= 0 or not 0 <= random_scale <= 1:
            raise ValueError("invalid mock parameters")
        self.mode, self.chunk_size, self.random_scale = mode, chunk_size, random_scale
        self.random = np.random.RandomState(seed)
        self.inference_count = 0

    def infer(self):
        started = time.perf_counter()
        actions = np.zeros((self.chunk_size, 7), dtype=np.float32)
        if self.mode == "random":
            actions[:, :6] = self.random.uniform(-self.random_scale, self.random_scale,
                                                  (self.chunk_size, 6)).astype(np.float32)
        if self.mode == "noop":
            # Hold one command for the entire chunk so the actuator has time to
            # move: open on the first request, close on the next, then repeat.
            actions[:, 6] = 1.0 if self.inference_count % 2 == 0 else -1.0
            self.inference_count += 1
        else:
            actions[:, 6] = -1.0
        return {"actions": actions, "metadata": {
            "server_inference_latency_ms": (time.perf_counter() - started) * 1000.0}}


async def serve(args):
    from openpi_client import msgpack_numpy
    import websockets
    policy = MockPolicy(args.mode, args.chunk_size, args.random_scale, args.seed)

    async def handler(websocket):
        await websocket.send(msgpack_numpy.packb({"model_name": "mock/" + args.mode,
            "server_type": "mock", "action_chunk_size": args.chunk_size}))
        async for message in websocket:
            try:
                request = msgpack_numpy.unpackb(message)
                if not isinstance(request, dict):
                    raise ValueError("request must be a mapping")
                await websocket.send(msgpack_numpy.packb(policy.infer()))
            except Exception as exc:
                await websocket.send(str(exc))

    async with websockets.serve(handler, args.host, args.port):
        print("mock policy ready: ws://{}:{} model=mock/{} chunk={}".format(
            args.host, args.port, args.mode, args.chunk_size), flush=True)
        await asyncio.Future()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mode", choices=("noop", "random"), default="noop")
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        asyncio.run(serve(args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
