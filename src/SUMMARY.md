# Summary

[Introduction](./introduction.md)

---

# Part I: Distributed Computing

- [Chapter 1: Your First Distributed Program](./part1-distributed/chapter01.md)
  - [verify_setup.py](./part1-distributed/chapter01-verify.md)
  - [hello_distributed.py](./part1-distributed/chapter01-hello.md)
- [Chapter 2: Point-to-Point Communication](./part1-distributed/chapter02.md)
  - [send_recv_basic.py](./part1-distributed/chapter02-send.md)
  - [pipeline_simulation.py](./part1-distributed/chapter02-pipeline.md)
- [Chapter 3: Collective Operations](./part1-distributed/chapter03.md)
  - [collective_cheatsheet.py](./part1-distributed/chapter03-cheatsheet.md)
  - [distributed_mean.py](./part1-distributed/chapter03-mean.md)
- [Chapter 4: NCCL and GPU Topology](./part1-distributed/chapter04.md)
  - [topology_inspector.py](./part1-distributed/chapter04-topology.md)
  - [benchmark_algorithms.py](./part1-distributed/chapter04-benchmark.md)

---

# Part II: Parallelism Strategies

- [Chapter 5: Data Parallelism Deep Dive](./part2-parallelism/chapter05.md)
  - [simple_ddp.py](./part2-parallelism/chapter05-ddp.md)
  - [gradient_sync_visualizer.py](./part2-parallelism/chapter05-gradient.md)
- [Chapter 6: Tensor Parallelism](./part2-parallelism/chapter06.md)
  - [tp_linear.py](./part2-parallelism/chapter06-linear.md)
  - [tp_mlp.py](./part2-parallelism/chapter06-mlp.md)
- [Chapter 7: Pipeline & Expert Parallelism](./part2-parallelism/chapter07.md)
  - [pipeline_schedule_viz.py](./part2-parallelism/chapter07-pipeline.md)
  - [parallel_strategy_calculator.py](./part2-parallelism/chapter07-calculator.md)

---

# Part III: LLM Inference Systems

- [Chapter 8: Server Anatomy](./part3-inference/chapter08.md)
  - [minimal_inference_server.py](./part3-inference/chapter08-server.md)
- [Chapter 9: KV Cache Management](./part3-inference/chapter09.md)
  - [kv_cache_calculator.py](./part3-inference/chapter09-calculator.md)
  - [prefix_sharing_demo.py](./part3-inference/chapter09-prefix.md)
- [Chapter 10: Scheduling & CUDA Graphs](./part3-inference/chapter10.md)
  - [cuda_graph_simple.py](./part3-inference/chapter10-cuda.md)
  - [scheduling_overhead_benchmark.py](./part3-inference/chapter10-benchmark.md)
- [Chapter 11: Speculative & Constraint Decoding](./part3-inference/chapter11.md)
  - [speculative_demo.py](./part3-inference/chapter11-speculative.md)
  - [json_constraint_demo.py](./part3-inference/chapter11-json.md)

---

# Part IV: RLHF Systems

- [Chapter 12: RL Fundamentals for LLMs](./part4-rlhf/chapter12.md)
  - [ppo_cartpole.py](./part4-rlhf/chapter12-ppo.md)
  - [gae_visualizer.py](./part4-rlhf/chapter12-gae.md)
- [Chapter 13: RLHF Computation Flow](./part4-rlhf/chapter13.md)
  - [rlhf_loop_pseudo.py](./part4-rlhf/chapter13-loop.md)
  - [reward_calculator.py](./part4-rlhf/chapter13-reward.md)
- [Chapter 14: RLHF System Architecture](./part4-rlhf/chapter14.md)
  - [weight_update_demo.py](./part4-rlhf/chapter14-weight.md)
  - [memory_timeline.py](./part4-rlhf/chapter14-memory.md)
