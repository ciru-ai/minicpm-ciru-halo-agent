# MiniCPM 4.7 Ultra: measured performance

Measurements collected October 14–15, 2026 on one AMD Ryzen AI Max+ 395 / Radeon 8060S (`gfx1151`) machine with 128 GB unified memory, running NixOS. The serving configuration uses 262,144 maximum context, eight request slots, a 24 GiB KV budget, native vision, prefix caching, and adaptive DSpark depth three for one active request and two for concurrent requests.

These results describe this hardware and custom runtime. They are not estimates for stock vLLM or other GPUs. Model-loading/startup time is excluded from request and task wall times.

## Long-context single-user speed

Thinking off; temperature 0.6; 63,870 prompt tokens; 512 output tokens. The fixed output length is a speed probe, not a completed quality episode.

| Prefix state | Newly processed prompt tokens | PP tokens/s | TG tokens/s | Request wall | Draft acceptance |
|---|---:|---:|---:|---:|---:|
| Uncached | 63,870 | 1,968.59 | 40.45 | 45.136 s | 54.92% |
| Cached | 622 | 1,063.70 | 40.73 | 13.185 s | 54.92% |

## Packaged serving check

Original captured vision/tool request bodies, with thinking enabled, temperature 1.0, top-p 0.95, top-k 20, repetition penalty 1.05, and the full remaining context available for completion. All requests completed naturally. The check validates serving and tool-call structure; it does not execute scene-building actions.

| Workload | Output tokens | PP tokens/s | Pooled TG/request | Aggregate TG | Wall | Draft acceptance |
|---|---:|---:|---:|---:|---:|---:|
| Single request, uncached prefix | 441 | 2,554.76 | 40.55 | 40.58 | 13.977 s | 44.21% |
| Single request, cached prefix | 501 | 2,052.06 | 43.38 | 43.41 | 12.112 s | 49.92% |
| Eight concurrent requests | 7,483 | 722.67 | 21.49 | 122.78 | 63.232 s | 78.05% |

Ten of ten requests passed tokenizer parity and returned valid tool names/JSON. Actual target execution reached eight concurrent requests. The two single-request rows generated different continuations; they are observed timings, not a controlled cache speedup ratio.

## Autonomous 3D graphics task

An eight-agent harness started from an empty scene and the original AMD graphics task. All eight roles finished, producing a 3D scene with AMD letters, symbol, pedestal, particles, lighting, and a dashboard. The model repaired two generated-design errors through its own workflow: a shader issue and a dashboard schema issue. No manual design edits or coaching were supplied during execution.

| Measurement | Result |
|---|---:|
| Complete task wall, after readiness | **141.813 s** |
| First visible contribution | 49.357 s |
| Completed model requests | 19 |
| Generated tokens | 12,992 |
| PP, newly computed tokens / summed prefill time | 1,015.34 tokens/s |
| Aggregate TG during active decode | 99.80 tokens/s |
| Pooled per-request TG | 19.82 tokens/s |
| Output tokens / entire task wall | 91.61 tokens/s |
| Draft acceptance | 71.45% |
| Serving/transport failures or preemptions | 0 |
| Vision-cache hits | 19/29 |

Thinking enabled, temperature 1.0, vision enabled. All 19 completion budgets matched the remaining 262,144-token context; none was truncated. The two fully solo decode turns ran at 42.58 and 42.75 tokens/s. The workflow is a practical demonstration, not a standardized benchmark score. A model-name label in the viewer was corrected after timing; the generated scene and original benchmark record were preserved.

## Numerical fidelity

Matched token prefixes against the original BF16 model; identical maximum-logprob/tie-set extraction on both sides. Tied leaders are excluded from the strict rate and reported separately.

| Panel | Matching unique leaders | Agreement | Scored positions | Ties excluded |
|---|---:|---:|---:|---:|
| Frozen prefill: four histories | 23,143 / 23,940 | 96.6708% | 24,330 | 390 |
| Live single-request continuations | 2,310 / 2,328 | 99.2268% | 2,341 | 13 |
| Live eight-request cohort | 6,182 / 6,231 | 99.2136% | 6,285 | 54 |

The live panels use actual Q8 target distributions and a BF16 teacher-forced evaluation of the identical accepted token history. They do not establish equivalence to a separate BF16 serial-decode implementation. Coverage includes tool transitions, code, technical documents, context lengths up to 63,870 input tokens, and naturally completed continuations. These fidelity checks are text-only; vision completion is verified separately in the 3D task. Top-20 observations do not provide full-vocabulary KL.

The frozen panel was 96.7173% on an earlier Q8 runtime. On the common set of unique-leader positions, the current runtime recovers 226 matches and loses 244. This small aggregate difference does not establish universal parity or settle every local discrepancy. No lossless-quantization claim is made.

The packaged installation additionally reproduces 4,929/4,929 frozen top-token sets from the validated Q8 runtime. This is a package-preservation check, not a second BF16 result.

## Reading the timings

- **PP:** newly computed prompt tokens divided by summed native prefill time. With concurrent work, prefill, decoding, and scheduling interact; PP is not whole-system throughput.
- **Pooled TG/request:** output tokens divided by summed request decode durations. Those durations overlap during concurrency.
- **Aggregate TG:** output tokens divided by the union of first-token-to-response-end intervals across requests.
- **Wall:** the complete request/cohort/task duration after readiness, including the relevant processing and orchestration.
- **Acceptance:** accepted draft tokens divided by proposed draft tokens. It is not a quality score.

The final mixed eight-request regression screen also recorded PP 1,215.26, pooled TG 10.45, and 43.402 s wall; its cached repeat recorded PP 384.54, TG 18.18, and 16.916 s wall. Against two bracketing earlier-runtime controls, PP was 4.33% lower uncached and 2.80% lower cached. The concurrent schedules and generated tokens differed, so those observations do not isolate a kernel regression; they remain part of the evidence and are not presented as speedups.

Warmup reduces specific first-use costs but does not eliminate all compilation. The complete graphics task still logged two first-use GDN warnings whose separate latency was not isolated. Startup depends on compiler caches: measured startup with existing artifacts is not an empty-cache guarantee.

Older agent-suite scores are not included as headline quality results because the suite had grading and termination caveats. No new Tool Eval score is claimed.
