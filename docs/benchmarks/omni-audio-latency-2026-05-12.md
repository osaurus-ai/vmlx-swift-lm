# Nemotron Omni Audio Latency Bench - 2026-05-12

Host: local M5 Max MacBook.

Model: `Nemotron-Omni-Nano-JANGTQ-CRACK`

Bench executable: `swift run OmniAudioLatencyBench`

## Command

```sh
BENCH_OMNI_AUDIO_PATH=batch \
BENCH_OMNI_AUDIO_PREENCODE=1 \
BENCH_OMNI_AUDIO_DISK_CACHE=1 \
BENCH_AUDIO_REPEATS=2 \
BENCH_AUDIO_FILE=Tests/MLXLMTests/Resources/audio_only.mov \
BENCH_MAX_TOKENS=8 \
BENCH_MODEL=<absolute path to Nemotron-Omni-Nano-JANGTQ-CRACK> \
/usr/bin/time -l .build/debug/OmniAudioLatencyBench
```

## Result

Audio fixture: `Tests/MLXLMTests/Resources/audio_only.mov`, 80,620 samples,
5,038.8 ms at 16 kHz.

Load: 2,295.6 ms, RSS 5,265.9 MiB.

Pre-encode: 1,777.8 ms for 63 audio tokens at hidden size 2,688.

| Path | Mode | Turn | First semantic delta | Total | Tokens | Effective tok/s | RSS MiB | Peak RSS MiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BatchEngine | raw PCM | 1 | 2,180.7 ms | 2,302.9 ms | 8 | 3.5 | 5,328.8 | 5,292.7 |
| BatchEngine | raw PCM | 2 | 1,473.8 ms | 1,590.6 ms | 8 | 5.0 | 5,355.4 | 5,354.1 |
| BatchEngine | pre-encoded Parakeet | 1 | 200.0 ms | 301.1 ms | 8 | 26.6 | 5,357.8 | 5,355.5 |
| BatchEngine | pre-encoded Parakeet | 2 | 211.9 ms | 327.2 ms | 8 | 24.4 | 5,360.9 | 5,360.8 |

`/usr/bin/time -l`: 9.40 s real, max RSS 7,747,796,992 bytes, swaps 0,
peak memory footprint 15,140,101,296 bytes.

## Read

This bench measures first semantic text delta, not output TTS first audio
byte. Raw PCM is still over the awkward-pause threshold on this fixture
because Parakeet audio encoding and multimodal prefill happen after endpoint.
Passing pre-encoded Parakeet embeddings into `UserInput.Audio.preEncoded`
brings first semantic delta to about 200 ms on the Osaurus BatchEngine path.

Turn 2 raw PCM is faster than turn 1, but it is still roughly 1.5 s. Do not
count raw identical-audio repeat as a solved conversational prefix-cache path
yet; the live-call path should stream/accumulate Parakeet embeddings while
the caller is speaking and submit pre-encoded audio at endpoint.
