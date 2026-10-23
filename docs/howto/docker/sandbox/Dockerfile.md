# `docker\sandbox\Dockerfile`

> The image `--sandbox docker` grades code benchmarks in: every toolchain HumanEval and MultiPL-E need (Python, Node.js with TypeScript, .NET 10, a JDK with javatuples, g++, Go, Rust), installed at build time, so programs run without network access as an unprivileged user.

**Read this when:** you want model-written code to run in a container instead of on your machine, `--sandbox docker` says the image is missing, or you are adding a toolchain to the sandbox.

---

> **Status: built and checked** on 2026-10-23, with Docker Desktop 4.39.0 (engine 28.0.1, WSL 2). It built the first time, as written, in about 3 minutes, to 3.54 GB. Graded in it, [`check_graders.py`](../../scripts/check_graders.py.md) reproduces HumanEval's verdicts and MultiPL-E's in all 8 languages as closely as the local runner does ([`BENCHMARKS.md` §11.4](../../../BENCHMARKS.md#114-are-its-graders-right)). Rerun that check after a rebuild, since the Ubuntu packages aren't pinned.

## What it does

A multi-stage build. Each base image is pinned by digest (its tag's multi-arch index on 2026-10-21), as is the `docker/dockerfile:1.7` syntax frontend:

| Stage | Base image (`ARG`) | What it contributes |
|---|---|---|
| `go` | `golang:1.25-bookworm@sha256:3b4a1151…` (`GO_IMAGE`) | `/usr/local/go` |
| `rust` | `rust:1.90-slim-bookworm@sha256:64232e65…` (`RUST_IMAGE`) | the toolchain itself (`rustc --print sysroot`), copied to `/opt/rust`, without rustup's proxies, which want a writable home |
| `node` | `node:24-bookworm-slim@sha256:55fb5f5a…` (`NODE_IMAGE`) | `node`, and TypeScript 5.9.3 (`TYPESCRIPT_VERSION`) installed under `/opt/kumaru/typescript` |
| final | `mcr.microsoft.com/dotnet/sdk:10.0-noble@sha256:747cbade…` (`DOTNET_IMAGE`, Ubuntu 24.04) | the .NET 10 SDK, plus `python3`, `g++` and `openjdk-21-jdk-headless` from Ubuntu (not pinned: whatever the archive has at build time); the stages above copied in |

Then it:

- downloads `javatuples-1.2.jar` from Maven Central to `/opt/kumaru/javatuples-1.2.jar`, checked against its SHA-256 (`ADD --checksum`);
- sets the same offline Go settings Kumaru uses locally (`GOTOOLCHAIN=local`, `GOPROXY=off`, `CGO_ENABLED=0`, …) and turns off .NET's telemetry and first-run messages;
- builds and runs a small Go test once with `GOCACHE=/opt/gocache`, so the standard-library packages the tests use are compiled ahead of time; each container copies that cache to `/tmp/gocache` (a setup step) because Go writes to its cache;
- prints every tool's version, so a broken stage fails the build;
- ends as `USER 65534:65534` (nobody) in `/work`.

What the 2026-10-23 build contains:

| Tool | Version | Size |
|---|---|---|
| Python | 3.12.3 (Ubuntu's `python3` 3.12.3-0ubuntu2.1) | |
| Node.js, TypeScript | 24.22.1, 5.9.3 | 23 MB for TypeScript |
| .NET SDK | 10.0.402 (runtime 10.0.13) | 624 MB |
| OpenJDK | 21.0.12.1 (`openjdk-21-jdk-headless` 21.0.12.1+1-1~24.04.4), javatuples 1.2 | 286 MB |
| g++ | 13.3.0 (`g++-13` 13.3.0-6ubuntu2~24.04.1, `libstdc++6` 14.2.0-4ubuntu2~24.04.1) | |
| Go | 1.25.14, with a 55 MB warm build cache | 243 MB |
| rustc | 1.90.0 (LLVM 20.1.8) | 567 MB |

## Why it exists

Without a sandbox, model-written programs run as you, with your files and your network ([`codeexec.md`](../../src/kumaru/eval/benchmarks/codeexec.md)). A container started with `--network none`, a read-only root, no capabilities, and limits on CPU (2), memory (2 GB, no swap) and processes (512) is a real boundary. Programs can't download anything inside it, so everything they need has to be in the image already.

## Mental model

```
docker build  ->  kumaru-sandbox (3.54 GB, about 3 minutes)
kumaru bench run ... --sandbox docker
  -> preflight: docker image inspect kumaru-sandbox  (its ID goes into the results fingerprint; if that
                fails, a container running /bin/true wakes a paused Docker Desktop, and it asks again)
  -> per answer: docker run --rm -i --pull never --network none --read-only ...
                   --entrypoint /bin/sh kumaru-sandbox -c <script>
       stdin: a tar of the program's files -> /work (in memory)
       script: each step under `timeout`, results back as records on stdout
```

Kumaru finds the tools by the names and paths below, so they must stay there:

| Tool | Where |
|---|---|
| `python3`, `node`, `dotnet`, `javac`, `java`, `g++`, `go`, `rustc` | on `PATH` |
| `tsc` | `/opt/kumaru/typescript/node_modules/typescript/lib/tsc.js` (`multipl_e.DOCKER_TSC`) |
| javatuples | `/opt/kumaru/javatuples-1.2.jar` (`multipl_e.DOCKER_JAVATUPLES`) |
| Go build cache | `/opt/gocache`, copied to `/tmp/gocache` |

## How to use it

Build it once, from the Kumaru folder:

```powershell
docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile .
```

It needs about 11 GB free where Docker keeps its data: after the build, `docker system df` counted 7.6 GB of images (the 3.54 GB image, plus layers it was built from) and 3.4 GB of build cache. On Windows that is Docker Desktop's virtual disk (`docker_data.vhdx`), which grows as needed but doesn't shrink by itself when images are deleted. Building pulls 4 images from Docker Hub, which allows anonymous users about 10 pulls an hour, so rebuild from the cache rather than after `docker system prune`.

Check that its graders agree with HumanEval's and MultiPL-E's own verdicts, then grade in it:

```powershell
python scripts\check_graders.py humaneval multipl_e --sandbox docker --workers 2
kumaru bench run humaneval multipl_e -c qwen36 --no-think --allow-code-exec --sandbox docker
```

`KUMARU_SANDBOX_IMAGE` selects another image (a tag or an ID). The image's ID is part of the results fingerprint, so a rebuilt image starts a new run instead of resuming an old one.

Each base image can be replaced at build time, with its tag and digest together so the pin stays a pin, for example `--build-arg GO_IMAGE=golang:1.24-bookworm@sha256:<digest>`; `docker buildx imagetools inspect golang:1.24-bookworm` prints the digest. `--build-arg TYPESCRIPT_VERSION=…` changes TypeScript. Different versions can grade differently ([`multipl_e.md`](../../src/kumaru/eval/benchmarks/multipl_e.md), differences from MultiPL-E).

## How to extend it

- **A new language:** install its toolchain here (a stage of its own if it has an official image), put it on `PATH`, add its plain name to `_docker_toolchain` in `multipl_e.py`, and add a version check to the final `RUN`.
- **Anything a program needs at run time** (a library, a jar, a warm cache) must be in the image: containers have no network.
- **Keep it unprivileged:** files the programs read must be world-readable (`chmod -R a+rX`), since they run as `nobody`.

## Failure modes

| Symptom | Likely cause | Fix |
|---|---|---|
| `Docker has no image 'kumaru-sandbox', or the Docker daemon is not running` | Not built yet, or Docker Desktop isn't running. | Start Docker, then build it (above). |
| `docker image inspect kumaru-sandbox` says `No such image`, though `docker image ls` lists it | Docker Desktop's [Resource Saver](https://docs.docker.com/desktop/use-desktop/resource-saver/) paused the engine after 5 minutes without a container. Until a container starts, Docker answers other commands without waking it, and finds the image by its ID but not by its name. Kumaru's preflight starts one (`/bin/true`) before it calls the image missing, so only commands you type see this. | Start any container from it (`docker run --rm kumaru-sandbox true`), or turn Resource Saver off in Docker Desktop's **Settings > Resources**. |
| The build fails at `ADD --checksum` | Maven Central served a different file, or an old Docker without BuildKit. | Check the download; use a current Docker (BuildKit is the default). |
| The build fails at `FROM` with `not found` for a `sha256:` digest | The registry no longer serves that pinned digest. | Move the pin: the same tag with its current digest (`docker buildx imagetools inspect <tag>`), passed as `--build-arg` or edited in the file. |
| The build fails at the final version checks | A stage changed its layout (e.g. a new base image). | Read which tool failed and fix its `COPY` or `PATH`. |
| `Sandbox setup step 'cp -R /opt/gocache /tmp/gocache' failed` | The image lacks the warm Go cache. | Rebuild from this file. |
| Every C# answer fails or times out in the sandbox | `dotnet build` tries to reach the network or a writable folder that isn't there. The image as built doesn't: 199 of 200 C# verdicts match MultiPL-E's. | Check `HOME` and `DOTNET_CLI_HOME` (both `/tmp/home`) and the `#:property` lines in `multipl_e.py`, then run `check_graders.py multipl_e_cs --sandbox docker`. |
| `%TEMP%\wsl-crashes` fills with `wsl-crash-*.dmp` files, up to 1.1 GB each | Docker Desktop's WSL VM sends the core dump of every crashed process to Windows. Kumaru's containers get `--ulimit core=1`, the only limit that stops them (0 doesn't), so a container started some other way. | Delete the files; start containers with `--ulimit core=1`. |

## Related files

- `src\kumaru\eval\benchmarks\codeexec.py`: starts the containers (`docker_command`, `docker_script`) ([`codeexec.md`](../../src/kumaru/eval/benchmarks/codeexec.md)).
- `src\kumaru\eval\benchmarks\multipl_e.py`: the tools each language uses in the image (`_docker_toolchain`) ([`multipl_e.md`](../../src/kumaru/eval/benchmarks/multipl_e.md)).
- `src\kumaru\eval\benchmarks\humaneval.py`: runs `python3` in it ([`humaneval.md`](../../src/kumaru/eval/benchmarks/humaneval.md)).
- `scripts\check_graders.py`: `--sandbox docker` checks the image ([`check_graders.py.md`](../../scripts/check_graders.py.md)).
