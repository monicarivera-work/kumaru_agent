# syntax=docker/dockerfile:1.7
#
# Kumaru's code sandbox: the image `kumaru bench run --sandbox docker` grades
# HumanEval and MultiPL-E in.
#
# STATUS: NOT YET BUILT OR TESTED. It was written alongside the MultiPL-E
# benchmark, on a machine where Docker was not set up; the local runner
# (no --sandbox) is what has been checked. Expect to fix small things on the
# first build, and run `python scripts/check_graders.py humaneval multipl_e
# --sandbox docker` against it before trusting its results (see
# docs/BENCHMARKS.md).
#
# Build it once, from the Kumaru folder (about 3 GB):
#
#   docker build -t kumaru-sandbox -f docker/sandbox/Dockerfile .
#
# then grade inside it:
#
#   kumaru bench run multipl_e -c <config> --allow-code-exec --sandbox docker
#
# Each program runs in its own container, started by
# src/kumaru/eval/benchmarks/codeexec.py (docker_command): no network, a
# read-only root, user nobody (65534), 2 CPUs, 2 GB of memory, 512 processes,
# and only /work and /tmp writable (in memory). So everything a program needs
# is installed here, at build time; nothing is downloaded while grading.
# KUMARU_SANDBOX_IMAGE names another image.
#
# The tools must be where multipl_e.py (_docker_toolchain) and humaneval.py
# look: python3, node, dotnet, javac, java, g++, go and rustc on PATH; tsc at
# /opt/kumaru/typescript/node_modules/typescript/lib/tsc.js; the javatuples
# jar at /opt/kumaru/javatuples-1.2.jar; a warm Go build cache at /opt/gocache.

ARG GO_VERSION=1.25
ARG RUST_VERSION=1.90
ARG NODE_VERSION=24
ARG TYPESCRIPT_VERSION=5.9.3

FROM golang:${GO_VERSION}-bookworm AS go

# The toolchain itself, not rustup's proxies, which want a writable home.
FROM rust:${RUST_VERSION}-slim-bookworm AS rust
RUN cp -r "$(rustc --print sysroot)" /opt/rust

FROM node:${NODE_VERSION}-bookworm-slim AS node
ARG TYPESCRIPT_VERSION
RUN npm install --prefix /opt/kumaru/typescript --no-audit --no-fund "typescript@${TYPESCRIPT_VERSION}"

# .NET 10 builds MultiPL-E's C# (Ubuntu 24.04, which also has OpenJDK 21).
FROM mcr.microsoft.com/dotnet/sdk:10.0-noble

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 g++ openjdk-21-jdk-headless \
 && rm -rf /var/lib/apt/lists/*

COPY --from=go /usr/local/go /usr/local/go
COPY --from=rust /opt/rust /opt/rust
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /opt/kumaru/typescript /opt/kumaru/typescript

# MultiPL-E's Java tests import org.javatuples. Checked against Maven Central's SHA-256.
ADD --chmod=644 --checksum=sha256:2eda5b19d9820e1cc2f69fcd01639a715a673c11f8507e3d1ed593cf765d5e0a \
    https://repo1.maven.org/maven2/org/javatuples/javatuples/1.2/javatuples-1.2.jar \
    /opt/kumaru/javatuples-1.2.jar

ENV PATH=/usr/local/go/bin:/opt/rust/bin:$PATH \
    GOTOOLCHAIN=local GOPROXY=off GOFLAGS=-buildvcs=false CGO_ENABLED=0 GOENV=off \
    DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1 DOTNET_SKIP_FIRST_TIME_EXPERIENCE=1 \
    DOTNET_CLI_WORKLOAD_UPDATE_NOTIFY_DISABLE=1

# Go compiles the standard library packages a test needs on first use. Doing
# that here, once, saves every program the time; each container copies this
# cache to /tmp, since Go must be able to write to its cache.
COPY <<'EOF' /tmp/warm/warm_test.go
package warm_test

import (
	"fmt"
	"math"
	"sort"
	"strconv"
	"strings"
	"testing"
	"unicode"
)

func TestWarm(t *testing.T) {
	_ = fmt.Sprint(math.Abs(-1), strconv.Itoa(1), strings.ToUpper("a"), unicode.IsDigit('1'))
	sort.Ints([]int{2, 1})
}
EOF
RUN cd /tmp/warm \
 && GOCACHE=/opt/gocache go test -c -o warm.test warm_test.go \
 && ./warm.test \
 && cd / && rm -rf /tmp/warm \
 && chmod -R a+rX /opt/gocache /opt/kumaru /opt/rust \
 && python3 --version && node --version && dotnet --version && javac -version \
 && g++ --version && go version && rustc --version \
 && node /opt/kumaru/typescript/node_modules/typescript/lib/tsc.js --version

WORKDIR /work
USER 65534:65534
