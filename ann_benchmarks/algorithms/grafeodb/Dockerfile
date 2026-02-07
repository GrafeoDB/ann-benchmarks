FROM ann-benchmarks

RUN apt-get update && apt-get install -y curl build-essential pkg-config libssl-dev
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

RUN pip install maturin

# Build grafeo from source with vector index support
COPY grafeo /tmp/grafeo
RUN cd /tmp/grafeo && maturin build --release -m crates/bindings/python/Cargo.toml && \
    pip install target/wheels/grafeo-*.whl

RUN python -c 'from grafeo import GrafeoDB; print("grafeo OK")'
