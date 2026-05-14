FROM registry.access.redhat.com/ubi10/python-312-minimal:10.1

WORKDIR /opt/app-root/src

COPY pyproject.toml ./
RUN mkdir -p hearth && touch hearth/__init__.py \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir . \
    && rm -rf hearth

COPY hearth/ hearth/
RUN pip install --no-cache-dir --no-deps .

USER 1001

ENTRYPOINT ["python", "-m", "hearth", "--liveness=http://0.0.0.0:8080/healthz"]
