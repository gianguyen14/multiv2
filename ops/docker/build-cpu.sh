# CPU Docker build: uses the default (CPU) PyTorch wheel.
docker build --pull --progress=plain \
  -t gianguyen14/aic-retrieval:sha-$(git rev-parse --short=12 HEAD) \
  -t gianguyen14/aic-retrieval:finals-20260917 \
  .
