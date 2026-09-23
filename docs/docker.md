# Docker

## Build

```bash
docker build -t bayesian-classifier-lab .
```

## Run the included demonstration

```bash
docker run --rm \
  -v "${PWD}/results:/app/results" \
  bayesian-classifier-lab
```

On Windows PowerShell:

```powershell
docker run --rm -v "${PWD}/results:/app/results" bayesian-classifier-lab
```

Or use Compose:

```bash
docker compose up --build
```

The default image generates a deterministic synthetic demonstration dataset during the image build and runs the configured 10-fold x 10-repeat experiment.

## Use your own data

Mount a local data directory over `/app/data` and provide a configuration file that points to the mounted CSV:

```bash
docker run --rm \
  -v "${PWD}/data:/app/data:ro" \
  -v "${PWD}/results:/app/results" \
  -v "${PWD}/config.yml:/app/config.yml:ro" \
  bayesian-classifier-lab \
  bayesclf --config /app/config.yml
```

Do not put confidential laboratory data into a public Docker image or public Git repository.
