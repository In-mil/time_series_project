# Deployment Scripts

Scripts for building and deploying the application to GCP Cloud Run.

## Contents
- `build_docker.sh` - Builds and pushes Docker image
- `check_deployment_status.sh` - Monitors Cloud Run deployment status
- `get_service_url.sh` - Retrieves the deployed service URL

## Usage
```bash
# From project root
./deployment/build_docker.sh
./deployment/get_service_url.sh
```

## Note
Docker configuration files (Dockerfile, docker-compose.yml) remain in project root per Docker conventions.
