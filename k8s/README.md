# Kubernetes Setup

## Prerequisites
- Docker installed
- Kubernetes cluster (minikube or Docker Desktop)
- kubectl configured

## Setup

### Install minikube
```bash
brew install minikube
minikube start
```

### Deploy
```bash
./k8s/deploy.sh
```

### Check status
```bash
kubectl get pods
kubectl get services
```

### Access app
```bash
kubectl port-forward service/vlm-tool-service 8000:8000
```

## Files
- **deployment.yaml** - App configuration
- **service.yaml** - Networking  
- **secret.yaml** - API keys
- **deploy.sh** - Deployment script

## Cleanup
```bash
kubectl delete -f k8s/
``` 