#!/bin/bash

echo "Deploying to Kubernetes..."

docker build -t vlm-tool-recognition:latest .

kubectl apply -f k8s/secret.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

echo "Done!"
echo ""
echo "kubectl get pods"
echo "kubectl get services" 