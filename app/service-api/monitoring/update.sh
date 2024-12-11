#!/bin/bash
kubectl apply -f .
grafd_cm=rakuten-cat-pred-fastapi-grafana-overview
kubectl -n monitoring delete cm $grafd_cm
kubectl -n monitoring create cm $grafd_cm --from-file=fastapi-dashboard.json
kubectl -n monitoring label cm $grafd_cm grafana_dashboard="1"