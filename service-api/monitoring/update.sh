#!/bin/bash
kubectl apply -f .
grafd_cm=rakuten-cat-pred-grafana-overview
kubectl -n monitoring delete cm $grafd_cm
kubectl -n monitoring create cm $grafd_cm --from-file=grafana-dashboard.json
kubectl -n monitoring label cm $grafd_cm grafana_dashboard="1"