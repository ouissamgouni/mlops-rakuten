<h1 align="center">MLops: Rakuten prediction system for product categorisation</h1>

Short description of the project

## Design

![alt text](docs/mlops_design.jpg)

## Installation

### Infra

There are only two prerequisites:

* [Docker](https://docs.docker.com/get-docker/)
* docker-compose
* minikube
* helm
``` bash
minikube start
```

* [Flyte](https://github.com/davidmirror-ops/flyte-the-hard-way/blob/main/docs/on-premises/single-node/002-single-node-onprem-install.md)
* [Mlflow](https://medium.com/@heisash24/-84bd8496f360)
* [Monitoring stack (Prometheus & Grafana)](https://medium.com/@brightband/deploying-prometheus-operator-to-a-kubernetes-cluster-c2378038c79b)
* [Private docker registry](https://hub.docker.com/r/ogouni604/mlops-rakuten) 


## Usage

``` bash
cd workflow
pyflyte run --remote workflow.py my_wf
```


## How it looks like

<p align="center">
  <img src="./dashboard.jpeg">
</p>

## References


