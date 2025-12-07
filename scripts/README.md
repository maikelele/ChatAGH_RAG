<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Experiments with Opik](#experiments-with-opik)
  - [Currently supported evaluation metrics:](#currently-supported-evaluation-metrics)
  - [Requirements](#requirements)
  - [Starting and running experiments](#starting-and-running-experiments)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Experiments with Opik
Experiments aim to accurately evaluate current capabilities of the system when faced with questions regarding the AGH University.

## Currently supported evaluation metrics:
- [Response Relevancy](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/answer_relevance/)

## Requirements
- Install and start Docker
```bash

# Debian/Ubuntu
sudo apt install docker
sudo systemctl start docker

# MacOS
TODO
```

- Opik
```bash
# Clone Opik to a location outside of this repository
git clone https://github.com/comet-ml/opik.git

# Navigate to the repository
cd /path/to/opik

# Start the Opik platform
# It should bind to localhost:5173
./opik.sh
```

## Starting and running experiments
1. Change to the `ChatAGH_RAG` repository.
2. Run `poetry install` to install dependencies
3. Run `opik configure` in the CLI
    1. Select `3` for local deployment
4. Run the experiments
```bash
python scripts/run_experiments.py
```
5. View results of your experiment run at `localhost:5173`
