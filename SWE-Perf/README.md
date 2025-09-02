# Mini-SWE-Agent Evaluation on SWE-Perf Benchmark

This repository contains the evaluation pipeline for running [Mini-SWE-Agent](https://github.com/SWE-agent/mini-swe-agent) on the [SWE-Perf](https://github.com/SWE-Perf/SWE-Perf) benchmark to assess code performance optimization capabilities.

## 📦 Environment Setup

### Prerequisites

- **Python 3.11+**
- **Docker Desktop** (for evaluation pipeline)
- **Git**
- **Conda** (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/gjz78910/GreenerAgent.git
cd GreenerAgent
```

2. **Create conda environment:**
```bash
conda env create -f environment.yml
conda activate sweperf
```

3. **Install additional dependencies:**
```bash
pip install docker datasets tqdm numpy scipy pandas matplotlib requests python-dotenv
```

### Required API Keys

Set the following environment variables:

```bash
# For Claude models (Anthropic)
export ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# For accessing datasets on Hugging Face
export HF_TOKEN="your_huggingface_token_here"
```

## 🚀 Quick Start

### 1. Generate Predictions

Run Mini-SWE-Agent on SWE-Perf instances:

```bash
# Test with 1 instance
python -m minisweagent.run.extra.sweperf \
  --subset sweperf \
  --split test \
  --slice 0:1 \
  --config mini-swe-agent/src/minisweagent/config/extra/sweperf_oracle.yaml \
  --workers 1

# Run on full dataset (140 instances)
python -m minisweagent.run.extra.sweperf \
  --subset sweperf \
  --split test \
  --config mini-swe-agent/src/minisweagent/config/extra/sweperf_oracle.yaml \
  --workers 4
```

**Output (the generated patches)**: `results/sweperf_predictions_YYYYMMDD_HHMMSS.jsonl`

### 2. Run Evaluation

```bash
python -m evaluation.run_evaluation \
  --dataset_name SWE-Perf/SWE-Perf \
  --split test \
  --predictions_path results/sweperf_predictions_YYYYMMDD_HHMMSS.jsonl \
  --max_workers 1 \
  --run_id mini_swe_agent_YYYYMMDD_HHMMSS
```

**Output (the runtime measurements)**: `results/mini_swe_agent_YYYYMMDD_HHMMSS/`

### 3. Calculate Metrics

```bash
python -m evaluation.check_evaluation \
  --dataset_dir SWE-Perf/SWE-Perf \
  --log_root results/mini_swe_agent_YYYYMMDD_HHMMSS/claude-3-5-sonnet-20241022 \
  --output_path results/performance_metrics_YYYYMMDD_HHMMSS.csv
```

**Output (the final SWE-Perf metrics)**: `results/performance_metrics_YYYYMMDD_HHMMSS.csv`

## ⚙️ Configuration

### Model Configuration

The system supports multiple language models through configuration files:

#### Example Models

| Model | Config Parameter | Description |
|-------|------------------|-------------|
| **Claude 3.5 Sonnet** | `claude-3-5-sonnet-20241022` | Default, balanced performance |
| **Claude 4 Sonnet** | `claude-sonnet-4-20250514` | Latest model, improved reasoning |
| **Claude 4 Opus** | `claude-4-opus-20240229` | Most capable, highest cost |

#### Override Model via Command Line

```bash
# Use Claude 4 Sonnet instead of default
python -m minisweagent.run.extra.sweperf \
  --subset sweperf \
  --split test \
  --config mini-swe-agent/src/minisweagent/config/extra/sweperf_oracle.yaml \
  --model claude-sonnet-4-20250514 \
  --workers 4
```

### Configuration Files

The repository includes multiple configuration versions:

- **`sweperf_oracle.yaml`**: Default configuration

#### Example Configuration

```yaml
# sweperf_oracle.yaml
environment:
  environment_class: local
  timeout: 60
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: 'off'
    TQDM_DISABLE: '1'

agent:
  name: DefaultAgent
  max_steps: 50
  
model:
  model_name: claude-3-5-sonnet-20241022
  per_instance_cost_limit: 3.0
  total_cost_limit: 0.0
```

## 📊 Understanding Results

### Key Metrics

The evaluation produces comprehensive performance metrics:

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| **Apply Rate** | % of patches that apply successfully | Higher = better compatibility |
| **Correctness** | % of patches that pass all tests | Higher = more reliable |
| **Performance** | Average execution time improvement | Higher = better optimization |

### Performance Calculation

Performance improvement is calculated as:

1. **Test Level**: Percentage improvement for each test
2. **Instance Level**: Average of all test improvements
3. **Project Level**: Sum of instance improvements / total instances

**Example**: If two instances show 45% and 38% improvement:
- Total improvement: 0.45 + 0.38 = 0.83
- Performance score: 0.83 ÷ 2 = **41.55%**

### Statistical Significance

The system uses Mann-Whitney U tests (p<0.1) to ensure improvements are statistically significant, not just measurement noise.

## 🏗️ Architecture

### Integration Pipeline

```
SWE-Perf Dataset → Mini-SWE-Agent → Git Diff → JSONL Predictions
       ↓
Docker Evaluation → Performance Tests → Statistical Analysis
       ↓
CSV Metrics Report (Human vs Model Comparison)
```

### Key Components

- **`minisweagent/run/extra/sweperf.py`**: SWE-Perf integration
- **`evaluation/`**: Docker-based evaluation pipeline
- **`mini-swe-agent/src/minisweagent/config/extra/`**: Agent configurations

## 🔧 Hyperparameters to Tune

All hyperparameters are configured in YAML files located in:
```
SWE-Perf/mini-swe-agent/src/minisweagent/config/extra/sweperf_oracle.yaml
```

### Model Hyperparameters

```yaml
model:
  model_name: "mistralai/Mistral-Large-Instruct-2407"
  model_kwargs:
    temperature: 0.0        # 0.0-1.0: creativity vs consistency
    top_p: 1.0             # 0.1-1.0: nucleus sampling diversity
    top_k: 50              # 1-100: top-k sampling limit
    max_tokens: 2048       # Maximum response length
```

### Agent Hyperparameters

```yaml
agent:
  step_limit: 40           # Maximum steps (0 = unlimited)
  cost_limit: 3.0          # Maximum cost in $ (0 = unlimited)
```

### Prompt Templates

#### System Template (Core Behavior)
```yaml
agent:
  system_template: |
    You are a helpful assistant that can interact with a computer shell.
    Your response must contain exactly ONE bash code block with ONE command.
    Include a THOUGHT section before your command explaining your reasoning.
```

#### Instance Template (Task Presentation)
```yaml
agent:
  instance_template: |
    <performance_optimization_task>
    Consider the following performance optimization task:
    {{task}}
    </performance_optimization_task>
```

#### Response Templates
```yaml
agent:
  action_observation_template: |
    <returncode>{{output.returncode}}</returncode>
    <output>{{output.output}}</output>

  format_error_template: |
    Please always provide EXACTLY ONE action in triple backticks.

  timeout_template: |
    Command timed out. Please try another command.
```
