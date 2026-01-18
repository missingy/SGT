# 实验框架说明（中文版）

本文档说明如何用统一入口和配置系统复现实验。

## 1. 快速开始

在默认数据集上跑完整流程：

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_default.yaml --resume
```

使用小数据集：

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data_100.yaml --experiment configs/experiments/exp_default.yaml --resume
```

## 2. 配置结构

配置合并顺序：

1) base.yaml  
2) dataset 配置  
3) experiment 配置  

后者覆盖前者。

顶层常用字段：

- seed, deterministic_tf  
- paths  
- dataset  
- steps  
- experiment  

### base.yaml（节选）

```
seed: 42
paths:
  artifacts_root: artifacts
  results_jsonl: results/results.jsonl
  results_csv: results/results.csv

dataset:
  id: data
  data_root: data/Data

steps:
  sgt:
    script: src/sgt_features.py
    alpha: 0.8
    max_dt_beats: 8
    token_mode: prog_pitch
    top_vocab: 64
  train:
    task: both
    n_splits: 5
```

### 数据集配置

```
dataset:
  id: data_100
  data_root: data/Data_100
```

### 实验配置

```
experiment:
  name: default

steps:
  sgt:
    script: src/sgt_features.py
  train:
    task: both
```

## 3. 步骤输出与缓存

每个步骤的输出目录：

```
artifacts/{dataset_id}/{step}/{hash}/...
```

hash 由以下内容决定：

- dataset 配置  
- step 配置  
- 依赖步骤的 hash  

使用 `--resume` 时，如果输出已存在则自动跳过。

## 4. 选择性重跑

跳过某一步：

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_default.yaml --skip embed,sgt
```

强制重跑某一步：

```
python src/runners/run_experiment.py --base configs/base.yaml --dataset configs/datasets/data.yaml --experiment configs/experiments/exp_default.yaml --force sgt
```

## 5. 结果记录

训练步骤会追加记录到：

- results/results.jsonl  
- results/results.csv  

每条记录包含：

- run_id  
- dataset_id  
- experiment 名称  
- mode, task, seed  
- accuracy/F1 的均值与方差  

## 6. 可复现注意事项

- 固定 seed  
- dataset 配置与 experiment 配置分离  
- 若改变特征参数，hash 会变化，缓存自动失效  
- embedding 等高耗时步骤建议配合 `--resume`  
