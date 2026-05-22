# Arabic AI-Generated Text Detection, Big Data Pipeline

## Where Spark, Hadoop, and MapReduce are used

Here is a clear map of where each Big Data technology is used:

### Apache Spark

| File | How Spark is used |
|------|-------------------|
| `data_preparation.py` | Spark DataFrames + UDFs to clean Arabic text in parallel |
| `mapreduce_jobs.py` | Spark RDDs with `flatMap` -> `reduceByKey` (MapReduce model) |
| `feature_engineering.py` | Spark UDFs for stylometric features + Spark MLlib for TF-IDF |
| `modeling.py` | Spark MLlib for distributed training of 3 classifiers |
| `streaming_pipeline.py` | Spark Structured Streaming for real-time scoring |
| `scalability_benchmark.py` | Measures Spark performance across different partition counts |

### Hadoop / HDFS

| Where | How it's used |
|-------|---------------|
| Input data | Can be read from HDFS via `hdfs:///user/marwah/...` path |
| Output data | Parquet files can be written directly to HDFS |
| Execution | `spark-submit --master yarn` runs Spark on a Hadoop YARN cluster |
| MapReduce I/O | The MapReduce outputs are saved as Hadoop-style text part-files |

### MapReduce (the model)

The MapReduce paradigm is implemented explicitly in `mapreduce_jobs.py` with **3 jobs**:

| Job | Map phase | Reduce phase |
|-----|-----------|--------------|
| **Job 1: Word Count** | `(text) → [(word, 1), ...]` | `sum counts per word` |
| **Job 2: Bigram Count** | `(text) → [((w₁, w₂), 1), ...]` | `sum counts per bigram` |
| **Job 3: Hapax Ratio (2-stage)** | Stage A = Job 1; Stage B re-keys to "hapax" or "total" | `sum per category` |

Each job prints `[Map]`, `[Shuffle]`, `[Reduce]` so you can see the phases happen step by step.

---

## Project structure

```
arabic-ai-text-detection/
├── data/
│   ├── raw/                    Original parquet from Hugging Face
│   └── processed/              Cleaned + balanced + features parquets
├── models/                     Saved Spark MLlib models
├── notebooks/                  Jupyter notebooks for EDA
├── reports/figures/            All plots used in the report
├── src/                        Python source code
│   ├── utils.py                Helpers (Spark session, Arabic NLP)
│   ├── data_acquisition.py     Phase 1: download from Hugging Face
│   ├── data_preparation.py     Phase 2: Spark UDFs for cleaning
│   ├── mapreduce_jobs.py       Phase 2b: MapReduce jobs (PySpark RDDs)
│   ├── feature_engineering.py  Phase 3a: Spark UDFs + MLlib TF-IDF
│   ├── modeling.py             Phase 3b: Spark MLlib classifiers
│   ├── evaluation.py           Phase 4: confusion matrices, ROC curves
│   ├── streaming_pipeline.py   Phase 5: Spark Structured Streaming
│   └── scalability_benchmark.py Phase 6: parallelism sweep
├── main.py                     Run any phase from Python
├── run_all.sh                  Run all phases from bash
└── requirements.txt
```

---

## How to run

### One-time setup (Linux)

```bash
sudo apt install -y openjdk-17-jre-headless python3-pip
pip3 install --break-system-packages -r requirements.txt
```

### Run the full pipeline

```bash
chmod +x run_all.sh
./run_all.sh
```

### Run just one phase

```bash
./run_all.sh mapreduce        # only MapReduce
./run_all.sh model            # only Spark MLlib training
./run_all.sh stream           # only streaming
```

### Run on Hadoop YARN 
```bash
# 1. Upload data to HDFS
hdfs dfs -mkdir -p /user/marwah/arabic_aigt/processed
hdfs dfs -put data/processed/processed_abstracts.parquet \
              /user/marwah/arabic_aigt/processed/

# 2. Run MapReduce jobs on Hadoop
export MR_INPUT_PATH=hdfs:///user/marwah/arabic_aigt/processed/processed_abstracts.parquet
export MR_OUTPUT_DIR=hdfs:///user/marwah/arabic_aigt/mr_output
spark-submit --master yarn --deploy-mode client src/mapreduce_jobs.py
```


