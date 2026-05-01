#!/bin/bash
# K8s manifest'lerini otomatik üretir

IMAGE="ysf0kaya/hami-multi-agent:latest"
NAMESPACE="multi-agent"
DATASET_PATH="/data/dataset.csv"

generate_manifest() {
  local name=$1
  local agent_script=$2
  local model_key=$3
  local category=$4
  local gpu_mem=$5

  cat <<YAML > k8s/${name}.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ${name}
  namespace: ${NAMESPACE}
  labels:
    app: ${name}
    category: ${category}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: ${name}
  template:
    metadata:
      labels:
        app: ${name}
        category: ${category}
    spec:
      schedulerName: hami-scheduler
      containers:
      - name: ${name}
        image: ${IMAGE}
        command: ["python3", "agents/${agent_script}"]
        env:
        - name: MODEL_NAME
          value: "${model_key}"
        - name: AGENT_ID
          value: "${name}"
        - name: BATCH_SIZE
          value: "32"
        - name: REPEAT
          value: "5"
        - name: DATASET_PATH
          value: "${DATASET_PATH}"
        volumeMounts:
        - name: dataset-volume
          mountPath: /data
        resources:
          limits:
            nvidia.com/gpu: 1
            nvidia.com/gpumem: ${gpu_mem}
          requests:
            memory: "2Gi"
            cpu: "500m"
      volumes:
      - name: dataset-volume
        hostPath:
          path: /tmp/hami-dataset
          type: DirectoryOrCreate
YAML
  echo "✅ k8s/${name}.yaml oluşturuldu"
}

# ── LLM Agent'ları (GPU_YOGUN) ─────────────────────────────
generate_manifest "llm-qwen"      "llm_agent.py" "qwen-0.5b"  "GPU_YOGUN" "1536"
generate_manifest "llm-tinyllama" "llm_agent.py" "tinyllama"  "GPU_YOGUN" "1536"
generate_manifest "llm-distilgpt" "llm_agent.py" "distilgpt2" "GPU_YOGUN" "512"
generate_manifest "llm-lamini"    "llm_agent.py" "lamini-gpt" "GPU_YOGUN" "512"
generate_manifest "llm-opt"       "llm_agent.py" "opt-125m"   "GPU_YOGUN" "512"

# ── NLP Agent'ları (CPU_YOGUN) ─────────────────────────────
generate_manifest "nlp-bert-base"  "nlp_agent.py" "bert-base"   "CPU_YOGUN" "512"
generate_manifest "nlp-distilbert" "nlp_agent.py" "distilbert"  "CPU_YOGUN" "512"
generate_manifest "nlp-roberta"    "nlp_agent.py" "roberta"     "CPU_YOGUN" "512"
generate_manifest "nlp-bert-tiny"  "nlp_agent.py" "bert-tiny"   "CPU_YOGUN" "256"
generate_manifest "nlp-albert"     "nlp_agent.py" "albert-base" "CPU_YOGUN" "512"

# ── Vision Agent'ları (IO_YOGUN) ───────────────────────────
generate_manifest "vision-vit"         "vision_agent.py" "vit-base"     "IO_YOGUN" "512"
generate_manifest "vision-resnet18"    "vision_agent.py" "resnet-18"    "IO_YOGUN" "256"
generate_manifest "vision-resnet50"    "vision_agent.py" "resnet-50"    "IO_YOGUN" "512"
generate_manifest "vision-mobilenet"   "vision_agent.py" "mobilenet"    "IO_YOGUN" "256"
generate_manifest "vision-efficientnet" "vision_agent.py" "efficientnet" "IO_YOGUN" "256"

# ── Audio Agent'ları (KARMA) ────────────────────────────────
generate_manifest "audio-whisper-tiny" "audio_agent.py" "whisper-tiny"  "KARMA" "512"
generate_manifest "audio-whisper-base" "audio_agent.py" "whisper-base"  "KARMA" "512"
generate_manifest "audio-wav2vec2"     "audio_agent.py" "wav2vec2-base" "KARMA" "512"

echo ""
echo "Toplam manifest: $(ls k8s/*.yaml | wc -l)"
