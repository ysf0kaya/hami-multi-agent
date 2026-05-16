#!/bin/bash
# Tüm agent'ları sırayla çalıştırır ve veri toplar

NAMESPACE="multi-agent"
DATASET="/home/ysf/hami-dataset/dataset.csv"

# Dataset klasörü
sudo mkdir -p /home/ysf/hami-dataset
sudo chmod 777 /home/ysf/hami-dataset

# Namespace
kubectl apply -f k8s/namespace.yaml

run_agent() {
    local manifest=$1
    local name=$2

    echo ""
    echo "▶ [$name] başlatılıyor..."
    kubectl apply -f k8s/${manifest}

    # Running olana kadar bekle
    echo "  Pod başlatılıyor..."
    sleep 15

    # Tamamlanana kadar bekle (max 10 dakika)
    for i in $(seq 1 60); do
        STATUS=$(kubectl get pods -n $NAMESPACE -l app=$name \
            -o jsonpath='{.items[0].status.phase}' 2>/dev/null)

        if [ "$STATUS" = "Succeeded" ] || \
           kubectl logs -n $NAMESPACE deployment/$name --tail=3 2>/dev/null | \
           grep -qE "tamamladı|bekleme moduna"; then
            echo "  ✅ [$name] tamamlandı"
            break
        elif [ "$STATUS" = "Failed" ]; then
            echo "  ❌ [$name] hata verdi"
            kubectl logs -n $NAMESPACE deployment/$name --tail=20
            break
        fi
        echo "  ⏳ Bekleniyor... ($i/60)"
        sleep 10
    done

    # Sil
    kubectl delete -f k8s/${manifest} --ignore-not-found
    sleep 5
}

echo "=============================="
echo " HAMi Veri Toplama Başlıyor"
echo "=============================="

# NLP Agent'ları
run_agent "nlp-bert-tiny.yaml"   "nlp-bert-tiny"
run_agent "nlp-bert-base.yaml"   "nlp-bert-base"
run_agent "nlp-distilbert.yaml"  "nlp-distilbert"
run_agent "nlp-roberta.yaml"     "nlp-roberta"
run_agent "nlp-albert.yaml"      "nlp-albert"

# LLM Agent'ları
run_agent "llm-distilgpt.yaml"   "llm-distilgpt"
run_agent "llm-lamini.yaml"      "llm-lamini"
run_agent "llm-opt.yaml"         "llm-opt"

# Vision Agent'ları
run_agent "vision-resnet18.yaml"     "vision-resnet18"
run_agent "vision-resnet50.yaml"     "vision-resnet50"
run_agent "vision-mobilenet.yaml"    "vision-mobilenet"
run_agent "vision-efficientnet.yaml" "vision-efficientnet"
run_agent "vision-vit.yaml"          "vision-vit"

# Audio Agent'ları
run_agent "audio-whisper-tiny.yaml"  "audio-whisper-tiny"
run_agent "audio-whisper-base.yaml"  "audio-whisper-base"
run_agent "audio-wav2vec2.yaml"      "audio-wav2vec2"

echo ""
echo "=============================="
echo " Veri Toplama Tamamlandı!"
echo "=============================="
echo ""
echo "Dataset:"
cat $DATASET
