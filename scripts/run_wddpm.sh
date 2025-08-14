#!/bin/bash

# Check if all required arguments are provided
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <accelerate_config_path> <dataset_name> <wavelet_level> <mode> <model_type> [model_dir]"
    echo "  <accelerate_config_path>: Path to the accelerate config file"
    echo "  <mode>: Operation mode (train or sample)"
    echo "  <wavelet_level>: Wavelet decomposition level (integer)"
    echo "  <dataset_name>: Dataset name (CIFAR10 or CELEBAHQ or STL10)"
    echo "  <model_type>: Model architecture (UNET or UKAN)"
    echo "  [model_dir]: Directory containing the trained model (required for sampling)"
    echo "  <num_samples>: Number of samples to generate (default: 5000)"
    echo "Example: $0 accelerate_config.yaml CIFAR10 1 train UNET"
    echo "Example: $0 accelerate_config.yaml CIFAR10 1 sample UKAN /path/to/model"
    exit 1
fi

CONFIG_PATH=$1
MODE=$2
WAVELET_LEVEL=$3
DATASET=$4
MODEL_TYPE=$5
PREDICTION_TYPE=$6
MODEL_DIR=$7
NUM_SAMPLES=$8
SAMPLER=$9
NUM_INFERENCE_STEPS=${10}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-1000}  # Default to 1000 if not provided
ETA=${11:-0.0}  # Default to 0.0 if not provided

NUM_SAMPLES=${NUM_SAMPLES:-5000}  # Default to 5000 if not provided
SAMPLER=${SAMPLER:-"ddpm"}  # Default to "ddpm" if not provided
EVAL_BATCH_SIZE=512
EVAL_SAMPLES=200
if [ "$MODE" == "train" ]; then
    if [ -z "$DATASET" ] || [ -z "$WAVELET_LEVEL" ] || [ -z "$MODEL_TYPE" ] || [ -z "$PREDICTION_TYPE" ]; then
        echo "DATASET, WAVELET_LEVEL, MODEL_TYPE and PREDICTION_TYPE are required for training mode."
        exit 1
    fi
    if [ "$WAVELET_LEVEL" -lt 0 ]; then
        echo "WAVELET_LEVEL must be a non-negative integer."
        exit 1
    fi
elif [ "$MODE" == "sample" ]; then
    echo "Running in sampling mode..."
    if [ -z "$WAVELET_LEVEL" ] || [ -z "$DATASET" ] || [ -z "$MODEL_TYPE" ] || [ -z "$PREDICTION_TYPE" ] || [ -z "$MODEL_DIR" ] || [ -z "$NUM_SAMPLES" ] || [ -z "$SAMPLER" ] || [ -z "$NUM_INFERENCE_STEPS" ]; then    
        echo "MODEL_DIR, MODEL_TYPE, WAVELET_LEVEL, PREDICTION_TYPE, NUM_SAMPLES, SAMPLER , DATASET and NUM_INFERENCE_STEPS are required for sampling mode."
        exit 1
    fi
else
    echo "Invalid mode: $MODE. Use 'train' or 'sample'."
    exit 1
fi


# Get channels configuration based on dataset, wavelet level, and model type
if [ "$DATASET" == "CIFAR10" ]; then
    NUM_EPOCHS=3000
    RESOLUTION=32
    TRAIN_BATCH_SIZE=512
    SAMPLING_BATCH_SIZE=1000
    LEARNING_RATE=1e-4
    if [ "$MODEL_TYPE" == "UNET" ]; then
        if [ "$WAVELET_LEVEL" == "1" ]; then
            CHANNELS="128-256-256"
        elif [ "$WAVELET_LEVEL" == "2" ]; then
            CHANNELS="64-128-128"
        else
            echo "Unsupported wavelet level for $DATASET with $MODEL_TYPE"
            exit 1
        fi
    elif [ "$MODEL_TYPE" == "UKAN" ]; then
        if [ "$WAVELET_LEVEL" == "0" ]; then
            CHANNELS="128-256-256"  # base_channels=128, channel_multipliers=(1, 2)
        elif [ "$WAVELET_LEVEL" == "1" ]; then
            CHANNELS="128-256"  # base_channels=128, channel_multipliers=(1, 2)
        elif [ "$WAVELET_LEVEL" == "2" ]; then
            CHANNELS="128"    # base_channels=128, channel_multipliers=[1]
        else
            echo "Unsupported wavelet level for $DATASET with $MODEL_TYPE"
            exit 1
        fi
    fi
    
elif [ "$DATASET" == "CELEBAHQ" ]; then
    NUM_EPOCHS=1000
    RESOLUTION=256
    TRAIN_BATCH_SIZE=64
    SAMPLING_BATCH_SIZE=200
    LEARNING_RATE=2e-4
    EVAL_BATCH_SIZE=68
    EVAL_SAMPLES=68

    if [ "$MODEL_TYPE" == "UNET" ]; then
        if [ "$WAVELET_LEVEL" == "0" ]; then
            CHANNELS="128-256-256-512"
        elif [ "$WAVELET_LEVEL" == "1" ]; then
            CHANNELS="64-128-256-256-256"
        elif [ "$WAVELET_LEVEL" == "2" ]; then
            CHANNELS="64-128-128-256"
        elif [ "$WAVELET_LEVEL" == "3" ]; then
            CHANNELS="64-128-256"
        else
            echo "Unsupported wavelet level for $DATASET with $MODEL_TYPE"
            exit 1
        fi
    elif [ "$MODEL_TYPE" == "UKAN" ]; then
        if [ "$WAVELET_LEVEL" == "0" ]; then
            CHANNELS="64-128-256-256"
        elif [ "$WAVELET_LEVEL" == "1" ]; then
            CHANNELS="64-128-256"
        elif [ "$WAVELET_LEVEL" == "2" ]; then
            CHANNELS="128-256-256-256"
        else
            echo "Unsupported wavelet level for $DATASET with $MODEL_TYPE"
            exit 1
        fi
    fi
    
elif [ "$DATASET" == "STL10" ]; then
    NUM_EPOCHS=2500
    RESOLUTION=64
    TRAIN_BATCH_SIZE=512
    SAMPLING_BATCH_SIZE=400
    LEARNING_RATE=1e-4
    if [ "$MODE" == "train" ]; then
        NUM_INFERENCE_STEPS=100
    fi
    if [ "$MODEL_TYPE" == "UNET" ]; then
        if [ "$WAVELET_LEVEL" == "1" ]; then
            CHANNELS="128-256-256-256"
        elif [ "$WAVELET_LEVEL" == "2" ]; then
            CHANNELS="128-256-256"
        else
            echo "Unsupported wavelet level for $DATASET with $MODEL_TYPE"
            exit 1
        fi
    elif [ "$MODEL_TYPE" == "UKAN" ]; then
        if [ "$WAVELET_LEVEL" == "1" ]; then
            CHANNELS="128-256-256-256"
        elif [ "$WAVELET_LEVEL" == "2" ]; then
            CHANNELS="128-256-256"
        else
            echo "Unsupported wavelet level for $DATASET with $MODEL_TYPE"
            exit 1
        fi
    fi
    
else
    echo "Unsupported dataset: $DATASET"
    exit 1
fi

# Construct output directory
OUTPUT_DIR="${MODEL_TYPE}-${DATASET}-${CHANNELS}-${NUM_EPOCHS}epochs-lvl${WAVELET_LEVEL}-${PREDICTION_TYPE}"

# Run based on mode
if [ "$MODE" == "train" ]; then
    echo "Running training with config: $CONFIG_PATH, dataset: $DATASET, wavelet level: $WAVELET_LEVEL, model type: $MODEL_TYPE"
    echo "Output directory: $OUTPUT_DIR"
    
    accelerate launch --config_file "$CONFIG_PATH" main.py \
        --dataset "$DATASET" \
        --wavelet-levels "$WAVELET_LEVEL" \
        --output-dir "$OUTPUT_DIR" \
        --num-epochs "$NUM_EPOCHS" \
        --eval-every-epochs 5 \
        --checkpoint-every-epochs 5 \
        --num-inference-steps "$NUM_INFERENCE_STEPS" \
        --prediction-type "$PREDICTION_TYPE" \
        --train-batch-size "$TRAIN_BATCH_SIZE" \
        --model-type "$MODEL_TYPE" \
        --prediction-type "$PREDICTION_TYPE" \
        --learning-rate "$LEARNING_RATE" \
        --eval-batch-size "$EVAL_BATCH_SIZE" \
        --eval-samples "$EVAL_SAMPLES" \
    
elif [ "$MODE" == "sample" ]; then
    # Extract the model name from MODEL_DIR (part after the last slash)
    MODEL_NAME=$(echo "$MODEL_DIR" | cut -d'/' -f2)
    
    OUTPUT_DIR="generated-images/${DATASET}/${MODEL_NAME}-${SAMPLER}-${NUM_INFERENCE_STEPS}steps-eta${ETA}"
    # OUTPUT_DIR="generated-images/steps-comparison/${DATASET}/${MODEL_NAME}-${SAMPLER}-${NUM_INFERENCE_STEPS}steps-eta${ETA}"
    echo "Running sampling with config: $CONFIG_PATH, wavelet level: $WAVELET_LEVEL, model type: $MODEL_TYPE"
    echo "Output directory: $OUTPUT_DIR"
    accelerate launch --config_file "$CONFIG_PATH" wavelet_sampling.py \
        --wavelet-level "$WAVELET_LEVEL" \
        --output-dir $OUTPUT_DIR\
        --num-steps "$NUM_INFERENCE_STEPS" \
        --prediction-type "$PREDICTION_TYPE" \
        --model-dir "$MODEL_DIR" \
        --model-type "$MODEL_TYPE" \
        --resolution "$RESOLUTION" \
        --scheduler "$SAMPLER" \
        --num-samples "$NUM_SAMPLES" \
        --batch-size "$SAMPLING_BATCH_SIZE" \
        --eta "$ETA" \
        
    
else
    echo "Unsupported mode: $MODE (use 'train' or 'sample')"
    exit 1
fi