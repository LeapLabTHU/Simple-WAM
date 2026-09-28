#!/bin/bash

run_libero_eval() {
    local task_list_file=$1
    echo "task_file: $task_list_file"

    require_non_empty() {
        local var_name="$1"
        local var_val="${!var_name}"
        if [ -z "$var_val" ]; then
            echo "Error: required variable $var_name is not set"
            exit 1
        fi
    }

    ROOT_DIR=${ROOT_DIR:-"$(pwd)"}
    export ROOT_DIR
    RUN_ID=${RUN_ID:-"eval_$(date +%Y%m%d_%H%M%S)"}
    export RUN_ID
    OUTPUT_DIR=${OUTPUT_DIR:-"$ROOT_DIR/evaluate_results/$RUN_ID"}
    export OUTPUT_DIR
    EXP_NAME=${EXP_NAME:-""}
    export EXP_NAME
    PYTHON_BIN=${PYTHON_BIN:-python}

    mkdir -p "$OUTPUT_DIR"
    echo "Evaluation results will be saved to: $OUTPUT_DIR"

    cp "$task_list_file" "$OUTPUT_DIR/"
    task_list_file="$OUTPUT_DIR/$(basename "$task_list_file")"
    echo "Task list file copied to: $task_list_file"

    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        require_non_empty "NUM_GPUS"
        AVAILABLE_GPUS=$(seq 0 $((NUM_GPUS-1)) | tr '\n' ',' | sed 's/,$//')
    else
        AVAILABLE_GPUS=$CUDA_VISIBLE_DEVICES
        NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    fi
    export NUM_GPUS
    IFS=',' read -r -a GPU_ARRAY <<< "$AVAILABLE_GPUS"

    require_non_empty "NUM_TRIALS"
    require_non_empty "CKPT"
    require_non_empty "CONFIG"

    CONFIG="${CONFIG#configs/}"
    CONFIG="${CONFIG#task/}"
    CONFIG="${CONFIG%.yaml}"
    export CONFIG

    TASK_STATUS_DIR="$OUTPUT_DIR/task_status"
    TASK_LOG_DIR="$OUTPUT_DIR/task_logs"
    TASK_SHARD_DIR="$OUTPUT_DIR/task_shards"
    FAILED_TASKS_FILE="$OUTPUT_DIR/failed_tasks.txt"
    mkdir -p "$TASK_STATUS_DIR" "$TASK_LOG_DIR" "$TASK_SHARD_DIR"
    : > "$FAILED_TASKS_FILE"

    echo "CKPT: $CKPT"
    echo "CONFIG: $CONFIG"
    echo "ROOT_DIR: $ROOT_DIR"
    echo "NUM_GPUS: $NUM_GPUS, AVAILABLE_GPUS: $AVAILABLE_GPUS"
    echo "NUM_TRIALS: $NUM_TRIALS"
    echo "Persistent worker mode: one model load per GPU worker"

    for gpu in "${GPU_ARRAY[@]}"; do
        : > "$TASK_SHARD_DIR/gpu${gpu}_tasks.txt"
    done

    local task_index=0
    while IFS= read -r task_line; do
        [ -z "$task_line" ] && continue
        local worker_index=$((task_index % NUM_GPUS))
        local gpu_id="${GPU_ARRAY[$worker_index]}"
        echo "$task_line" >> "$TASK_SHARD_DIR/gpu${gpu_id}_tasks.txt"
        task_index=$((task_index + 1))
    done < "$task_list_file"

    local total_tasks
    total_tasks=$(wc -l < "$task_list_file")
    echo "Total tasks: $total_tasks"

    local pids=()
    local worker_gpus=()
    for gpu_id in "${GPU_ARRAY[@]}"; do
        local shard_file="$TASK_SHARD_DIR/gpu${gpu_id}_tasks.txt"
        local shard_tasks
        shard_tasks=$(wc -l < "$shard_file" 2>/dev/null || echo 0)
        if [ "$shard_tasks" -eq 0 ]; then
            echo "GPU$gpu_id has no assigned tasks; skip."
            continue
        fi

        local log_file="$TASK_LOG_DIR/worker_gpu${gpu_id}.log"
        local status_file="$TASK_STATUS_DIR/worker_gpu${gpu_id}.status"
        rm -f "$status_file"
        echo "Launching persistent worker on GPU$gpu_id with $shard_tasks tasks; log=$log_file"

        (
            cd "$ROOT_DIR" || exit 1
            extra_args_array=()
            if [ -n "$EXTRA_ARGS" ]; then
                eval "extra_args_array=($EXTRA_ARGS)"
            fi
            CUDA_VISIBLE_DEVICES="$gpu_id" "$PYTHON_BIN" experiments/libero/eval_libero_worker.py \
                "task=$CONFIG" \
                "ckpt=$CKPT" \
                "gpu_id=$gpu_id" \
                "MULTIRUN.task_file=$shard_file" \
                "EVALUATION.num_trials=$NUM_TRIALS" \
                "EVALUATION.output_dir=$OUTPUT_DIR" \
                "${extra_args_array[@]}"
        ) > "$log_file" 2>&1 &

        local pid=$!
        pids+=("$pid")
        worker_gpus+=("$gpu_id")
        echo "RUNNING|$gpu_id|$pid|$(date +%s)|$log_file" > "$status_file"
    done

    local failed=0
    for i in "${!pids[@]}"; do
        local pid="${pids[$i]}"
        local gpu_id="${worker_gpus[$i]}"
        local log_file="$TASK_LOG_DIR/worker_gpu${gpu_id}.log"
        local status_file="$TASK_STATUS_DIR/worker_gpu${gpu_id}.status"
        if wait "$pid"; then
            echo "SUCCESS|$gpu_id|0|$(date +%s)|$log_file" > "$status_file"
            echo "Worker GPU$gpu_id completed."
        else
            local rc=$?
            failed=1
            echo "FAILED|$gpu_id|$rc|$(date +%s)|$log_file" > "$status_file"
            echo "$(date '+%Y-%m-%d %H:%M:%S'),worker,gpu=$gpu_id,rc=$rc,log=$log_file" >> "$FAILED_TASKS_FILE"
            echo "Worker GPU$gpu_id failed with rc=$rc. See $log_file"
        fi
    done

    if [ "$failed" -ne 0 ]; then
        echo "Detected failed workers. Failure details: $FAILED_TASKS_FILE"
        cat "$FAILED_TASKS_FILE"
        return 2
    fi

    local total_completed
    total_completed=$(find "$OUTPUT_DIR" -type f -name "gpu*_task*_results.json" | wc -l)
    if [ "$total_completed" -ne "$total_tasks" ]; then
        echo "Evaluation incomplete: completed $total_completed/$total_tasks result files."
        return 2
    fi

    echo "All tasks completed successfully!"
    echo "Generating evaluation report..."
    "$PYTHON_BIN" experiments/libero/summarize_results.py --output_dir="$OUTPUT_DIR"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    if [ $# -lt 1 ]; then
        echo "Error: task file path is required"
        echo "Usage: $0 <task_file>"
        exit 1
    fi
    test_file="$1"
    run_libero_eval "$test_file"
    exit $?
fi
