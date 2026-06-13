#!/usr/bin/env bash

# 串行执行多个不同的命令并记录日志
# 支持：
#   1) 从命令文件读取 (-f file)，忽略空行与以#开头的注释行
#   2) 直接在脚本后面追加多个命令参数（需用引号包裹含空格的命令）
#   3) 同时提供时，先执行文件中的，再执行参数中的
# 使用示例：
#   ./run_experiment.sh -o run.log -f commands.txt
#   ./run_experiment.sh -o run.log "python a.py" "python b.py --flag" "echo done"
#   ./run_experiment.sh "python x.py" "python y.py"    # 输出日志默认 run_commands.log

set -u -o pipefail

OUTPUT_FILE="run_commands.log"
COMMAND_FILE=""
N_JOBS=1  # 并行度，默认串行
GPU_LIST=""  # 逗号分隔 GPU id 列表，例如 0,1,2

print_usage() {
	cat <<'EOF'
用法: ./run_experiment.sh [选项] [命令1] [命令2] ...
选项:
	-o <file>   指定日志输出文件 (默认: run_commands.log)
	-f <file>   指定包含多行命令的文件
	-j <n>      同时并行的最大任务数 (默认:1)
	-g <gpus>   指定可用 GPU 列表 (逗号分隔, 例如: 0,1,2)。
	            分配策略: 轮询。命令中若包含占位符 {gpu} 则替换；
	            否则若未显式包含 --cuda / --device / CUDA_VISIBLE_DEVICES，则自动在子进程前添加 CUDA_VISIBLE_DEVICES 设置。
	-h          显示帮助
说明:
	既可以通过 -f 提供命令文件，也可以直接把命令作为参数写在脚本后面。
	两种方式并存时，先执行文件里的，再执行参数中的。
日志内容:
	- 每条命令开始/结束时间
	- 退出状态码
	- 运行耗时 (秒)
	- 汇总成功/失败数量
EOF
}

while getopts ":o:f:j:g:h" opt; do
	case "$opt" in
		o) OUTPUT_FILE="$OPTARG" ;;
		f) COMMAND_FILE="$OPTARG" ;;
		j) N_JOBS="$OPTARG" ;;
		g) GPU_LIST="$OPTARG" ;;
		h) print_usage; exit 0 ;;
		:) echo "选项 -$OPTARG 需要参数" >&2; exit 1 ;;
		\?) echo "未知选项: -$OPTARG" >&2; print_usage; exit 1 ;;
	esac
done
shift $((OPTIND-1))

declare -a COMMANDS

# 读取文件中的命令
if [[ -n "$COMMAND_FILE" ]]; then
	if [[ ! -f "$COMMAND_FILE" ]]; then
		echo "命令文件不存在: $COMMAND_FILE" >&2
		exit 1
	fi
	while IFS=$'\n' read -r line; do
		# 去掉前后空白
		cmd="${line}";
		# 跳过空行与注释
		if [[ -z "${cmd//[[:space:]]/}" ]] || [[ "$cmd" =~ ^[[:space:]]*# ]]; then
			continue
		fi
		COMMANDS+=("$cmd")
	done < "$COMMAND_FILE"
fi

# 追加参数中的命令
if [[ $# -gt 0 ]]; then
	for arg_cmd in "$@"; do
		COMMANDS+=("$arg_cmd")
	done
fi

if [[ ${#COMMANDS[@]} -eq 0 ]]; then
	echo "未提供任何要执行的命令。使用 -h 查看帮助。" >&2
	exit 1
fi

# 校验并行度
if ! [[ "$N_JOBS" =~ ^[0-9]+$ ]]; then
    echo "-j 应为正整数, 当前: $N_JOBS" >&2; exit 1
fi
if (( N_JOBS < 1 )); then
    N_JOBS=1
fi

# 处理 GPU 列表
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
GPU_COUNT=0
if [[ -n "$GPU_LIST" ]]; then
	# 过滤空白项
	TMP_GPU_ARRAY=()
	for g in "${GPU_ARRAY[@]}"; do
		g_trim="${g//[[:space:]]/}"
		[[ -z "$g_trim" ]] && continue
		if ! [[ "$g_trim" =~ ^[0-9]+$ ]]; then
				echo "GPU 列表包含非法ID: $g_trim" >&2; exit 1
		fi
		TMP_GPU_ARRAY+=("$g_trim")
	done
	GPU_ARRAY=("${TMP_GPU_ARRAY[@]}")
	GPU_COUNT=${#GPU_ARRAY[@]}
	if (( GPU_COUNT == 0 )); then
		echo "解析后的 GPU 列表为空" >&2; exit 1
	fi
fi

START_ALL=$(date +%s)
{
	echo "Start time: $(date)"
	echo "Output file: $OUTPUT_FILE"
	echo "Total commands: ${#COMMANDS[@]}"
	echo "n_jobs: $N_JOBS"
	echo "gpu_list: ${GPU_LIST:-<none>}"
	echo "======================================"
} > "$OUTPUT_FILE"

declare -a CMD_LOG_DIRS CMD_LOG_FILES CMD_EXIT_FILES
total=${#COMMANDS[@]}
idx=0
running_batch=0

for cmd in "${COMMANDS[@]}"; do
	idx=$((idx+1))
	tmpdir=$(mktemp -d -t runexp_XXXXXXXX)
	CMD_LOG_DIRS[idx]="$tmpdir"
	log_file="$tmpdir/cmd.log"
	exit_file="$tmpdir/exit_code"
	CMD_LOG_FILES[idx]="$log_file"
	CMD_EXIT_FILES[idx]="$exit_file"

	assigned_gpu=""
	wrapped_cmd="$cmd"
	if (( GPU_COUNT > 0 )); then
		gpu_index=$(( (idx - 1) % GPU_COUNT ))
		assigned_gpu="${GPU_ARRAY[$gpu_index]}"
		# 占位符替换 {gpu}
		if [[ "$wrapped_cmd" == *"{gpu}"* ]]; then
			wrapped_cmd="${wrapped_cmd//\{gpu\}/$assigned_gpu}"
		else
			# 如果命令里没有 --cuda / --device / CUDA_VISIBLE_DEVICES，则加环境变量限制
			if ! [[ "$wrapped_cmd" =~ --cuda[=[:space:]] || "$wrapped_cmd" =~ --device || "$wrapped_cmd" =~ CUDA_VISIBLE_DEVICES ]]; then
				wrapped_cmd="CUDA_VISIBLE_DEVICES=$assigned_gpu $wrapped_cmd"
			fi
		fi
	fi

	(
		start_ts=$(date +%s)
		{
			echo ""
			echo "======== Command $idx / $total ========"
			echo "Command: $cmd"
			if [[ -n "$assigned_gpu" ]]; then
				echo "Assigned GPU: $assigned_gpu"
			fi
			echo "Start: $(date)"
			echo "--- STDOUT/STDERR BEGIN ---"
			eval "$wrapped_cmd"
			ec=$?
			echo "--- STDOUT/STDERR END (exit=$ec) ---"
			end_ts=$(date +%s)
			duration=$(( end_ts - start_ts ))
			if [[ $ec -eq 0 ]]; then
				status="SUCCESS"
			else
				status="FAIL (exit=$ec)"
			fi
			echo "Finish: $(date)"
			echo "Status: $status"
			echo "Duration: ${duration}s"
			echo "======== End of Command $idx ========"
		} > "$log_file" 2>&1
		echo "$ec" > "$exit_file"
	) &

	running_batch=$((running_batch+1))
	# 若达到并行上限，等待当前批次所有完成
	if (( running_batch >= N_JOBS )); then
		wait
		running_batch=0
	fi
done

# 等待剩余的
wait || true

# 汇总到主日志文件（按原始顺序）
success_cnt=0
fail_cnt=0
for i in $(seq 1 $total); do
	cat "${CMD_LOG_FILES[$i]}" >> "$OUTPUT_FILE"
	if [[ -f "${CMD_EXIT_FILES[$i]}" ]]; then
		ec=$(<"${CMD_EXIT_FILES[$i]}")
		if [[ "$ec" == "0" ]]; then
			success_cnt=$((success_cnt+1))
		else
			fail_cnt=$((fail_cnt+1))
		fi
	else
		fail_cnt=$((fail_cnt+1))
	fi
done

TOTAL_SECS=$(( $(date +%s) - START_ALL ))
{
	echo ""
	echo "======================================"
	echo "All commands finished: $(date)" 
	echo "Success: $success_cnt  Fail: $fail_cnt  Total: $((success_cnt+fail_cnt))"
	echo "Total elapsed: ${TOTAL_SECS}s"
	echo "(Parallel execution with n_jobs=$N_JOBS)"
	echo "Log saved to $OUTPUT_FILE"
} >> "$OUTPUT_FILE"

# 可选：清理临时目录（若需保留中间日志可注释）
for i in $(seq 1 $total); do
	[[ -d "${CMD_LOG_DIRS[$i]}" ]] && rm -rf "${CMD_LOG_DIRS[$i]}"
done