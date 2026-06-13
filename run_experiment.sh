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

print_usage() {
	cat <<'EOF'
用法: ./run_experiment.sh [选项] [命令1] [命令2] ...
选项:
	-o <file>   指定日志输出文件 (默认: run_commands.log)
	-f <file>   指定包含多行命令的文件
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

while getopts ":o:f:h" opt; do
	case "$opt" in
		o) OUTPUT_FILE="$OPTARG" ;;
		f) COMMAND_FILE="$OPTARG" ;;
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
	while IFS= read -r line || [[ -n "$line" ]]; do
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

START_ALL=$(date +%s)
{
	echo "Start time: $(date)"
	echo "Output file: $OUTPUT_FILE"
	echo "Total commands: ${#COMMANDS[@]}"
	echo "======================================"
} > "$OUTPUT_FILE"

success_cnt=0
fail_cnt=0
idx=0

for cmd in "${COMMANDS[@]}"; do
	idx=$((idx+1))
	echo "" >> "$OUTPUT_FILE"
	echo "======== Command $idx / ${#COMMANDS[@]} ========" >> "$OUTPUT_FILE"
	echo "Command: $cmd" >> "$OUTPUT_FILE"
	echo "Start: $(date)" >> "$OUTPUT_FILE"
	start_ts=$(date +%s)

	# 运行命令
	{
		echo "--- STDOUT/STDERR BEGIN ---"
		eval "$cmd"
		exit_code=$?
		echo "--- STDOUT/STDERR END (exit=$exit_code) ---"
	} >> "$OUTPUT_FILE" 2>&1

	end_ts=$(date +%s)
	duration=$(( end_ts - start_ts ))
	if [[ ${exit_code:-1} -eq 0 ]]; then
		success_cnt=$((success_cnt+1))
		status="SUCCESS"
	else
		fail_cnt=$((fail_cnt+1))
		status="FAIL (exit=$exit_code)"
	fi
	echo "Finish: $(date)" >> "$OUTPUT_FILE"
	echo "Status: $status" >> "$OUTPUT_FILE"
	echo "Duration: ${duration}s" >> "$OUTPUT_FILE"
	echo "======== End of Command $idx ========" >> "$OUTPUT_FILE"
done

TOTAL_SECS=$(( $(date +%s) - START_ALL ))
echo "" >> "$OUTPUT_FILE"
echo "======================================" >> "$OUTPUT_FILE"
echo "All commands finished: $(date)" >> "$OUTPUT_FILE"
echo "Success: $success_cnt  Fail: $fail_cnt  Total: $((success_cnt+fail_cnt))" >> "$OUTPUT_FILE"
echo "Total elapsed: ${TOTAL_SECS}s" >> "$OUTPUT_FILE"
echo "Log saved to $OUTPUT_FILE"