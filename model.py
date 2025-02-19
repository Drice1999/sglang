from sglang.utils import (
    execute_shell_command,
    wait_for_server,
    terminate_process,
    print_highlight,
)

from sglang.srt.server import launch_server
from sglang.srt.server_args import prepare_server_args
from sglang.srt.utils import kill_process_tree

from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

if __name__ == "__main__":
    server_args = prepare_server_args(
        [
            "--model-path", "/root/autodl-tmp/sglang/model/pixtral",
            "--port", "30000", 
            "--host", "0.0.0.0",
            "--tokenizer-mode", "mistral",
            "--load-format", "mistral"
        ])

    launch_server(server_args)

    # server_process = execute_shell_command(
    #     """
    # python -m sglang.launch_server --model-path /root/autodl-tmp/sglang/model/pixtral \
    # --port 30000 --host 0.0.0.0
    # """
    # )
    