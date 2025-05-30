# SPDX-License-Identifier: Apache-2.0
import argparse
import asyncio

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams


async def main(dp_start_rank: int):
    engine_args = AsyncEngineArgs(
        model="ibm-research/PowerMoE-3b",
        tensor_parallel_size=2,
        data_parallel_size=2,
        dtype="auto",
        max_model_len=2048,
        data_parallel_address="127.0.0.1",
        data_parallel_rpc_port=29550,
        enforce_eager=True,
    )
    vllm_config = AsyncEngineArgs.create_engine_config(engine_args)
    vllm_config.parallel_config.data_parallel_size_local = 1
    vllm_config.parallel_config.data_parallel_rank = dp_start_rank
    engine_client = AsyncLLMEngine.from_vllm_config(vllm_config)

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=100,
    )

    prompt = "Who won the 2004 World Series?"
    async for output in engine_client.generate(
        prompt=prompt,
        sampling_params=sampling_params,
        request_id="abcdef",
    ):
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp-start-rank", type=int, required=True)
    args = parser.parse_args()

    asyncio.run(main(args.dp_start_rank))
