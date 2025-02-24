(
    CUDA_VISIBLE_DEVICES=1 nohup python3 generator/gen.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --input_dir open-thoughts/OpenThoughts-114k --split train --world_size 1 --chat --seed 0 --temp 0.7 --top_p 1.0 --top_k -1 --max_new_tokens 8192 --output_prefix qwen2.5_openthoughts > gen0.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=1 nohup python3 generator/gen.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --input_dir open-thoughts/OpenThoughts-114k --split train --world_size 1 --chat --seed 3 --temp 0.7 --top_p 1.0 --top_k -1 --max_new_tokens 8192 --output_prefix qwen2.5_openthoughts > gen3.log 2>&1
) &
(
    CUDA_VISIBLE_DEVICES=2 nohup python3 generator/gen.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --input_dir open-thoughts/OpenThoughts-114k --split train --world_size 1 --chat --seed 1 --temp 0.7 --top_p 1.0 --top_k -1 --max_new_tokens 8192 --output_prefix qwen2.5_openthoughts > gen1.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=2 nohup python3 generator/gen.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --input_dir open-thoughts/OpenThoughts-114k --split train --world_size 1 --chat --seed 4 --temp 0.7 --top_p 1.0 --top_k -1 --max_new_tokens 8192 --output_prefix qwen2.5_openthoughts > gen4.log 2>&1
) &
(
    CUDA_VISIBLE_DEVICES=3 nohup python3 generator/gen.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --input_dir open-thoughts/OpenThoughts-114k --split train --world_size 1 --chat --seed 2 --temp 0.7 --top_p 1.0 --top_k -1 --max_new_tokens 8192 --output_prefix qwen2.5_openthoughts > gen2.log 2>&1 ;
    CUDA_VISIBLE_DEVICES=3 nohup python3 generator/gen.py --model Qwen/Qwen2.5-7B-Instruct --revision a09a35458c702b33eeacc393d103063234e8bc28 --input_dir open-thoughts/OpenThoughts-114k --split train --world_size 1 --chat --seed 5 --temp 0.7 --top_p 1.0 --top_k -1 --max_new_tokens 8192 --output_prefix qwen2.5_openthoughts > gen5.log 2>&1
) &