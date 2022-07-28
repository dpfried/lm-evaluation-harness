#!/bin/bash
python -u main.py \
	--model recoder_tagger \
	--model_args pretrained=/private/home/dpf/projects/iterative-refinement/expts/taggers/layers_finetune_pl-2_eol--1_lr-5e-5_sf-500/checkpoint-17000 \
	--device 0 \
	--batch_size 1 \
	--tasks recoder_tagger \
	--limit 10000 \
	--no_cache \
| tee expts/finetune-17K_recoder-10K.out

