#!/bin/bash

# Run this to start 7 jobs in parallel. Each will create regions and labels for the given map name.

sbatch -p 1080ti-short --gres=gpu:1 --mem=64000 --output=images_to_extend_0.log run_generate_ims_lab.sh D0042-1070001

sbatch -p 1080ti-short --gres=gpu:1 --mem=64000 --output=images_to_extend_1.log run_generate_ims_lab.sh D0042-1070002

sbatch -p 1080ti-short --gres=gpu:1 --mem=64000 --output=images_to_extend_2.log run_generate_ims_lab.sh D0042-1070006

sbatch -p 1080ti-short --gres=gpu:1 --mem=64000 --output=images_to_extend_3.log run_generate_ims_lab.sh D0042-1070007

sbatch -p 1080ti-short --gres=gpu:1 --mem=64000 --output=images_to_extend_4.log run_generate_ims_lab.sh D0117-5755018

sbatch -p 1080ti-short --gres=gpu:1 --mem=64000 --output=images_to_extend_5.log run_generate_ims_lab.sh D0117-5755035

sbatch -p 1080ti-short --gres=gpu:1 --mem=64000 --output=images_to_extend_6.log run_generate_ims_lab.sh D0117-5755036


