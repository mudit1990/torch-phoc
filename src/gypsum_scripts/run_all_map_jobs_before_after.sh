#!/bin/bash

# sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_dec6/images_before_after_0.log gypsum_scripts/gypsum_before_after.sh D0042-1070001

# sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_dec6/images_before_after_1.log gypsum_scripts/gypsum_before_after.sh D0042-1070002

sbatch -p m40-short --gres=gpu:1 --mem=240000 --output=logs/logs_dec6/images_before_after_2.log gypsum_scripts/gypsum_before_after.sh D0042-1070006

# sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_dec6/images_before_after_3.log gypsum_scripts/gypsum_before_after.sh D0042-1070007

sbatch -p m40-short --gres=gpu:1 --mem=240000 --output=logs/logs_dec6/images_before_after_4.log gypsum_scripts/gypsum_before_after.sh D0117-5755018

sbatch -p m40-short --gres=gpu:1 --mem=240000 --output=logs/logs_dec6/images_before_after_5.log gypsum_scripts/gypsum_before_after.sh D0117-5755035

# sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_dec6/images_before_after_6.log gypsum_scripts/gypsum_before_after.sh D0117-5755036
