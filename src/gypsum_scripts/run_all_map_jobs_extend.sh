#!/bin/bash

#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_0.log gypsum_scripts/gypsum.sh D0042-1070001
#
#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_1.log gypsum_scripts/gypsum.sh D0042-1070002
#
#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_2.log gypsum_scripts/gypsum.sh D0042-1070006

sbatch -p 1080ti-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_3.log gypsum_scripts/gypsum.sh D0042-1070007

#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_4.log gypsum_scripts/gypsum.sh D0117-5755018
#
#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_5.log gypsum_scripts/gypsum.sh D0117-5755035
#
#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_6.log gypsum_scripts/gypsum.sh D0117-5755036

#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_7.log gypsum_scripts/gypsum.sh D0117-5755018_1
#
#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_8.log gypsum_scripts/gypsum.sh D0117-5755018_2
#
#sbatch -p m40-short --gres=gpu:1 --mem=100000 --output=logs/logs_2var/images_extend_9.log gypsum_scripts/gypsum.sh D0117-5755018_3


