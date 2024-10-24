#!/bin/bash

# 定义防御方法
attack_methods=("dba" "edges")

# 定义异构方式及其参数
heterogeneities=(
    "homo,,,0"
    "dirichlet,0.1,,1"
    "dirichlet,0.5,,1"
    "noniid-#label2,,,3"
    "noniid-#label3,,,3"
    "quantity,0.5,,4"
    "homo,,0.1,4"
)

# 遍历防御方式和异构方式
for attack in "${attack_methods[@]}"; do
    for heter in "${heterogeneities[@]}"; do
        # 解析异构方式的参数
        IFS=',' read -r heter_type alpha gau_noise gpu <<< "$heter"

        # 构建命令
        command="python main_fed.py --dataset cifar --epochs 600 --lr_m 0.05 \
        --lr_b 0.05 --malicious 0.3 --attack $attack --start_attack 300 \
        --heter $heter_type --gpu $gpu --defence avg --init model_bank/cifar/model_last.pt.tar.epoch_300"

        if [ -n "$alpha" ]; then
            command+=" --alpha $alpha"
        fi
        if [ -n "$gau_noise" ]; then
            command+=" --gau_noise $gau_noise"
        fi

        # 打印并运行命令
        echo "Running: $command"
        eval $command &
    done
done
