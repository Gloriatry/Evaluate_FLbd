#!/bin/bash  
  
# 定义防御方法  
attack_methods=("dba")  
  
# 定义异构方式及其参数  
defense_methods=(  
    "avg,0"
    "flame,0"  
    "multi-metrics,0"  
    "fltrust,0"  
    "fl-defender,0"  
    "foolsgold,0"
    "freqfed,1"  
    # "fldetector,0"  
    "crowdguard,1"
    "flshield,1"  
)  

  
# 遍历防御方式和异构方式  
for attack in "${attack_methods[@]}"; do  
    for defense_method in "${defense_methods[@]}"; do  
        # 解析异构方式的参数  
        IFS=',' read -r defense gpu <<< "$defense_method"  

        extra_param=""
        if [ "$defense" == "flshield" ]; then  
            extra_param="--bijective"  
        fi

        # 构建命令  
        command="python main_fed.py --dataset femnist --epochs 600 \
--lr_m 0.01 --lr_b 0.01 --local_ep_m 3 --local_ep_b 3 \
--malicious 0.3 --attack $attack --start_attack 200 --heter homo \
--gpu $gpu --defence $defense --start_defence 200 \
--init /root/project/model_bank/ --frac 0.25 $extra_param --poison_frac 0.1"   
  
        # 打印并运行命令  
        echo "Running: $command"  
        eval $command &  
    done  
done  
  
wait