#!/bin/zsh
zmodload zsh/mathfunc

pardir=$(dirname $(readlink -f "$0" ))

# 2 views
# for i in {0..127..16}; do
#     imax=$(($i+65))
#     echo "i=$i, imax=$imax"
#     python $pardir/../nerfstudio/nerfstudio/scripts/train.py nerf_xray \
#         --data $pardir/../nerf_data/synthetic/balls \
#         --vis tensorboard \
#         --max_num_iterations 1000 \
#         xray-dataparser \
#         --imin $i \
#         --imax $imax \
#         --istep 64
# done

# 3 views
# for i in {0..86..12}; do
#     imax=256
#     echo "i=$i, imax=$imax"
#     python $pardir/../nerfstudio/nerfstudio/scripts/train.py nerf_xray \
#         --data "$pardir/../nerf_data/synthetic/balls" \
#         --vis "tensorboard" \
#         --max_num_iterations 1000 \
#         xray-dataparser \
#         --imin $i \
#         --imax $imax \
#         --istep 86
# done
for nviews in {1,3,5,9,17,33,65,129}; do
    step=$((int(ceil(256./$nviews))))
    di=$((int(floor($step/8.))))
    di=$(( di<1 ? 1 : $di ))
    echo "### Step $step, nviews=$nviews ###"
    for i in {0..$((step-1))..$di}; do
        imax=256
        echo "i=$i, imax=$imax"
        python $pardir/../nerfstudio/nerfstudio/scripts/train.py nerf_xray \
            --data "$pardir/../nerf_data/synthetic/lattice" \
            --vis "tensorboard" \
            --max_num_iterations 1000 \
            xray-dataparser \
            --imin $i \
            --imax $imax \
            --istep $step
    done
done