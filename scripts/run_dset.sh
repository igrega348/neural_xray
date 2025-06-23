#!/bin/bash

pardir="/teamspace/studios/this_studio/neural_xray/scripts"   # it is a variable name defined the parent directory
# run with run_dset.sh dset mode batch_size suf suf2 n0 n1 steps lrpw wus timedelta numsteps
echo "Running with args: $@"     # Output a text to the terminal 

dset=$1
mode=$2
batch_size=$3
suf=$4
suf2=$5
n0=$6
n1=$7
steps=$8
lrpw=$9
wus=${10}
timedelta=${11}
numsteps=${12}
echo "mode=$mode, batch_size=$batch_size, suf=$suf, suf2=$suf2, n0=$n0, n1=$n1, steps=$steps, lrpw=$lrpw, wus=$wus, timedelta=$timedelta, numsteps=$numsteps"
 
dsetpath="/teamspace/studios/this_studio/neural_xray/data/experimental/$dset"
data0=$(find $dsetpath -mindepth 1 -maxdepth 1 -regex '.*/transforms_[0]+\.json')
data1=$(find $dsetpath -mindepth 1 -maxdepth 1 -regex '.*/transforms_[1-9][0-9]*\.json' | sort -V | tail -n 1)
dataall=$(find $dsetpath -mindepth 1 -maxdepth 1 -regex '.*/transforms_[0]+_to_[0-9]+\.json')
grid0=$(find $dsetpath -mindepth 1 -maxdepth 1 -name '*.npz' | sort -V | head -n 1)
grid1=$(find $dsetpath -mindepth 1 -maxdepth 1 -name '*.npz' | sort -V | tail -n 1)
# print these paths to the terminal
echo "###########################################################"
echo "data0: $data0"
echo "data1: $data1"
echo "dataall: $dataall"
echo "grid0: $grid0"
echo "grid1: $grid1"
echo "###########################################################"

outdir="/teamspace/studios/this_studio/neural_xray/outputs"

weight_nn_width_0=20
weight_nn_width_1=20
downscale_factor=4

padsteps=$(printf '%09d' $steps)
eval_batch_size=$((batch_size / 2))


if [ $mode = "canonical" ]; then
	echo "Training canonical volume forward"
	python $pardir/../nerfstudio/nerfstudio/scripts/train.py nerf_xray \
		--data $data0 \
		--output_dir $outdir \
		--logging.local-writer.max-log-size 10 \
		--pipeline.volumetric_supervision True \
		--pipeline.volumetric_supervision_coefficient 1e-4 \
		--pipeline.datamanager.volume_grid_file $grid0 \
		--pipeline.datamanager.train_num_rays_per_batch $batch_size \
		--pipeline.datamanager.eval_num_rays_per_batch $eval_batch_size \
		--pipeline.model.eval_num_rays_per_chunk $eval_batch_size \
		--pipeline.flat_field_penalty 0.005 \
		--pipeline.model.flat_field_trainable True \
		--max-num-iterations $(($numsteps + 1)) \
		--optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
		--optimizers.fields.scheduler.lr_final 1e-4 \
		--optimizers.fields.scheduler.warmup_steps 50 \
		--optimizers.fields.scheduler.steady_steps 2000 \
		--optimizers.fields.scheduler.max_steps $numsteps \
		--optimizers.flat_field.scheduler.lr_pre_warmup 1e-8 \
		--optimizers.flat_field.scheduler.lr_final 1e-4 \
		--optimizers.flat_field.scheduler.warmup_steps 200 \
		--optimizers.flat_field.scheduler.steady_steps 2000 \
		--optimizers.flat_field.scheduler.max_steps $numsteps \
		--timestamp "canonical_F$suf" \
		multi-camera-dataparser --downscale-factors.val $downscale_factor --downscale-factors.test $downscale_factor || exit 1
	
	echo "Training canonical volume backward"
	python $pardir/../nerfstudio/nerfstudio/scripts/train.py nerf_xray \
		--data $data1 \
		--output_dir $outdir \
		--logging.local-writer.max-log-size 10 \
		--pipeline.volumetric_supervision True \
		--pipeline.volumetric_supervision_coefficient 1e-4 \
		--pipeline.datamanager.volume_grid_file $grid1 \
		--pipeline.datamanager.train_num_rays_per_batch $batch_size \
		--pipeline.datamanager.eval_num_rays_per_batch $eval_batch_size \
		--pipeline.model.eval_num_rays_per_chunk $eval_batch_size \
		--pipeline.flat_field_penalty 0.005 \
		--pipeline.model.flat_field_trainable True \
		--max-num-iterations $(($numsteps + 1)) \
		--optimizers.fields.scheduler.lr_pre_warmup 1e-8 \
		--optimizers.fields.scheduler.lr_final 1e-4 \
		--optimizers.fields.scheduler.warmup_steps 50 \
		--optimizers.fields.scheduler.steady_steps 2000 \
		--optimizers.fields.scheduler.max_steps $numsteps \
		--optimizers.flat_field.scheduler.lr_pre_warmup 1e-8 \
		--optimizers.flat_field.scheduler.lr_final 1e-4 \
		--optimizers.flat_field.scheduler.warmup_steps 200 \
		--optimizers.flat_field.scheduler.steady_steps 2000 \
		--optimizers.flat_field.scheduler.max_steps $numsteps \
		--timestamp "canonical_B$suf" \
		multi-camera-dataparser --downscale-factors.val $downscale_factor --downscale-factors.test $downscale_factor || exit 1
    
elif [ $mode = "vfield" ]; then
	if [ ! -f $dataall ]; then
		echo "dataall not found"
		exit 1
	fi
	echo "Training velocity field hybrid. Output to vel_${n1}${suf2}"
	load_optimizer=False
	if [ $n1 -lt 17 ]; then
		bspline_method='matrix'
	else         
		bspline_method='neighborhood'
	fi

	if [ ! -f $outdir/$dset/xray_vfield/vel_${n0}${suf2}/nerfstudio_models/step-$padsteps.ckpt ]; then
	
		mkdir -p $outdir/$dset/xray_vfield/vel_${n1}${suf2}/nerfstudio_models
		
		python $pardir/../nerfstudio-xray/nerf-xray/nerf_xray/combine_forward_backward_checkpoints.py \
			--fwd_ckpt $outdir/$dset/nerf_xray/canonical_F${suf}/nerfstudio_models/step-$padsteps.ckpt \
			--bwd_ckpt $outdir/$dset/nerf_xray/canonical_B${suf}/nerfstudio_models/step-$padsteps.ckpt \
			--out_fn $outdir/$dset/xray_vfield/vel_${n1}${suf2}/nerfstudio_models/step-$padsteps.ckpt || exit 1
	elif [ $n1 -eq $n0 ]; then
		load_optimizer=True
	fi
		
	if [ ! $n1 -eq $n0 ]; then
		mkdir -p $outdir/$dset/xray_vfield/vel_${n1}${suf2}/nerfstudio_models
		python $pardir/../nerfstudio-xray/nerf-xray/nerf_xray/refine_vfield.py \
			--load-config $outdir/$dset/xray_vfield/vel_${n0}${suf}/config.yml \
			--new-resolution $n1 \
			--new-nn-width $weight_nn_width_1 \
			--out-path $outdir/$dset/xray_vfield/vel_${n1}${suf2}/nerfstudio_models/step-$padsteps.ckpt || exit 1
	fi

	python $pardir/../nerfstudio/nerfstudio/scripts/train.py xray_vfield \
		--data $dataall \
		--output_dir $outdir \
		--max-num-iterations $numsteps \
		--steps_per_eval_image 500 \
		--steps_per_save 250 \
		--logging.local-writer.max-log-size 10 \
		--load-checkpoint $outdir/$dset/xray_vfield/vel_${n1}${suf2}/nerfstudio_models/step-$padsteps.ckpt \
		--load-optimizer $load_optimizer \
		--pipeline.volumetric_supervision True \
		--pipeline.volumetric_supervision_coefficient 1e-4 \
		--pipeline.volumetric_supervision_start_step 4000 \
		--pipeline.datamanager.init_volume_grid_file $grid0 \
		--pipeline.datamanager.final_volume_grid_file $grid1 \
		--pipeline.model.deformation_field.num_control_points $n1 $n1 $n1 \
		--pipeline.model.deformation_field.weight_nn_width $weight_nn_width_1 \
		--pipeline.model.deformation_field.weight_nn_bias True \
		--pipeline.model.deformation_field.weight_nn_gain 1.0 \
		--pipeline.model.deformation_field.timedelta $timedelta \
		--pipeline.model.deformation_field.displacement_method $bspline_method \
		--pipeline.model.flat_field_trainable True \
		--pipeline.model.train_field_weighing False \
		--pipeline.datamanager.train_num_rays_per_batch $batch_size \
		--pipeline.datamanager.eval_num_rays_per_batch $eval_batch_size \
		--pipeline.model.eval_num_rays_per_chunk $eval_batch_size \
		--pipeline.model.distortion_loss_mult 0.0 \
		--pipeline.model.interlevel_loss_mult 0.0 \
		--pipeline.model.disable_mixing True \
		--pipeline.density_mismatch_start_step -1 \
		--pipeline.density_mismatch_coefficient 1e-3 \
		--pipeline.flat_field_loss_multiplier 0.0 \
		--optimizers.fields.optimizer.lr 1e-4 \
		--optimizers.fields.optimizer.weight_decay 1e-1 \
		--optimizers.fields.scheduler.lr_pre_warmup $lrpw \
		--optimizers.fields.scheduler.lr_final 1e-6 \
		--optimizers.fields.scheduler.warmup_steps $wus \
		--optimizers.fields.scheduler.steady_steps $(($numsteps - 1000)) \
		--optimizers.fields.scheduler.max_steps $numsteps \
		--optimizers.flat_field.optimizer.lr 1e-5 \
		--timestamp "vel_${n1}${suf2}" \
		--machine.seed 40 \
		multi-camera-dataparser --downscale-factors.val $downscale_factor --downscale-factors.test $downscale_factor || exit 1

elif [ $mode = "spatiotemporal_mix" ]; then
	echo "Training spatiotemporal mixing. Output to vel_${n0}${suf2}"

	if [ $n0 -lt 17 ]; then
		bspline_method0='matrix'
	else         
		bspline_method0='neighborhood'
	fi

	if [ $n1 -lt 17 ]; then
		bspline_method1='matrix'
	else         
		bspline_method1='neighborhood'
	fi
	
	mkdir -p $outdir/$dset/spatiotemporal_mix/vel_${n0}${suf2}/nerfstudio_models
	cp $outdir/$dset/xray_vfield/vel_${n0}${suf}/nerfstudio_models/step-$padsteps.ckpt $outdir/$dset/spatiotemporal_mix/vel_${n0}${suf2}/nerfstudio_models/step-$padsteps.ckpt	

	# python $pardir/../nerfstudio-xray/nerf-xray/nerf_xray/pretrain_mixing.py \
	# 		--load-config $outdir/$dset/spatiotemporal_mix/vel_${n0}${suf2}/config.yml \
	# 		--out-path $outdir/$dset/spatiotemporal_mix/vel_${n0}${suf2}/nerfstudio_models/step-$padsteps.ckpt || exit 1


	python $pardir/../nerfstudio/nerfstudio/scripts/train.py spatiotemporal_mix \
		--data $dataall \
		--output_dir $outdir \
		--max-num-iterations $numsteps \
		--steps_per_eval_image 500 \
		--steps_per_save 250 \
		--logging.local-writer.max-log-size 10 \
		--load-checkpoint $outdir/$dset/spatiotemporal_mix/vel_${n0}${suf2}/nerfstudio_models/step-$padsteps.ckpt \
		--load-optimizer False \
		--pipeline.volumetric_supervision False \
		--pipeline.datamanager.init_volume_grid_file $grid0 \
		--pipeline.datamanager.final_volume_grid_file $grid1 \
		--pipeline.model.field_weighing.num_control_points $n1 $n1 $n1 \
		--pipeline.model.field_weighing.displacement_method $bspline_method1 \
		--pipeline.model.deformation_field.num_control_points $n0 $n0 $n0 \
		--pipeline.model.deformation_field.weight_nn_width $weight_nn_width_1 \
		--pipeline.model.deformation_field.weight_nn_bias True \
		--pipeline.model.deformation_field.weight_nn_gain 1.0 \
		--pipeline.model.deformation_field.timedelta $timedelta \
		--pipeline.model.deformation_field.displacement_method $bspline_method0 \
		--pipeline.model.flat_field_trainable False \
		--pipeline.model.train_field_weighing True \
		--pipeline.datamanager.train_num_rays_per_batch $batch_size \
		--pipeline.datamanager.eval_num_rays_per_batch $eval_batch_size \
		--pipeline.model.eval_num_rays_per_chunk $eval_batch_size \
		--pipeline.model.distortion_loss_mult 0.0 \
		--pipeline.model.interlevel_loss_mult 0.0 \
		--pipeline.density_mismatch_start_step -1 \
		--pipeline.density_mismatch_coefficient 1e-3 \
		--pipeline.flat_field_loss_multiplier 0.0 \
		--pipeline.model.disable_mixing False \
		--optimizers.field_weighing.optimizer.lr 1e-2 \
		--optimizers.field_weighing.optimizer.weight_decay 1e-1 \
		--optimizers.field_weighing.scheduler.steady_steps 2000 \
		--optimizers.field_weighing.scheduler.max_steps $numsteps \
		--optimizers.field_weighing.scheduler.warmup_steps 200 \
		--timestamp "vel_${n0}${suf2}" \
		--machine.seed 40 \
		multi-camera-dataparser --downscale-factors.val $downscale_factor --downscale-factors.test $downscale_factor

elif [ $mode = "export_canonical" ]; then
    
	dname="$outdir/$dset/$suf"
	config_path=$dname/config.yml
	resolution=$batch_size
	echo "Exporting $dname at resolution $resolution"
	
    python $pardir/../nerfstudio/nerfstudio/scripts/exporter.py volume-grid \
        --fmt npz \
        --load-config $dname/config.yml \
        --output-dir $dname \
		--export-dtype uint8 \
        --resolution $resolution

elif [ $mode = "export_slices" ]; then
    
	dname="$outdir/$dset/$suf"
	config_path=$dname/config.yml
	which=$suf2
	resolution=$batch_size
	echo "Exporting $dname at resolution $resolution -- $which"

	times=($(seq 0 0.5 1))
	python $pardir/../nerfstudio/nerfstudio/scripts/exporter.py image-stack \
		--load-config $config_path \
		--resolution 500 \
		--num-slices 5 \
		--plane xz \
		--target field \
		--output-dir $dname/slices \
		--max_density 3.0 \
		--times ${times[@]}

elif [ $mode = "export" ]; then
    
	dname="$outdir/$dset/$suf"
	config_path=$dname/config.yml
	which=$suf2
	resolution=$batch_size
	echo "Exporting $dname at resolution $resolution -- $which"

	datasets=$(find data/experimental/$dset -maxdepth 1 -type f -name '*.npz' | sort -t '-' -k3 -n)
	nums=($(sed -E 's/.*-([0-9]+)\.npz/\1/' <<< "$datasets"))
	min_num=$(echo "${nums[0]}")
	max_num=$(echo "${nums[-1]}")

	# normalized_times=()
	# for num in "${nums[@]}"; do
	# 	normalized_time=$(awk -v n="$num" -v min="$min_num" -v max="$max_num" 'BEGIN { printf "%.2f", (n - min) / (max - min) }')
	# 	normalized_times+=($normalized_time)
	# done
	normalized_times=(0.0 0.2 0.4 0.6 0.8 1.0)
	
    for t in ${normalized_times[@]}; do
        echo "Exporting $dname, t=$t"
        python $pardir/../nerfstudio/nerfstudio/scripts/exporter.py volume-grid \
            --fmt npz \
            --load-config $dname/config.yml \
            --output-dir $dname \
			--export_dtype uint8 \
            --resolution $resolution \
            --time $t \
			--max_density 2.0 \
			--which $which
        mv $dname/volume.npz $dname/volume_t-${t}_${which}.npz

    done

elif [ $mode = "eval" ]; then

    
    dname="$outdir/$dset/$suf"
	echo "Evaluating $dname"
	config_path=$dname/config.yml

	datasets=$(find data/experimental/$dset -maxdepth 1 -type f -name '*.npz' | sort -t '-' -k3 -n)
	nums=($(sed -E 's/.*-([0-9]+)\.npz/\1/' <<< "$datasets"))
	min_num=$(echo "${nums[0]}")
	max_num=$(echo "${nums[-1]}")

	normalized_times=()
	for num in "${nums[@]}"; do
		normalized_time=$(awk -v n="$num" -v min="$min_num" -v max="$max_num" 'BEGIN { printf "%.2f", (n - min) / (max - min) }')
		normalized_times+=($normalized_time)
	done
	echo "normalized_times: ${normalized_times[@]}"
	echo "datasets: ${datasets[@]}"

	python $pardir/../nerfstudio/nerfstudio/scripts/eval.py compute-normed-correlation \
		--load-config $config_path \
		--target-times ${normalized_times[@]} \
		--target-files $(printf "%s\n" "${datasets[@]}") \
		--output-path $dname/eval_metrics_${dset}_${suf}.json \
		--npoints 400 \
		--extent -0.8 0.8 -0.8 0.8 -0.8 0.8

fi