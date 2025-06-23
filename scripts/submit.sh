#!/bin/zsh

dset=$1

# Only create a copy if we're not already running from a copy
if [ -z "$RUNNING_FROM_COPY" ]; then
    # Create a temporary copy of this script
    SCRIPT_COPY=$(mktemp)
	RUNSCRIPT_COPY=$(mktemp)
    cp "$0" "$SCRIPT_COPY"
	cp ./run_dset.sh "$RUNSCRIPT_COPY"
    chmod +x "$SCRIPT_COPY"
	chmod +x "$RUNSCRIPT_COPY"
	echo "Running from $SCRIPT_COPY"
    # Re-execute from the copy with a flag
    RUNNING_FROM_COPY=1 RUNSCRIPT_PATH="$RUNSCRIPT_COPY" "$SCRIPT_COPY" "$@"
	echo "Removing $SCRIPT_COPY and $RUNSCRIPT_COPY"
    rm "$SCRIPT_COPY"
	rm "$RUNSCRIPT_COPY"
    exit
fi

echo "Script running from $RUNSCRIPT_PATH"

numsteps=3000

# /bin/zsh "$RUNSCRIPT_PATH" $dset canonical 2048 '' '' '' '' '' '' '' '' $numsteps || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset export_canonical 250 'nerf_xray/canonical_F' || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset vfield 2048 '' '' 6 6 3000 1e-3 1000 0.1 $numsteps || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset vfield 1024 '' '' 6 9 6000 1e-3 200 0.09 $numsteps || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset vfield 512 '' '' 9 15 9000 1e-7 200 0.085 $numsteps || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset vfield 1024 '' '' 15 27 12000 1e-7 200 0.075 $numsteps || exit 1	
# /bin/zsh "$RUNSCRIPT_PATH" $dset vfield 1024 '' '' 27 51 15000 1e-7 200 0.0499 $numsteps || exit 1	
# /bin/zsh "$RUNSCRIPT_PATH" $dset spatiotemporal_mix 2048 '' '' 51 9 18000 1e-7 200 0.0499 $numsteps || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset export 500 'spatiotemporal_mix/vel_51' 'mixed' || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset export 500 'spatiotemporal_mix/vel_51' 'forward' || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset export 500 'spatiotemporal_mix/vel_51' 'backward' || exit 1
# /bin/zsh "$RUNSCRIPT_PATH" $dset eval 0 'spatiotemporal_mix/vel_51' || exit 1
