#!/bin/bash
#SBATCH --job-name=io-hang-repro
#SBATCH --gres=gpu:4
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:30:00
#SBATCH --output=io-hang-repro-%j.out

OUT=./
mkdir -p ${OUT}

sleep 5

for ntasks in 1 2; do
	for bs in 1M 2M 3M 4M 5M 4096000; do
	
		echo "------------------------------------------"
		echo "Running dd with bs=${bs} and ntasks=${ntasks}"
	
		srun -ul -n${ntasks} \
		bash -c "dd if=/dev/zero of=$OUT/dd_largefile.\$SLURM_PROCID bs=${bs} count=1000 status=progress"
	
		echo "Finished."
	
		rm $OUT/dd_largefile.*
		sleep 5
	done
done


