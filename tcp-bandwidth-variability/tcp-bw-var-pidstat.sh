#!/bin/bash
#SBATCH -J tcp-bw-var
#SBATCH -t 1:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH --exclusive --mem=450G

IPERF="numactl --cpunodebind=0 --membind=0 iperf3"

OFILE=nj

function submit_pair()
{
	server=$1
	client=$2

	SRUN="srun -u -N1 -n1 --exclusive --cpus-per-task=${SLURM_NTASKS_PER_NODE} --mem=50G"
	TS="%Y-%m-%dT%H:%M:%S%t"
	LOG=${OFILE}-${server}-${client}.out

	XNAME_SERVER=$(${SRUN} -w ${server} cat /etc/cray/xname)
	XNAME_CLIENT=$(${SRUN} -w ${client} cat /etc/cray/xname)

	COMMON_OPTS="--forceflush --timestamps=${TS} --affinity=10 --bind-dev=hsn0"

	SERVER_OPTS="${COMMON_OPTS} -s"
	${SRUN} -w ${server} ${IPERF} ${SERVER_OPTS}  &
	SERVER_PID=$!


	# wait for server to start
	sleep 5


	echo "date:   $(date +${TS})"            >  ${LOG}
	echo "server: ${server} ${XNAME_SERVER}" >> ${LOG}
	echo "client: ${client} ${XNAME_CLIENT}" >> ${LOG}

	for i in $(seq 60); do
		CLIENT_OPTS="${COMMON_OPTS} -c ${server} --time=20" # --zerocopy --affinity=10 --bitrate=64G"
		${SRUN} --open-mode=append -o ${LOG} -w ${client} bash -c "export S_COLORS=never; mpstat -I CPU 1 -P 10 & ${IPERF} ${CLIENT_OPTS}"
		#${SRUN} --open-mode=append -o ${LOG} -w ${client} pidstat -h -udrw 1 -e ${IPERF} ${CLIENT_OPTS}
		sleep 1
	done

	kill ${SERVER_PID}
}


NODES=$(scontrol show hostnames $SLURM_JOB_NODELIST)

submit_pair ${NODES}

wait

