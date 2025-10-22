#!/bin/bash
#SBATCH --job-name=io-hang-repro
#SBATCH --gres=gpu:4
#SBATCH --mem=460000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=00:03:00
#SBATCH --output=io-hang-repro-%j.out

#OUT=/capstor/scratch/cscs/jpcoles/stuck-io-temp-dir
#OUT=/capstor/store/cscs/cscs/csstaff/jpcoles/stuck-io-temp-dir
OUT=/capstor/store/cscs/cscs/csstaff/jpcoles3
#OUT=/capstor/store/cscs/cscs/csstaff/jpcoles2
#mkdir -p ${OUT}

#lfs setstripe -S 128k ${OUT}
#lfs setstripe -i 0 -c 2 ${OUT}
lfs setstripe -c 1 ${OUT}
#lfs setstripe -i 0 -c 1 ${OUT}

echo "Writing to $OUT"
echo "HOSTNAME $(hostname)"
lfs quota -p $(lsattr -p -d $OUT | awk '{print $1}') $OUT -h

QUOTA=$(($(lfs quota -p $(lsattr -p -d $OUT | awk '{print $1}') $OUT -q | xargs | awk '{print $3}' | tr -d '*') * 1024))

#lfs getstripe -v ${OUT}/dd_largefile.* 2>/dev/null | awk '{print "OLD:", $0}'

sync $OUT/dd_largefile.*
rm $OUT/dd_largefile.*
#sleep 10

#BS_LIST="${BS_LIST}  1075761  1075770  1075800  1076000  1080000  1093000  2010000  2080000  2095000  4096000"
#BS_LIST="${BS_LIST} 1M 2M 3M 4M 8M 16M 32M"
#BS_LIST="${BS_LIST} 10757610 10757700 10758000 10760000 10800000 10930000 20100000 20800000 20950000 40960000"
#BS_LIST="${BS_LIST} $(shuf -i 1-4096000 -n 10000)"

# This list is very important. The sizes trigger a hang when the lustre bug is present
BS_LIST="${BS_LIST} 1075761 1075770 1075800 1076000  1080000  1093000  2010000  2080000  2095000  4096000"

COUNTS="1000 7000 9000"
COUNTS="${COUNTS} 70000 90000"
#COUNTS="70000 90000"

#BS_LIST=1048575
#COUNTS=10240

for count in ${COUNTS}; do
	for bs in ${BS_LIST}; do

                if [[ ${QUOTA} -gt 0 ]]; then
                    if [[ $((${bs} * ${count} > ${QUOTA})) == 1 ]]; then
                        echo ${count} ${bs} is over quota ${QUOTA}. Skipping.
                        continue;
                    fi
                fi

                echo 

                for ((i=0; i < 10; i+=1)); do
                    USAGE=$(($(lfs quota -p $(lsattr -p -d $OUT | awk '{print $1}') $OUT -q | xargs | awk '{print $2}' | tr -d '*') * 1024))
                    if [[ ${USAGE} -ge 10240 ]]; then
                        echo "Files still occupying ${USAGE}. Waiting 5s for fs to update."
                        sleep 5
                    fi
                done

		echo "------------------------------------------"
		lfs quota -p $(lsattr -p -d $OUT | awk '{print $1}') $OUT -h
		echo "Running dd with bs=${bs} and ntasks=${ntasks} at $(date)"

		CMD="dd if=/dev/zero of=$OUT/dd_largefile.0 bs=${bs} count=${count} status=progress" 
		echo ${CMD}
		bash -c "${CMD}"

		#lfs getstripe -v ${OUT}/dd_largefile.* 2>/dev/null | awk '{print "NEW:", $0}'

		sync $OUT/dd_largefile.*
		sleep 5
                ~/TMP/file-ost.sh $OUT/dd_largefile.*
		rm $OUT/dd_largefile.*
                sync
                lfs df > /dev/null
		#sleep 10

		echo "Finished."

	done
done

echo
echo "SUCCESS"
echo


