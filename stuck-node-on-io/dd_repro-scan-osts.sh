#!/bin/bash

OUT=/capstor/store/cscs/cscs/csstaff/jpcoles3
OUT=/capstor/scratch/cscs/jpcoles/tmp/alps-gh200-reproducers/stuck-node-on-io/test-dir

echo "Writing to $OUT"
echo "HOSTNAME $(hostname)"
lfs quota -p $(lsattr -p -d $OUT | awk '{print $1}') $OUT -h

sync $OUT/dd_largefile.*
rm $OUT/dd_largefile.*
sync

bs=1075761
count=400

for ((tries=0; tries<=40; tries+=1)); do
    for ((ost=0; ost<=40; ost+=1)); do
        lfs setstripe -o $ost ${OUT}

        CMD="dd if=/dev/zero of=$OUT/dd_largefile.${tries}.${ost} bs=${bs} count=${count} status=progress oflag=dsync" 
        CMD="dd if=/dev/zero of=$OUT/dd_largefile.${tries}.${ost} bs=${bs} count=${count} status=progress" 
        echo ${CMD}
        ${CMD}

        if [[ $? != 0 ]]; then
            lfs quota -p $(lsattr -p -d $OUT | awk '{print $1}') $OUT -h
            lfs getstripe ${OUT}/dd_largefile.*
            lfs quota -p $(lsattr -p -d $OUT | awk '{print $1}') $OUT -hv
        fi

        rm ${OUT}/dd_largefile.*
        sync

    done
done

