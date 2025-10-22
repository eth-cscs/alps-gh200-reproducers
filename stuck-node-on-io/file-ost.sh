#!/bin/bash
#
# Display the usage/quota of each Lustre OST that is hosting a given file.
# 
# Usage: file-ost.sh <files...>
#
# Example output:
#
# $ ./file-ost.sh /iopsstor/scratch/cscs/jpcoles/CONTAINERS/ngc_pt_jan.sqsh
# /iopsstor/scratch/cscs/jpcoles/CONTAINERS/ngc_pt_jan.sqsh
# iopsstor-MDT0000_UUID  236.1M*  -  4k  -  391597*  -  1  -
# iopsstor-MDT0001_UUID       0k  -  0k  -        0  -  0  -
# iopsstor-OST0000_UUID   8.144T  -  0k  -        -  -  -  -
# iopsstor-OST0001_UUID   7.938T  -  0k  -        -  -  -  -
# iopsstor-OST0002_UUID   8.363T  -  0k  -        -  -  -  -
# iopsstor-OST0003_UUID   8.011T  -  0k  -        -  -  -  -
# iopsstor-OST0004_UUID   8.302T  -  0k  -        -  -  -  -
# iopsstor-OST0005_UUID    8.32T  -  0k  -        -  -  -  -
# iopsstor-OST0006_UUID   7.489T  -  0k  -        -  -  -  -
# iopsstor-OST0007_UUID   7.583T  -  0k  -        -  -  -  -
# iopsstor-OST0008_UUID   7.837T  -  0k  -        -  -  -  -
# iopsstor-OST0009_UUID   7.864T  -  0k  -        -  -  -  -
# iopsstor-OST000a_UUID   8.554T  -  0k  -        -  -  -  -
# iopsstor-OST000b_UUID   7.947T  -  0k  -        -  -  -  -
# iopsstor-OST000c_UUID   7.743T  -  0k  -        -  -  -  -
# iopsstor-OST000d_UUID   8.067T  -  0k  -        -  -  -  -
# iopsstor-OST000e_UUID   7.762T  -  0k  -        -  -  -  -
# iopsstor-OST000f_UUID   7.848T  -  0k  -        -  -  -  -
# iopsstor-OST0010_UUID   9.027T  -  0k  -        -  -  -  -
# iopsstor-OST0011_UUID   7.756T  -  0k  -        -  -  -  -
# iopsstor-OST0012_UUID   7.593T  -  0k  -        -  -  -  -
# iopsstor-OST0013_UUID   8.729T  -  0k  -        -  -  -  -
#
# Written by Jonathan Coles <jonathan.coles@cscs.ch>
#

function ost() {
    FILE=$1
    MNT=$(stat -c %m ${FILE})
    DIR=$(dirname ${FILE})

    QTMP=$(mktemp)
    if [[ ! -e ${QTMP} ]]; then
        echo "Error creating temporary file. Skipping ${FILE}."
        return
    fi

    echo ${FILE}
    lfs quota -v -h -p $(lsattr -p -d $DIR | awk '{print $1}') $DIR > ${QTMP}

    # loop over the obj indices for the file
    for IDX in $(lfs getstripe -q ${FILE} | tail -n +3 | awk '{print $1}'); do
        # map those indices to the actual OST
        for OST in $(lfs df ${MNT} | awk '{print $1, $6}' | egrep "OST:|MDT:" | grep ":${IDX}]" | awk '{print $1}'); do
            # show the quota for the path on just that OST
            grep -A1 ${OST} ${QTMP} | xargs
        done 
    done | sort -k1,1 | column -t -R 1,2,3,4,5,6,7,8,9 

    rm -f ${QTMP}
}

FILES="$*"
if [[ ! -z ${FILES} ]]; then
    for f in $(realpath $*); do
        ost $f
    done
fi
