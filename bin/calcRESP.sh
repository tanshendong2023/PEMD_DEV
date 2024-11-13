# A script to calculate RESP or RESP2 charges by invoking Multiwfn
# Written by Tian Lu (sobereva@sina.com), 2020-Jun-30
# Example of calculation of RESP charges: ./calcRESP.sh H2CO_gas.fchk
# Example of calculation of RESP2 charges: ./calcRESP.sh H2CO_gas.fchk H2CO_water.fchk

#!/bin/bash

# Load Multiwfn module
module load Multiwfn/3.8
input_dir=$(dirname "$1")
cd "$input_dir"

echo Calculating RESP charge for $1 ...
Multiwfn $1 -nt 32 << EOF > /dev/null
7
18
1
y
0
0
q
EOF
chgfile="${1%.*}.chg"
echo RESP charges for $1 has been exported to $chgfile in current folder

if [ $2 ];then
echo
echo Calculating RESP charge for $2 ...
Multiwfn $2 -nt 32 << EOF > /dev/null
7
18
1
y
0
0
q
EOF
chgfile2=${2%.*}.chg
echo RESP charges for $1 has been exported to $chgfile2 in current folder

delta=0.5
if [ $3 ];then
	delta=$3
fi
echo
echo delta parameter is $delta

#Calculate RESP2 charge
paste $chgfile $chgfile2 |awk '{printf $1 " " $2 " " $3 " " $4 " " (1-d)*$5+d*$10 "\n"}' d=$delta > RESP2.chg
echo RESP2 charges has been written to RESP2.chg
fi
