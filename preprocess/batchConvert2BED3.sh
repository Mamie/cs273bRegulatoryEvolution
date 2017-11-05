DIRECTORY=$1

for i in $DIRECTORY/*_macs; do
    python convertPeakCall2BED3.py $i "$i.BED3"
done
