DIRECTORY=$1

for i in $DIRECTORY/*.BED3; do
    python createSlidingWindow.py $i "${i%BED3}tsv"
done
