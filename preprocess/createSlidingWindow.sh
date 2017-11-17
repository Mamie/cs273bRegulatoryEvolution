DIRECTORY=$1

for i in $DIRECTORY/*bed; do
    python createSlidingWindow.py $i "${i%.bed}_slided.bed"
done
