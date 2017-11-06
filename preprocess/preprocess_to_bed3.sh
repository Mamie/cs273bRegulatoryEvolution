mkdir preprocessed_peaks
for filename in */*macs; do
	echo "$filename"
	awk '{print $1,$2,$3}' $filename >> preprocessed_peaks/`basename "$filename"`_bed
done