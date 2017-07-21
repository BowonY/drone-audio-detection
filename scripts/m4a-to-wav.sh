#!/bin/bash
#
# <before use>
# mkdir m4a
# mkdir wav
# ...
# cp -r ... m4a

echo "** m4a to wav conversion start"
for f in m4a/*.m4a; do
	[ -f "$f" ] || continue
	filename="$(basename $f)"	#filename (with extension)
	justfilename="${filename%.*}"	#filename (without extension)
	#echo "Processing $f file"
	ffmpeg -loglevel panic -i $f -f wav "wav/$justfilename.wav"
	#ffmpeg -i $f -f wav -ac 2 "wav/$justfilename.wav"
	#echo "$filename"
	#printf "*** ffmpeg -i $f wav/$justfilename.wav\n"
done
echo "** m4a to wav conversion end"
