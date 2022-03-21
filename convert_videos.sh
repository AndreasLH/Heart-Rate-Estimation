# converts all avi videoes in all subfolders to mp4
cd cpi

# unzip -o *.zip

find . -type f -name "*.avi" -exec bash -c 'FILE="$1"; ffmpeg -i "${FILE}"  "${FILE%.avi}.mp4";' _ '{}' \;

find . -type f -name "*.avi" -exec bash -c 'FILE="$1"; rm "${FILE}";' _ '{}' \;