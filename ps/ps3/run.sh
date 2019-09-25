#!/usr/bin/env bash
cd /home/ubuntu/repo/cs344/ps/ps3/
rm HW3_
mv HW3 HW3_
make
./HW3 memorial_raw.png
mv HW3_output.png          ./out.jpg
cp memorial_png.gold ./out2.jpg
cp out.jpg  ../../
cp out2.jpg ../../

# scp ubuntu@$CIP:"/home/ubuntu/repo/cs344/ps/ps1/HW2_output.png" ~/repo/cs344/
# scp ubuntu@$CIP:"/home/ubuntu/repo/cs344/ps/ps1/HW2_differenceImage.png" ~/repo/cs344/