#!/usr/bin/env bash
cd /home/ubuntu/repo/cs344/ps/ps2/
rm HW2_
mv HW2 HW2_
make
./HW2 cinque_terre_small.jpg
mv HW2_output.png ./out.jpg
mv HW2_differenceImage.png ./out2.jpg
cp out.jpg ../../
cp out2.jpg ../../

# scp ubuntu@$CIP:"/home/ubuntu/repo/cs344/ps/ps1/HW2_output.png" ~/repo/cs344/
# scp ubuntu@$CIP:"/home/ubuntu/repo/cs344/ps/ps1/HW2_differenceImage.png" ~/repo/cs344/