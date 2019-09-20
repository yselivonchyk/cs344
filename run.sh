#!/usr/bin/env bash
cd /home/ubuntu/repo/cs344/ps/ps1/
rm HW1_
mv HW1 HW1_
make
./HW1 cinque_terre_small.jpg out.jpg
cp out.jpg ../../

# scp ubuntu@$CIP:"/home/ubuntu/repo/cs344/ps/ps1/out.jpg" ~/repo/cs344/