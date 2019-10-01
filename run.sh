#!/usr/bin/env bash
cd /home/ubuntu/repo/cs344/ps/ps4/
rm HW4_
mv HW4 HW4_
make
./HW4 red_eye_effect_5.jpg red_eye_effect_template_5.jpg ../../out.jpg
cp ./red_eye_effect.gold ../../out2.jpg

# scp ubuntu@$CIP:"/home/ubuntu/repo/cs344/ps/ps1/HW2_output.png" ~/repo/cs344/
# scp ubuntu@$CIP:"/home/ubuntu/repo/cs344/ps/ps1/HW2_differenceImage.png" ~/repo/cs344/