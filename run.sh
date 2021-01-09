#! /bin/bash
trap "exit" INT
if [ ! -d results ]; then
    mkdir results
fi
if [ -f results/checkPoint ]; then
    checkPoint=$(cat results/checkPoint)
fi
echo "New run at $(date)">>results/runLog.log
trackers=("mosse" "kcf" "tld" "medianflow" "csrt")
models=("ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy" "ssd_mobilenet_v1_coco_2017_11_17/shicai_mobilenet_deploy" "ssd_mobilenet_v1_coco_2017_11_17/shicai_mobilenet_v2_deploy")
netType=("" "_dwc")
evalData=("test_images/ILSVRC/ILSVRC2017_train_00006000" "test_images/ILSVRC/ILSVRC2017_train_00024000" "test_images/ILSVRC/ILSVRC2017_train_00066000")
hw=("gpu" "cpu")
methode=("serial" "pipeline")
fixedRatio=(1 2 3 4 5 6 7 8 9 10 15 20 25 30 35)
currentPoint=0
echo "Dynamic pipeline $(date)">>results/runLog.log

for h in "${hw[@]}"; do
  if [ $h = "gpu" ]; then
    gMode="-g "
  else
    gMode=""
  fi
    for t in "${trackers[@]}"; do
      tType="-t $t "
        for mo in "${models[@]}"; do
          for nt in "${netType[@]}"; do
            pFile="-p ${mo}${nt}.prototxt -m ${mo}.caffemodel "
            for e in "${evalData[@]}"; do
              if [[ $currentPoint -ge $checkPoint ]]; then
                vFile="-v ${e}.mp4 -a ${e}/ "
                cmd="python3 caffeOD.py $pFile$vFile$gMode$tType"
                cmdEval="$cmd-e"
                cmdDebug="$cmd-d -e"
                echo "$currentPoint - $cmd"
                echo "$currentPoint - $cmd">>results/runLog.log
#                eval $cmd
                eval $cmdEval
#                eval $cmdDebug
              fi
              currentPoint=$((currentPoint+1))
              echo $currentPoint>results/checkPoint
            done
      done
    done
  done
done
exit
echo "Others $(date)">>results/runLog.log
for h in "${hw[@]}"; do
  if [ $h = "gpu" ]; then
    gMode="-g "
  else
    gMode=""
  fi
  for m in "${methode[@]}"; do
    if [ $m = "serial" ]; then
      sMode="-s "
    else
      sMode=""
    fi
    for t in "${trackers[@]}"; do
      tType="-t $t "
      for r in "${fixedRatio[@]}"; do
        ratio="-r $r "
        for mo in "${models[@]}"; do
          for nt in "${netType[@]}"; do
            pFile="-p ${mo}${nt}.prototxt -m ${mo}.caffemodel "
            for e in "${evalData[@]}"; do
              if [[ $currentPoint -ge $checkPoint ]]; then
                vFile="-v ${e}.mp4 -a ${e}/ "
                cmd="python3 caffeOD.py $pFile$vFile$gMode$sMode$tType$ratio"
                cmdEval="$cmd-e"
                cmdDebug="$cmd-d -e"
                echo "$currentPoint - $cmd"
                echo "$currentPoint - $cmd">>results/runLog.log
#                eval $cmd
                eval $cmdEval
#                eval $cmdDebug
              fi
              currentPoint=$((currentPoint+1))
              echo $currentPoint>results/checkPoint
            done
          done
        done
      done
    done
  done
done
echo "1" | sudo -S shutdown now -h
