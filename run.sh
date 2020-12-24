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
prototxt=("ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt" "ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt")
evalData=("test_images/ILSVRC/ILSVRC2017_train_00006000" "test_images/ILSVRC/ILSVRC2017_train_00024000" "test_images/ILSVRC/ILSVRC2017_train_00066000")
hw=("gpu" "cpu")
methode=("serial" "pipeline")
fixedRatio=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25)
currentPoint=0
echo "Dynamic portion $(date)">>results/runLog.log
for p in "${prototxt[@]}"; do
    pFile="-p $p "
    for e in "${evalData[@]}"; do
        vFile="-v ${e}.mp4 -a ${e}/ "
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
                if [ "$currentPoint" -gt "$checkPoint" ]; then
                    cmd="python3 caffeOD.py $pFile$vFile$gMode$sMode"
                    cmdDebug="$cmd-d -e"
                    echo "$currentPoint - $cmd"
                    echo "$currentPoint - $cmd">>results/runLog.log
                    eval $cmd
                    echo "$currentPoint - $cmdDebug"
                    echo "$currentPoint - $cmdDebug">>results/runLog.log
                    eval $cmdDebug
                fi
                currentPoint=$((currentPoint+1))
                echo $currentPoint>results/checkPoint
            done
        done
    done
done
echo "Static portion $(date)">>results/runLog.log
for p in "${prototxt[@]}"; do
    pFile="-p $p"
    for e in "${evalData[@]}"; do
        vFile="-v ${e}.mp4 -a ${e}/"
        for h in "${hw[@]}"; do
            if [ $h = "gpu" ]; then
                gMode="-g"
            else
                gMode=""
            fi
            for m in "${methode[@]}"; do
                if [ $m = "serial" ]; then
                    sMode="-s"
                else
                    sMode=""
                fi
                for r in "${fixedRatio[@]}"; do
                    ratio="-r $r"
                    if [ "$currentPoint" -gt "$checkPoint" ]; then
                      cmd="python3 caffeOD.py $pFile $vFile $gMode $sMode $ratio"
                      cmdDebug="$cmd -d -e"
                      echo "$currentPoint - $cmd"
                      echo "$currentPoint - $cmd">>results/runLog.log
                      eval $cmd
                      echo "$currentPoint - $cmdDebug"
                      echo "$currentPoint - $cmdDebug">>results/runLog.log
                      eval $cmdDebug
                    fi
                    currentPoint=$((currentPoint+1))
                    echo $currentPoint>results/checkPoint
                done
            done
        done
    done
done

echo "1" | sudo -S shutdown now -h
