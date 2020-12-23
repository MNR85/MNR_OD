#! /bin/bash
trap "exit" INT
if [ ! -d results ]; then
    mkdir results
fi
echo "New run at $(date)">>results/runLog.log
trackers=("mosse" "kcf" "tld" "medianflow" "csrt")
prototxt=("ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt" "ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt")
evalData=("test_images/ILSVRC/ILSVRC2017_train_00006000" "test_images/ILSVRC/ILSVRC2017_train_00024000" "test_images/ILSVRC/ILSVRC2017_train_00066000")
hw=("gpu" "cpu")
methode=("serial" "pipeline")
fixedRatio=(10)
echo "Dynamic portion $(date)">>results/runLog.log
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
                    cmd="python3 caffeOD.py -d -e $pFile $vFile $gMode $sMode"
                    echo "$cmd"
                    echo "$cmd">>results/runLog.log
                    eval $cmd
                done

            done
        done
    done
done
echo "Static portion $(date)">>results/runLog.log
fixedRatio=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 20 25)
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
                    cmd="python3 caffeOD.py -d -e $pFile $vFile $gMode $sMode" #$ratio"
                    echo "$cmd"
                    echo "$cmd">>results/runLog.log
                    eval $cmd
                done

            done
        done
    done
done
