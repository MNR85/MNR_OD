#! /bin/bash
echo "New run at $(date)">>results/runLog.log
trackers=("mosse" "kcf" "tld" "medianflow" "csrt")
prototxt=("ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy.prototxt" "ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt")
evalData=("test_images/ILSVRC/ILSVRC2017_train_00006000" "test_images/ILSVRC/ILSVRC2017_train_00024000" "test_images/ILSVRC/ILSVRC2017_train_00066000")
hw=("gpu" "cpu")
methode=("serial" "pipeline")
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
                echo "python3 caffeOD.py $pFile $vFile $gMode $sMode">>results/runLog.log
            done
        done
    done
done
