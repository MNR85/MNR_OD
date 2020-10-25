echo "mosse"
python3 caffeOD.py -v test_images/retail.mp4 -t mosse
python3 caffeOD.py -v test_images/los_angeles.mp4 -t mosse
python3 caffeOD.py -v test_images/chair.mp4 -t mosse
echo "mosse dwc"
python3 caffeOD.py -v test_images/retail.mp4 -t mosse -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/los_angeles.mp4 -t mosse -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/chair.mp4 -t mosse -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt

echo "kcf"
python3 caffeOD.py -v test_images/retail.mp4 -t kcf
python3 caffeOD.py -v test_images/los_angeles.mp4 -t kcf
python3 caffeOD.py -v test_images/chair.mp4 -t kcf
echo "kcf dwc"
python3 caffeOD.py -v test_images/retail.mp4 -t kcf -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/los_angeles.mp4 -t kcf -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/chair.mp4 -t kcf -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt

echo "tld"
python3 caffeOD.py -v test_images/retail.mp4 -t tld
python3 caffeOD.py -v test_images/los_angeles.mp4 -t tld
python3 caffeOD.py -v test_images/chair.mp4 -t tld
echo "tld dwc"
python3 caffeOD.py -v test_images/retail.mp4 -t tld -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/los_angeles.mp4 -t tld -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/chair.mp4 -t tld -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt

echo "medianflow"
python3 caffeOD.py -v test_images/retail.mp4 -t medianflow
python3 caffeOD.py -v test_images/los_angeles.mp4 -t medianflow
python3 caffeOD.py -v test_images/chair.mp4 -t medianflow
echo "medianflow dwc"
python3 caffeOD.py -v test_images/retail.mp4 -t medianflow -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/los_angeles.mp4 -t medianflow -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/chair.mp4 -t medianflow -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt

echo "csrt"
python3 caffeOD.py -v test_images/retail.mp4 -t csrt
python3 caffeOD.py -v test_images/los_angeles.mp4 -t csrt
python3 caffeOD.py -v test_images/chair.mp4 -t csrt
echo "csrt dwc"
python3 caffeOD.py -v test_images/retail.mp4 -t csrt -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/los_angeles.mp4 -t csrt -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt
python3 caffeOD.py -v test_images/chair.mp4 -t csrt -p ssd_mobilenet_v1_coco_2017_11_17/MobileNetSSD_deploy_dwc.prototxt