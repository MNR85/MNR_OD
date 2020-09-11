My works on this internet code:
https://github.com/ambakick/Person-Detection-and-Tracking.git

Done:
- Separating detection and tracking
- Draw box for detection (not used yet)
- add class type to tracker matching
Working:
- change tracker
- Fixing tracking bug when a single bad object detected
Future:
- nms
- pipelining
- distribute 1-5 between track and detection (less detection more tracking... need think for algorithm!)
- Optimize code (loops and ...)
- Using TensorRT/Tensorflow-lite instead of Tensorflow
- use other tracking like: https://github.com/abewley/sort and https://github.com/SpyderXu/ssd_sort
- divide image in small part and run lighter network on each part
