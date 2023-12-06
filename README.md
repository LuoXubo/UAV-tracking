# UAV-tracking

Source codes of the UAV tracking under planetary scenes.

## Test

To test the tracking methods, you can use the following command:

```
python tracking.py --video_path=<video_path> --tracker=<tracker_id>
```

Specifically, let video_path=0, if you want to use your camera steam. There are 3 tracking methods of OpenCV implemented in this repo:

1. KCF
2. MOSSE
3. CSRT
