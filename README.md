### The official "Look where you’re going: Classifying drivers' attention through 3D gaze estimation" repository


### Run the monitoring system:

* Install all requirements:
    ```bash
    pip install -r requirements.txt 
    ```

* Download the gaze estimation model L2CSNet_gaze360.pkl from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) or [here](https://github.com/Ahmednull/L2CS-Net) and store it in the `./models/` folder

* Run it:
```bash
python3 driver_monitoring_system.py --gaze_model ./models/L2CSNet_gaze360.pkl --gpu 0 --video_source {{SOURCE}} --video_output {{OUTPUT}} --distraction_model ./models/gnb.pkl
```

