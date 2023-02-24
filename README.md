### The official repository for "Look where you’re going: Classifying drivers' attention through 3D gaze estimation"

![Gif demo](/demo/low_quality.gif)

### Run the monitoring system:

* Install all requirements:
    ```bash
    pip3 install -r requirements.txt 
    ```

* Download the gaze estimation model L2CSNet_gaze360.pkl from [here](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd?usp=sharing) or [here](https://github.com/Ahmednull/L2CS-Net) and store it in the `./models/` folder

* Run it:
    ```bash
    python3 driver_monitoring_system.py --gaze_model ./models/L2CSNet_gaze360.pkl --gpu 0 --video_source {{SOURCE}} --video_output {{OUTPUT}} --distraction_model ./models/gnb.pkl
    ```


### References


https://dmd.vicomtech.org/
```
@InProceedings{jOrtega2020, 
    author="Ortega, Juan Diego and Kose, Neslihan and Cañas, Paola and Chao, Min-An and Unnervik, Alexander and Nieto, Marcos and Otaegui, Oihana and Salgado, Luis",
    editor="Bartoli, Adrien and Fusiello, Andrea",
    title="DMD: A Large-Scale Multi-modal Driver Monitoring Dataset for Attention and Alertness Analysis",
    booktitle="Computer Vision -- ECCV 2020 Workshops", year="2020",
    publisher="Springer International Publishing", pages="387--405", isbn="978-3-030-66823-5",
    doi="10.1007/978-3-030-66823-5_23"
}
```

https://github.com/Ahmednull/L2CS-Net
```
@misc{https://doi.org/10.48550/arxiv.2203.03339,
    doi = {10.48550/ARXIV.2203.03339},  
    url = {https://arxiv.org/abs/2203.03339},
    author = {Abdelrahman, Ahmed A. and Hempel, Thorsten and Khalifa, Aly and Al-Hamadi, Ayoub},
    keywords = {Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Robotics (cs.RO), FOS: Computer and information sciences, FOS: Computer and information sciences},
    title = {L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments},
    publisher = {arXiv},
    year = {2022},
    copyright = {Creative Commons Attribution 4.0 International}
}
```

https://github.com/elliottzheng/face-detection
```
@inproceedings{deng2019retinaface,
    title={RetinaFace: Single-stage Dense Face Localisation in the Wild},
    author={Deng, Jiankang and Guo, Jia and Yuxiang, Zhou and Jinke Yu and Irene Kotsia and Zafeiriou, Stefanos},
    booktitle={arxiv},
    year={2019}
}
```
