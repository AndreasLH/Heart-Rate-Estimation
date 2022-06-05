Code for my B.Sc. project "heart rate estimation in videos of faces"

Much of the code is structured in jupyter notebooks everything related to the ICA method in contained in the notebooks folder while the RhythmNet folder contains everything related to the RhythmNet model, the main notebooks are. 
```
├───notebooks
│   ├───regression_ica.ipynb
│   ├───ICA_evaluation.ipynb
└───RhythmNet
    ├───train.ipynb
```

Quick links to the notebooks

- [regression_ica.ipynb](https://nbviewer.org/github/AndreasLH/Heart-Rate-Estimation/blob/main/notebooks/regression_ica.ipynb)
- [ICA_evaluation.ipynb](https://nbviewer.org/github/AndreasLH/Heart-Rate-Estimation/blob/main/notebooks/ICA_evaluation.ipynb)
- [RhythmNet training](https://nbviewer.org/github/AndreasLH/Heart-Rate-Estimation/blob/main/RhythmNet/train.ipynb)

- [Samples of modified videos](https://drive.google.com/drive/folders/1XFiRorYi2KYkomA9VEbjXJshq_uuz7g0?usp=sharing) (roughly the videos in notebooks/videos)

data is available [here](https://github.com/partofthestars/LGI-PPGI-DB)

## Project Structure
Due to storage limitations, datafiles are not included on in the repo, but they should be structured like the following
```
├───checkpoint
├───cpi
│   ├───idx
├───data
│   ├───hr_data
│   ├───st_maps
│   │   ├───test
│   │   └───train
│   ├───test
│   └───train
├───notebooks
│   ├───videos
│   ├───x
└───RhythmNet
    ├───x
```


## Compressing videos with ffmpeg
scale to 2/3

```ffmpeg -i cpi_gym.avi -vf "scale=trunc(iw*(2/3)/2)*2:trunc(ih*(2/3)/2)*2" comp_vid.mp4```

only compress (what I used)

```ffmpeg -i cpi_gym.avi -vf comp_vid.mp4```

```ffmpeg -i cpi_gym.avi -vf comp_vid.mjpg```

compress entire directory 
```for i in *.avi; do ffmpeg -i "$i" "${i%.*}.mp4"; done```
