Code for my B.Sc. project "heart rate estimation in videos of faces"

Much of the code is structured in jupyter notebooks the main notebooks are
```
├───notebooks
│   ├───regression_ica.ipynb
│   ├───ICA_evaluation.ipynb
└───RhythmNet
    ├───train.ipynb
```

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
