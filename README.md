# Heart-Rate-Estimation
B.Sc. project "heart rate estimation in videos of faces"


##komprimer video
skaler til 2/3

```ffmpeg -i cpi_gym.avi -vf "scale=trunc(iw*(2/3)/2)*2:trunc(ih*(2/3)/2)*2" comp_vid.mp4```

kun komprimer

```ffmpeg -i cpi_gym.avi -vf comp_vid.mp4```

```ffmpeg -i cpi_gym.avi -vf comp_vid.mjpg```

komprimer hel mappe 
```for i in *.avi; do ffmpeg -i "$i" "${i%.*}.mp4"; done```
