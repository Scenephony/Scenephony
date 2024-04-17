## Dataset

### Overview

This dataset is a subset from [SymMV](https://github.com/zhuole1025/SymMV), a dataset from the paper [[ICCV 2023] Video Background Music Generation: Dataset, Method and Evaluation](https://arxiv.org/abs/2211.11248). All videos from the original SymMV are listed in [SymMV.csv](./SymMV.csv). Videos in the csv are provided as YouTube links, however, that some video entries are no longer available due to various reasons. 

### Setup

50 videos are downloaded with the script [video_crawl.py](./video_crawl.py), with command
```shell
python video_crawl.py 0 52
```
where the two arguments specifies the start and end video indices from [SymMV.csv](./SymMV.csv). Note that videos `00034.mp4` and `00046.mp4` are no longer available, so `[0: 52]` includes exactly 50 videos. 

The associated audio files (chord and midi) are in their respective directories. 