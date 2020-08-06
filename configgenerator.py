import json

jsondata = {
    "word2vecfile": "/root/tempdir/text-segmentation/data/word2vec/GoogleNews-vectors-negative300.bin",
    "choidataset": "/root/tempdir/text-segmentation/data/choi",
    "manifestodataset": "/root/tempdir/text-segmentation/data/manifesto",
    "wikidataset": "/home/omri/datasets/wikipedia/process_dump_r",
}

with open('config.json', 'w') as f:
    json.dump(jsondata, f)
