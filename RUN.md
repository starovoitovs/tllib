Create symbolic links to the datasets:

    ln -s ../wbc/Datasets/Acevedo_20/ examples/domain_adaptation/image_classification/data/wbc
    ln -s ../wbc/Datasets/Matek_19/ examples/domain_adaptation/image_classification/data/wbc
    ln -s ../wbc/Datasets/WBC1/ examples/domain_adaptation/image_classification/data/wbc

Run this command from the root of `tll`:

    # train
    CUDA_VISIBLE_DEVICES=0 python examples/domain_adaptation/image_classification/mdd.py \
        examples/domain_adaptation/image_classification/data/wbc \
        --log logs/mdd_sample_weights/WBC_AM2W \
        --phase train \
        -d WBC -s A M -t W -v W \
        -a resnet18 \
        --epochs 20 \
        --train-resizing=crop.resize \
        --val-resizing=crop.resize \
        --scale 0.8 1.0 \
        --ratio 0.8 1.2

Arguments `-s` and `-t` specify source and target dataset:

* A = Acevedo_20
* M = Matek_19
* W = WBC1

You can specify several letters (for example `A M` for Acevedo_20 and Matek_19).

Don't forget to put the root of `tlda` in the `PYTHONPATH` so the `tllib` can be imported.
