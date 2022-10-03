Create symbolic links to the datasets:

    ln -s ../wbc/Datasets/Acevedo_20/ examples/domain_adaptation/image_classification/data/wbc
    ln -s ../wbc/Datasets/Matek_19/ examples/domain_adaptation/image_classification/data/wbc
    ln -s ../wbc/Datasets/WBC1/ examples/domain_adaptation/image_classification/data/wbc

Run this command from the root of `tlda`:

    # old training
    CUDA_VISIBLE_DEVICES=0 python examples/domain_adaptation/image_classification/mdd.py \
        /p/project/hai_ds_isa/starovoitovs1/Datasets \
        -d WBC \
        --source A M --target W \
        -a resnet18 \
        --epochs 300 \
        --iters-per-epoch 100 \
        --seed 1 \
        --log ../tll_old/logs/custom_mean_std/WBC_AM2W \
        --margin 4 \
        --trade-off 1 \
        --phase train

Arguments `-s` and `-t` specify source and target dataset:

* A = Acevedo_20
* M = Matek_19
* W = WBC1
* T = WBC2

You can specify several letters (for example `A M` for Acevedo_20 and Matek_19).

Don't forget to put the root of `tlda` in the `PYTHONPATH` so the `tllib` can be imported.
