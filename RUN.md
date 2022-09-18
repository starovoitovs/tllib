Create symbolic links to the datasets:

    ln -s ../wbc/Datasets/Acevedo_20/ examples/domain_adaptation/image_classification/data/wbc
    ln -s ../wbc/Datasets/Matek_19/ examples/domain_adaptation/image_classification/data/wbc
    ln -s ../wbc/Datasets/WBC1/ examples/domain_adaptation/image_classification/data/wbc

Run this command from the root of `tlda`:

    CUDA_VISIBLE_DEVICES=0 python examples/domain_adaptation/image_classification/dann.py examples/domain_adaptation/image_classification/data/wbc -d WBC -s A -t M -a resnet18 --epochs 20 --seed 1 --log logs/dann/WBC_A2M

Arguments `-s` and `-t` specify source and target dataset:

* A = Acevedo_20
* M = Matek_19
* B = both Acevedo_20 and Matek_19
* W = WBC1

Don't forget to put the root of `tlda` in the `PYTHONPATH` so the `tllib` can be imported.
