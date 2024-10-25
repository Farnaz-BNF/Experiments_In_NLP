Models can be downloaded from [here](https://drive.google.com/drive/folders/1QkOgxTg4XkNd_qaE7gSZ_NUoeuT3DpNc?usp=sharing).

Each directory contains all files of one model.


Names of the models follow this pattern:
`xnli_{first-phase-language}_{second-phase-language}_{lr}_{epochs}`.


e.g:

the model's name `xnli_de_k10_en_3e-5_2.0` means that, first, the model is first fine-tuned with 10 instances (k) from German (de) as the target language and then, continues fine-tuning on English (en) as the source language with lr=3e-5 and epochs=2 for each phase.
