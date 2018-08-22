# Deep Compressive Autoencoder for Large-Scale Spike Compression

The repository contains source codes for the paper *Deep Compressive Autoencoder for Action Potential Compression in Large-Scale Neural Recording* accepted by *Journal of Neural Engineering*.

[Pytorch 0.4.1](https://pytorch.org/) is required to run the model. An Nvidia GPU with over 2GB memory  is preferred.

## Authors
Tong Wu<sup>1</sup>, Wenfeng Zhao<sup>1</sup>, Edward Keefer<sup>2</sup>, Zhi Yang<sup>1</sup>

<sup>1</sup> Department of Biomedical Engineering, University of Minnesota, Minneapolis, MN, 55455, USA

<sup>2</sup> Nerves Incorporated, Dallas, TX, 75214, USA

## Datasets
Three datasets are used in the paper, which are all publicly available and can be downloaded at the following links:

- [**Wave_Clus**](https://github.com/csn-le/wave_clus)
- [**HC1**](http://crcns.org/data-sets/hc/hc-1/about)
- [**Neuropixels**](http://data.cortexlab.net/singlePhase3/) (requires permission from [Dr. Nick Steinmetz](http://www.nicksteinmetz.com/) if used for publication)

Spikes can be extracted from the datasets either according to their ground truth timing (if available) or using detection methods.

Alignment of spikes is required before compression.

Spikes need to be organized into dimensions (*Num_batch* x *Num_channel* x *Num_pionts*) before loading into CAE, where *Num_channel* is *M<sub>spk</sub>* (typically equal to or smaller than the number of recording channels), *Num_points* is the length of each spike.

Data should be in .mat format and put in the folder `./data` (not contained in the repository; needs to be created by users).

## How to use

`spk_vq_cae.ipynb` is the top-level notebook. 

Under the section `Parameter Configuration`:

- `spk_ch` is *M<sub>spk</sub>*, by default 4.
- `spk_dim` is *Num_points*.
- `vq_num` is the number of vector quantization codes.
- `cardinality` is the number of groups in a CNN layer, by default 32.
- `dropRate` is the rate of dropout, by default 0.2.
- `batch_size` is the batch size, by default 48.

Under the section `Preparing data loaders`:

- For non-denoising autoencoder, `noise_file` and `clean_file` refer to the same dataset.
- For denoising autoencoder, `noise_file` refers to a distorted version of `clean_file`.

