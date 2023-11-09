---
annotations_creators:
- machine-generated
language:
- en
language_creators:
- found
license:
- cdla-permissive-1.0
multilinguality:
- monolingual
pretty_name: PubLayNet
size_categories: []
source_datasets:
- original
tags:
- graphic design
- layout-generation
task_categories:
- image-classification
- image-segmentation
- image-to-text
- question-answering
- other
- multiple-choice
- token-classification
- tabular-to-text
- object-detection
- table-question-answering
- text-classification
- table-to-text
task_ids:
- multi-label-image-classification
- multi-class-image-classification
- semantic-segmentation
- image-captioning
- extractive-qa
- closed-domain-qa
- multiple-choice-qa
- named-entity-recognition
---

# Dataset Card for PubLayNet

[![CI](https://github.com/shunk031/huggingface-datasets_PubLayNet/actions/workflows/ci.yaml/badge.svg)](https://github.com/shunk031/huggingface-datasets_PubLayNet/actions/workflows/ci.yaml)

## Table of Contents
- [Dataset Card Creation Guide](#dataset-card-creation-guide)
  - [Table of Contents](#table-of-contents)
  - [Dataset Description](#dataset-description)
    - [Dataset Summary](#dataset-summary)
    - [Supported Tasks and Leaderboards](#supported-tasks-and-leaderboards)
    - [Languages](#languages)
  - [Dataset Structure](#dataset-structure)
    - [Data Instances](#data-instances)
    - [Data Fields](#data-fields)
    - [Data Splits](#data-splits)
  - [Dataset Creation](#dataset-creation)
    - [Curation Rationale](#curation-rationale)
    - [Source Data](#source-data)
      - [Initial Data Collection and Normalization](#initial-data-collection-and-normalization)
      - [Who are the source language producers?](#who-are-the-source-language-producers)
    - [Annotations](#annotations)
      - [Annotation process](#annotation-process)
      - [Who are the annotators?](#who-are-the-annotators)
    - [Personal and Sensitive Information](#personal-and-sensitive-information)
  - [Considerations for Using the Data](#considerations-for-using-the-data)
    - [Social Impact of Dataset](#social-impact-of-dataset)
    - [Discussion of Biases](#discussion-of-biases)
    - [Other Known Limitations](#other-known-limitations)
  - [Additional Information](#additional-information)
    - [Dataset Curators](#dataset-curators)
    - [Licensing Information](#licensing-information)
    - [Citation Information](#citation-information)
    - [Contributions](#contributions)

## Dataset Description

- **Homepage:** https://developer.ibm.com/exchanges/data/all/publaynet/
- **Repository:** https://github.com/shunk031/huggingface-datasets_PubLayNet
- **Paper (Preprint):** https://arxiv.org/abs/1908.07836
- **Paper (ICDAR2019):** https://ieeexplore.ieee.org/document/8977963

### Dataset Summary

PubLayNet is a dataset for document layout analysis. It contains images of research papers and articles and annotations for various elements in a page such as "text", "list", "figure" etc in these research paper images. The dataset was obtained by automatically matching the XML representations and the content of over 1 million PDF articles that are publicly available on PubMed Central.

### Supported Tasks and Leaderboards

[More Information Needed]

### Languages

[More Information Needed]

## Dataset Structure

### Data Instances

```python
import datasets as ds

dataset = ds.load_dataset(
    path="shunk031/PubLayNet",
    decode_rle=True, # True if Run-length Encoding (RLE) is to be decoded and converted to binary mask.
)
```

### Data Fields

[More Information Needed]

### Data Splits

[More Information Needed]

## Dataset Creation

### Curation Rationale

[More Information Needed]

### Source Data

[More Information Needed]

#### Initial Data Collection and Normalization

[More Information Needed]

#### Who are the source language producers?

[More Information Needed]

### Annotations

[More Information Needed]

#### Annotation process

[More Information Needed]

#### Who are the annotators?

[More Information Needed]

### Personal and Sensitive Information

[More Information Needed]

## Considerations for Using the Data

### Social Impact of Dataset

[More Information Needed]

### Discussion of Biases

[More Information Needed]

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

[More Information Needed]

### Licensing Information

- [CDLA-Permissive](https://cdla.io/permissive-1-0/)

### Citation Information


```bibtex
@inproceedings{zhong2019publaynet,
  title={Publaynet: largest dataset ever for document layout analysis},
  author={Zhong, Xu and Tang, Jianbin and Yepes, Antonio Jimeno},
  booktitle={2019 International Conference on Document Analysis and Recognition (ICDAR)},
  pages={1015--1022},
  year={2019},
  organization={IEEE}
}
```

### Contributions

Thanks to [ibm-aur-nlp/PubLayNet](https://github.com/ibm-aur-nlp/PubLayNet) for creating this dataset.
