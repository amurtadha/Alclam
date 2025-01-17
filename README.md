# AlcLaM
AlcLaM is an Arabic dialectal pretrained-language model introduced in the paper. 

## Huggingface
[alclam-base-v1](https://huggingface.co/rahbi/alclam-base-v1) and [alclam-base-v2](https://huggingface.co/rahbi/alclam-base-v2) are publically available in huggingface:https://huggingface.co/rahbi/

## Pipeline
A pipeline for DID is also available on huggingface:
```
from transformers import pipeline

classifier = pipeline("text-classification", model="rahbi/alclam-base-v1-DID")
result = classifier("ما ذلحين والله ما عاد اصدقك")
print(result)
```
Output:
```
[{'label': 'يمني', 'score': 0.999321460723877}]
```
## Comparision
To use our model, simply run `fine_tuning.py`. To achieve the results presented in the paper, use the folder `Ft`.
Below is a performance comparison of AlcLaM with other popular pretrained-language models on various datasets.
#### F1 Comparison of Various PLMs on Different Datasets

|              |    Task    | mBERT                     | LaBSE          | AraBERT                   | ArBERT         | MdBERT         | CAMeL          | MARBERT        | AlcLaM         |
|--------------|------------|---------------------------|----------------|---------------------------|----------------|----------------|----------------|----------------|----------------|
| DID          | MADAR-2    | 72.9 ± 16.9               | 86.6 ± 0.5     | 87.1 ± 0.2                 | 87.1 ± 0.2     | 86.0 ± 0.6     | 87.5 ± 1.0     | 85.3 ± 3.8     | **98.2 ± 0.1** |
|              | MADAR-6    | 91.3 ± 0.1                | 91.1 ± 0.2     | 91.6 ± 0.1                 | 91.6 ± 0.2     | 91.6 ± 0.0     | 92.0 ± 0.1     | 92.2 ± 0.2     | **93.2 ± 0.1**$^*$ |
|              | MADAR-9    | 75.5 ± 0.5                | 75.7 ± 0.2     | 76.8 ± 0.3                 | 74.5 ± 4.3     | 75.9 ± 0.5     | 77.5 ± 0.4     | 78.2 ± 0.3     | **81.9 ± 0.3** |
|              | MADAR-26   | 60.5 ± 0.2                | 62.0 ± 0.2     | 62.0 ± 0.1                 | 61.7 ± 0.1     | 60.2 ± 0.4     | 62.9 ± 0.1     | 61.5 ± 0.4     | **66.3 ± 0.1**$^*$ |
|              | NADI       | 17.6 ± 0.5                | 17.6 ± 0.5     | 22.6 ± 0.5                 | 22.6 ± 0.5     | 24.9 ± 0.6     | 25.9 ± 0.5     | **28.6 ± 0.8**$^*$ | 25.6 ± 0.6     |
| SA           | SemEval    | 51.3 ± 1.3                | 64.2 ± 0.7     | 65.4 ± 0.5                 | 64.4 ± 0.9     | 65.6 ± 0.3     | 67.1 ± 0.7     | 66.4 ± 0.3     | **69.2 ± 0.4**$^*$ |
|              | ASAD       | 59.8 ± 0.0                | 62.4 ± 0.0     | 41.3 ± 0.0                 | 66.9 ± 0.0     | **67.5 ± 0.0** | 65.8 ± 0.0     | 66.8 ± 0.0     | 66.7 ± 0.0     |
|              | AJGT       | 86.4 ± 0.3                | 92.4 ± 0.7     | 92.7 ± 0.3                 | 92.6 ± 0.4     | 93.6 ± 0.0     | 93.6 ± 0.3     | 93.7 ± 0.1     | **95.0 ± 0.3**$^*$ |
|              | ASTD       | 46.3 ± 1.4                | 55.7 ± 0.4     | 57.5 ± 2.3                 | 59.7 ± 0.1     | 61.9 ± 0.4     | 60.2 ± 0.2     | 61.0 ± 0.5     | **64.6 ± 0.1**$^*$ |
|              | LABR       | 81.1 ± 0.0                | 85.4 ± 0.0     | 85.9 ± 0.0                 | 85.9 ± 0.0     | 84.7 ± 0.0     | **86.3 ± 0.0** | 85.0 ± 0.0     | 84.9 ± 0.0     |
|              | ARSAS      | 73.2 ± 0.7                | 76.2 ± 0.6     | 76.8 ± 0.3                 | 76.1 ± 0.2     | 76.3 ± 0.2     | 77.1 ± 0.3     | 76.2 ± 0.2     | **77.9 ± 0.3**$^*$ |
| HSOD         | HateSpeech | 67.9 ± 1.4                | 73.7 ± 1.1     | 76.4 ± 1.2                 | 76.8 ± 1.4     | 80.0 ± 0.1     | 78.8 ± 0.6     | 80.0 ± 0.8     | **81.4 ± 0.5**$^*$ |
|              | Offense    | 85.3 ± 0.5                | 87.2 ± 0.5     | 90.5 ± 0.4                 | 90.5 ± 0.4     | 90.8 ± 0.2     | 89.2 ± 0.5     | 90.8 ± 0.3     | **91.3 ± 0.3**$^*$ |
|              | Adult      | 87.9 ± 0.1                | 87.2 ± 0.3     | 88.6 ± 0.1                 | 88.4 ± 0.6     | 88.1 ± 0.0     | 88.6 ± 0.3     | 88.3 ± 0.1     | **89.3 ± 0.3**$^*$ |




#### Accuracy Comparison of Various PLMs on Different Datasets

|            | Task        | mBERT         | LaBSE         | AraBERT       | ArBERT       | MdBERT        | CAMeL         | MARBERT       | AlcLaM        |
|------------|-------------|---------------|---------------|---------------|--------------|---------------|---------------|---------------|---------------|
| **DID**    | MADAR-2     | 97.3 ± 0.8    | 98.0 ± 0.1    | 98.1 ± 0.0    | 98.1 ± 0.0   | 98.0 ± 0.1    | 98.1 ± 0.1    | 97.2 ± 0.7    | **99.7 ± 0.0**|
|            | MADAR-6     | 91.3 ± 0.1    | 91.1 ± 0.2    | 91.6 ± 0.1    | 91.6 ± 0.2   | 91.6 ± 0.0    | 92.0 ± 0.1    | 92.2 ± 0.2    | **93.2 ± 0.1** * |
|            | MADAR-9     | 78.5 ± 0.5    | 79.1 ± 0.1    | 80.4 ± 0.2    | 77.7 ± 3.6   | 79.1 ± 0.5    | 80.5 ± 0.2    | 81.1 ± 0.3    | **83.4 ± 0.4**|
|            | MADAR-26    | 60.6 ± 0.2    | 61.9 ± 0.2    | 61.9 ± 0.1    | 61.7 ± 0.2   | 60.1 ± 0.3    | 62.9 ± 0.2    | 61.3 ± 0.3    | **66.1 ± 0.2** * |
|            | NADI        | 33.4 ± 0.6    | 33.4 ± 0.6    | 38.9 ± 1.7    | 38.9 ± 1.7   | 41.9 ± 1.9    | 42.7 ± 1.6    | **47.3 ± 0.1** * | 46.6 ± 1.0   |
| **SA**     | SemEval     | 53.4 ± 1.5    | 65.0 ± 0.6    | 66.1 ± 0.5    | 65.1 ± 0.8   | 66.1 ± 0.3    | 68.0 ± 0.3    | 66.9 ± 0.3    | **69.5 ± 0.3** * |
|            | ASAD        | 74.6 ± 0.0    | 75.2 ± 0.0    | 70.6 ± 0.0    | 78.4 ± 0.0   | 77.6 ± 0.0    | 77.0 ± 0.0    | 77.6 ± 0.0    | **79.5 ± 0.0**|
|            | AJGT        | 86.4 ± 0.3    | 92.4 ± 0.7    | 92.8 ± 0.3    | 92.6 ± 0.4   | 93.6 ± 0.0    | 93.6 ± 0.3    | 93.8 ± 0.1    | **95.0 ± 0.3** * |
|            | ASTD        | 46.7 ± 1.7    | 55.6 ± 0.6    | 57.7 ± 2.4    | 59.7 ± 0.3   | 62.0 ± 0.3    | 60.1 ± 0.2    | 61.0 ± 0.3    | **64.9 ± 0.1** * |
|            | LABR        | 90.4 ± 0.0    | 92.3 ± 0.0    | 92.8 ± 0.0    | 92.8 ± 0.0   | 91.9 ± 0.0    | **93.0 ± 0.0**| 92.6 ± 0.0    | 92.6 ± 0.0   |
|            | ARSAS       | 74.5 ± 0.8    | 77.2 ± 0.7    | 77.6 ± 0.3    | 77.0 ± 0.3   | 77.5 ± 0.3    | 78.0 ± 0.3    | 77.4 ± 0.4    | **78.6 ± 0.5** * |
| **HSOD**   | HateSpeech  | 75.2 ± 2.2    | 80.0 ± 0.7    | 80.5 ± 1.4    | 80.8 ± 1.9   | 84.3 ± 0.3    | 83.3 ± 0.6    | 84.4 ± 0.4    | **84.6 ± 0.7** * |
|            | Offense     | 91.7 ± 0.1    | 92.8 ± 0.4    | 94.5 ± 0.2    | 94.6 ± 0.4   | 94.6 ± 0.2    | 93.6 ± 0.2    | 94.8 ± 0.0    | **94.9 ± 0.1** * |
|            | Adult       | 95.0 ± 0.0    | 94.4 ± 0.2    | 95.2 ± 0.1    | 94.9 ± 0.4   | 95.1 ± 0.1    | 95.2 ± 0.2    | 95.1 ± 0.0    | **95.6 ± 0.0**|




 If you use the code,  please cite the paper: 
```
@article{murtadha2024alclam,
	author       = {Murtadha, Ahmed and Saghir, Alfasly and Wen, Bo and Jamaal, Qasem and  Mohammed, Ahmed and Liu, Yunfeng},
	title        = {AlcLaM: Arabic Dialectal Language Model},
	journal      = {Arabic NLP 2024},
	year         = {2024},
	url          = {https://arxiv.org/abs/2407.13097},
	eprinttype    = {arXiv},
	eprint       = {2407.13097}
}
```
