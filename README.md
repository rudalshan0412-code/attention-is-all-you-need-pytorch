# attention-is-all-you-need-pytorch
PyTorch implementation of the Transformer model from scratch based on the paper "Attention Is All You Need", including custom tokenizer, dataset pipeline, and training loop.

# Transformer From Scratch with PyTorch

> PyTorch로 직접 구현한 Encoder-Decoder Transformer

## 프로젝트 소개

이 프로젝트는 논문 **Attention Is All You Need**에서 제안한 Transformer 구조를 이해하기 위해 PyTorch로 주요 구성요소를 직접 구현한 개인 학습 프로젝트입니다.

`nn.Transformer`, `nn.MultiheadAttention`과 같은 완성형 API를 사용하지 않고 Attention부터 Encoder, Decoder, 전체 Transformer까지 직접 구현했습니다.

이후 Tokenizer, Vocabulary, Dataset, Training, Inference까지 연결하여 작은 English → Korean 번역 데이터로 전체 학습 파이프라인을 확인했습니다.

마지막에는 직접 구현한 Transformer와 PyTorch의 `nn.Transformer`를 비교하여 각 구성요소가 공식 구현과 어떻게 대응되는지도 확인했습니다.

---

## 프로젝트 목적

Transformer 논문의 구조를 단순히 이론으로 이해하는 것이 아니라 실제 코드로 구현하면서 다음 내용을 확인하는 것이 목표였습니다.

- Scaled Dot-Product Attention의 계산 과정
- Q, K, V의 역할
- Multi-Head Attention의 Head 분리와 결합
- Positional Encoding
- Encoder / Decoder 구조
- Residual Connection과 Layer Normalization
- Padding Mask와 Causal Mask
- Decoder의 Cross-Attention
- Target Shift를 이용한 학습
- Autoregressive Inference
- Custom Transformer와 `nn.Transformer`의 구조적 대응 관계

---

## 구현 범위

프로젝트는 다음 순서로 진행했습니다.

```text
01) Scaled Dot-Product Attention
02) Multi-Head Attention
03) Positional Encoding
04) Position-wise Feed Forward Network
05) Encoder Layer
06) Encoder
07) Mask
08) Decoder Layer
09) Decoder
10) Transformer
11) Tokenizer / Vocabulary / Dataset
12) Training
13) Inference
14) English → Korean Toy Dataset 학습
15) PyTorch nn.Transformer 비교
16) 최종 코드 및 프로젝트 정리
```

---

## 프로젝트 구조

```text
attention_is_all_you_need/
├── README.md
└── src/
    ├── attention.py
    ├── multi_head_attention.py
    ├── positional_encoding.py
    ├── feed_forward.py
    ├── encoder_layer.py
    ├── encoder.py
    ├── mask.py
    ├── decoder_layer.py
    ├── decoder.py
    ├── transformer.py
    ├── tokenizer.py
    ├── vocabulary.py
    └── dataset.py
```

---

# Transformer 전체 구조

```text
Source Token IDs
        │
        ▼
Source Embedding
        │
        ▼
× sqrt(d_model)
        │
        ▼
Positional Encoding
        │
        ▼
     Encoder
        │
        │ encoder_output
        ▼
     Decoder
        ▲
        │
Positional Encoding
        ▲
        │
× sqrt(d_model)
        ▲
        │
Target Embedding
        ▲
        │
Target Token IDs

Decoder Output
        │
        ▼
Linear Projection
        │
        ▼
Vocabulary Logits
```

Source와 Target은 각각 Embedding과 Positional Encoding을 거칩니다.

Encoder의 출력은 Decoder의 Cross-Attention에서 Key와 Value로 사용됩니다.

Decoder 출력은 마지막 Linear Layer를 거쳐 Target Vocabulary 크기의 logits으로 변환됩니다.

---

# 직접 구현한 구성요소

## 1. Scaled Dot-Product Attention

Attention의 핵심 연산을 직접 구현했습니다.

```text
QK^T
 ↓
÷ sqrt(d_k)
 ↓
Mask
 ↓
Softmax
 ↓
× V
```

계산식은 다음과 같습니다.

```text
Attention(Q, K, V)
=
softmax(QK^T / sqrt(d_k))V
```

Mask는 Softmax 이전의 Attention Score에 적용됩니다.

이를 통해 어떤 위치의 token을 Attention 계산에서 제외할 수 있습니다.

---

## 2. Multi-Head Attention

Multi-Head Attention에서는 입력으로부터 Q, K, V를 각각 Linear Projection한 뒤 여러 Head로 분리합니다.

```text
Q / K / V
   │
   ▼
Linear Projection
   │
   ▼
Head Split
   │
   ▼
Scaled Dot-Product Attention
   │
   ▼
Head Concatenation
   │
   ▼
Final Linear
```

각 Head는 서로 다른 Representation 공간에서 Attention을 수행합니다.

`nn.MultiheadAttention`은 사용하지 않았습니다.

---

## 3. Positional Encoding

Transformer에는 RNN과 같은 순차 구조가 없기 때문에 token의 위치 정보를 추가하기 위해 Sinusoidal Positional Encoding을 직접 구현했습니다.

Embedding에는 다음 scaling을 적용했습니다.

```python
embedding * math.sqrt(d_model)
```

그 후 Positional Encoding을 더합니다.

```text
Embedding
    +
Positional Encoding
```

---

## 4. Position-wise Feed Forward Network

Feed Forward Network는 각 token 위치에 독립적으로 적용됩니다.

```text
d_model
   │
   ▼
Linear
   │
   ▼
d_ff
   │
   ▼
ReLU
   │
   ▼
Linear
   │
   ▼
d_model
```

현재 구현에서는 ReLU를 사용했습니다.

---

# Encoder

## Encoder Layer

Encoder Layer는 다음 구조입니다.

```text
Input
  │
  ▼
Multi-Head Self-Attention
  │
  ▼
Dropout
  │
  ▼
Residual Add
  │
  ▼
LayerNorm
  │
  ▼
Feed Forward
  │
  ▼
Dropout
  │
  ▼
Residual Add
  │
  ▼
LayerNorm
```

Residual Connection 이후 Layer Normalization을 수행하는 **Post-LayerNorm** 구조입니다.

---

## Encoder Stack

여러 개의 Encoder Layer를 순서대로 통과시켜 Encoder를 구성했습니다.

```text
Input
 ↓
Encoder Layer 1
 ↓
Encoder Layer 2
 ↓
...
 ↓
Encoder Output
```

각 Encoder Layer의 Self-Attention Weight도 확인할 수 있도록 반환합니다.

---

# Mask

프로젝트의 Custom Mask 규칙은 다음과 같습니다.

```text
True  = Attention 허용
False = Attention 차단
```

## Padding Mask

Padding Token을 Attention 대상에서 제외합니다.

Shape:

```text
(B, 1, 1, S)
```

## Causal Mask

Decoder가 현재 위치보다 미래의 token을 참조하지 못하도록 합니다.

Shape:

```text
(1, 1, T, T)
```

## Target Mask

Target Padding Mask와 Causal Mask를 결합합니다.

```text
target_mask
=
target_padding_mask
&
causal_mask
```

최종 Shape:

```text
(B, 1, T, T)
```

---

# Decoder

## Decoder Layer

Decoder Layer는 세 개의 주요 Sublayer로 구성됩니다.

```text
Masked Self-Attention
        │
        ▼
Dropout + Residual + LayerNorm
        │
        ▼
Cross-Attention
        │
        ▼
Dropout + Residual + LayerNorm
        │
        ▼
Feed Forward
        │
        ▼
Dropout + Residual + LayerNorm
```

Self-Attention에서는 Causal Mask를 적용하여 미래 token을 볼 수 없도록 합니다.

---

## Cross-Attention

Cross-Attention에서는 다음 값을 사용합니다.

```text
Q = Decoder Hidden State
K = Encoder Output
V = Encoder Output
```

이를 통해 Decoder가 Source Sentence의 정보를 참조하면서 다음 token을 예측합니다.

---

## Decoder Stack

여러 개의 Decoder Layer를 순서대로 연결했습니다.

각 Decoder Layer에서는 다음 두 종류의 Attention Weight를 확인할 수 있습니다.

```text
Decoder Self-Attention Weight
Decoder Cross-Attention Weight
```

---

# 전체 Transformer

최종 Transformer는 다음 요소를 연결하여 구성했습니다.

```text
Source Embedding
Target Embedding
Positional Encoding
Encoder
Decoder
Final Linear
```

Constructor의 기본 형태는 다음과 같습니다.

```python
Transformer(
    source_vocab_size,
    target_vocab_size,
    d_model,
    num_heads,
    d_ff,
    num_encoder_layers,
    num_decoder_layers,
    max_len,
    dropout=0.1,
)
```

Forward는 Source와 Target Token ID 및 Mask를 입력으로 받습니다.

반환값은 다음과 같습니다.

```text
logits

encoder_attention_weights_list

decoder_self_attention_weights_list

decoder_cross_attention_weights_list
```

Attention Weight를 직접 반환하도록 구현하여 Transformer 내부 동작을 확인할 수 있도록 했습니다.

---

# Tokenizer

프로젝트에서는 별도의 pretrained tokenizer를 사용하지 않고 간단한 tokenizer를 직접 사용했습니다.

예를 들어:

```text
나는 사과를 좋아한다.
```

는 대략 다음과 같이 처리됩니다.

```python
[
    "나는",
    "사과를",
    "좋아한다",
    ".",
]
```

한국어 형태소 분석기나 Subword Tokenizer는 사용하지 않았습니다.

---

# Vocabulary

Source와 Target Vocabulary를 각각 별도로 생성했습니다.

프로젝트 전체에서 사용하는 Special Token은 다음과 같습니다.

```text
<PAD>
<UNK>
<BOS>
<EOS>
```

Index는 다음과 같습니다.

```python
PAD_IDX = 0
UNK_IDX = 1
BOS_IDX = 2
EOS_IDX = 3
```

Train Vocabulary에 존재하지 않는 token은 `<UNK>`로 변환됩니다.

---

# Dataset

`TranslationDataset`을 직접 구현하여 Source Sentence와 Target Sentence를 Token ID 형태로 변환했습니다.

각 문장은 내부적으로 다음 구조를 가집니다.

```text
<BOS> tokens... <EOS>
```

Dataset의 반환값은 다음과 같습니다.

```text
source_ids
target_ids
```

두 값 모두 `torch.long` Tensor입니다.

Batch 생성 시 `collate_fn`을 이용해 문장 길이를 맞추기 위한 Padding도 적용했습니다.

---

# Training

Training에서는 Teacher Forcing 형태로 Target Sequence를 한 칸 Shift하여 사용했습니다.

## Target Shift

예를 들어 실제 Target Sequence가 다음과 같다면:

```text
<BOS> 나는 행복하다 . <EOS>
```

Decoder Input은:

```text
<BOS> 나는 행복하다 .
```

Label은:

```text
나는 행복하다 . <EOS>
```

가 됩니다.

코드에서는 다음과 같이 처리했습니다.

```python
decoder_input = target_batch[:, :-1]
labels = target_batch[:, 1:]
```

Decoder는 이전 token들을 입력받아 다음 token을 예측합니다.

---

# Loss

Loss Function은 다음을 사용했습니다.

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=PAD_IDX,
)
```

Padding Token은 Loss 계산에서 제외합니다.

Transformer는 마지막 Linear Layer의 raw logits를 반환하며, 모델 내부에서는 Softmax를 별도로 적용하지 않습니다.

---

# Optimizer

대표 Training 설정에서는 Adam Optimizer를 사용했습니다.

```python
optimizer = torch.optim.Adam(
    transformer.parameters(),
    lr=1e-3,
)
```

---

# Inference

Inference에서는 Greedy Decoding을 직접 구현했습니다.

```text
<BOS>
 ↓
Transformer
 ↓
마지막 위치의 Logits
 ↓
argmax
 ↓
Token 추가
 ↓
Transformer 재실행
 ↓
EOS가 나올 때까지 반복
```

매 단계에서 가장 확률이 높은 token 하나를 선택합니다.

Beam Search, Top-k, Top-p Sampling은 사용하지 않았습니다.

---

# English → Korean Toy Dataset

전체 Transformer 학습 파이프라인을 확인하기 위해 작은 English → Korean 병렬 데이터를 사용했습니다.

전체 데이터:

```text
50 sentence pairs
```

구성:

```text
Train      40
Validation 10
```

예시:

```text
I am happy.
→ 나는 행복하다.

You are happy.
→ 너는 행복하다.

I like apples.
→ 나는 사과를 좋아한다.

I read a book.
→ 나는 책을 읽는다.
```

이 데이터는 Transformer의 전체 학습 및 추론 과정이 정상적으로 연결되는지 확인하기 위한 **Toy Dataset**입니다.

실제 번역 성능을 평가하기 위한 데이터셋은 아닙니다.

Vocabulary는 Train Dataset을 기준으로 생성했기 때문에 Validation Dataset에 처음 등장하는 token은 `<UNK>`로 처리될 수 있습니다.

---

# 주요 Hyperparameter

대표적으로 다음 설정을 사용했습니다.

| Parameter | Value |
|---|---:|
| `d_model` | 64 |
| `num_heads` | 4 |
| `d_ff` | 128 |
| Encoder Layers | 2 |
| Decoder Layers | 2 |
| Dropout | 0.1 |
| Batch Size | 8 |
| Optimizer | Adam |
| Learning Rate | 1e-3 |
| Epochs | 약 200 |
| Loss | CrossEntropyLoss |
| Padding Ignore | `PAD_IDX` |

---

# 주요 Tensor Shape

| Tensor | Shape |
|---|---|
| Source Batch | `(B, S)` |
| Target Batch | `(B, T)` |
| Source Embedding | `(B, S, d_model)` |
| Target Embedding | `(B, T, d_model)` |
| Source Mask | `(B, 1, 1, S)` |
| Target Padding Mask | `(B, 1, 1, T)` |
| Causal Mask | `(1, 1, T, T)` |
| Target Mask | `(B, 1, T, T)` |
| Encoder Output | `(B, S, d_model)` |
| Decoder Output | `(B, T, d_model)` |
| Logits | `(B, T, target_vocab_size)` |

여기서:

```text
B = Batch Size
S = Source Sequence Length
T = Target Sequence Length
```

입니다.

---

# Custom Transformer와 PyTorch nn.Transformer 비교

프로젝트 마지막 단계에서는 직접 구현한 Transformer와 PyTorch 공식 `nn.Transformer`를 비교했습니다.

## Custom Transformer

```text
Embedding
   ↓
Positional Encoding
   ↓
Custom Encoder
   ↓
Custom Decoder
   ↓
Linear
```

## PyTorch Transformer

```text
Embedding
   ↓
Positional Encoding
   ↓
nn.Transformer
   ↓
Linear
```

`nn.Transformer`만 호출한다고 전체 번역 모델이 완성되는 것은 아닙니다.

Embedding, Positional Encoding, Final Linear Layer, Tokenizer, Dataset, Training, Inference 등은 별도로 구성해야 합니다.

---

## Custom ↔ PyTorch 구조 대응

| Custom Implementation | PyTorch |
|---|---|
| `ScaledDotProductAttention` | `nn.MultiheadAttention` 내부 Attention |
| `MultiHeadAttention` | `nn.MultiheadAttention` |
| `EncoderLayer` | `nn.TransformerEncoderLayer` |
| `Encoder` | `nn.TransformerEncoder` |
| `DecoderLayer` | `nn.TransformerDecoderLayer` |
| `Decoder` | `nn.TransformerDecoder` |
| Encoder + Decoder | `nn.Transformer` |

---

# nn.Transformer 비교 설정

직접 구현한 구조와 최대한 비슷한 조건으로 비교했습니다.

```python
nn.Transformer(
    d_model=64,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    activation="relu",
    batch_first=True,
    norm_first=False,
)
```

`activation="relu"`는 Custom Feed Forward Network가 ReLU를 사용하기 때문에 설정했습니다.

`norm_first=False`는 Custom Encoder와 Decoder가 Post-LayerNorm 구조이기 때문에 사용했습니다.

`batch_first=True`는 프로젝트 전체에서 Tensor Shape을 다음 형태로 사용했기 때문입니다.

```text
(B, S, d_model)
```

---

# Custom Mask와 PyTorch Mask 차이

Custom 구현과 PyTorch의 Boolean Mask는 의미가 반대라는 점을 확인했습니다.

## Custom

```text
True  = Attention 허용
False = Attention 차단
```

## PyTorch

```text
True  = Attention 차단
False = Attention 허용
```

PyTorch에서는 Padding Mask와 Causal Mask를 별도로 전달합니다.

```text
src_key_padding_mask
(B, S)

tgt_key_padding_mask
(B, T)

memory_key_padding_mask
(B, S)

tgt_mask
(T, T)
```

반면 Custom Transformer에서는 Padding Mask와 Causal Mask를 직접 결합하여 사용했습니다.

---

# Attention Weight

Custom Transformer에서는 Attention Weight를 직접 반환합니다.

```text
Encoder Self-Attention

Decoder Self-Attention

Decoder Cross-Attention
```

따라서 각 Layer에서 Attention이 어떻게 형성되는지 직접 확인할 수 있습니다.

PyTorch의 `nn.Transformer`는 기본 Forward 결과에서 동일한 형태로 Attention Weight를 반환하지 않습니다.

이 프로젝트에서는 PyTorch 내부에 별도의 Hook을 추가하지 않았습니다.

---

# 실행 환경

프로젝트는 Google Colab 환경에서 진행했습니다.

주요 사용 도구:

```text
Python
PyTorch
Google Colab
Google Drive
Matplotlib
```

프로젝트 파일은 Google Drive의 다음 경로를 기준으로 사용했습니다.

```text
/content/drive/MyDrive/attention_is_all_you_need
```

Colab에서는 먼저 Drive를 연결합니다.

```python
from google.colab import drive

drive.mount('/content/drive')
```

프로젝트 경로는 다음과 같이 추가할 수 있습니다.

```python
from pathlib import Path
import sys

PROJECT_ROOT = Path(
    "/content/drive/MyDrive/attention_is_all_you_need"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )
```

---

# 프로젝트를 통해 확인한 내용

이 프로젝트를 진행하면서 Transformer 내부에서 일어나는 연산을 단계별로 확인할 수 있었습니다.

특히 다음 내용을 코드 수준에서 이해하는 것을 목표로 했습니다.

- Attention Score가 `QK^T`로 계산되는 과정
- `sqrt(d_k)` Scaling이 적용되는 위치
- Mask가 Softmax 이전에 적용되는 이유
- Multi-Head Attention의 Tensor Shape 변화
- Encoder Self-Attention의 동작
- Decoder Masked Self-Attention의 동작
- Encoder와 Decoder 사이의 Cross-Attention
- Residual Connection과 LayerNorm의 위치
- Positional Encoding이 Embedding에 추가되는 과정
- Target Shift와 다음 Token Prediction의 관계
- Autoregressive Inference 과정
- Custom 구현과 PyTorch 공식 구현의 구조적 대응 관계

논문의 구조를 직접 구현하면서 각 Layer가 어떤 입력과 출력을 가지는지 Tensor Shape을 중심으로 확인할 수 있었습니다.

---

# 직접 구현하면서 확인한 장점

완성된 API를 사용하는 것보다 내부 동작을 직접 확인하기 쉬웠습니다.

```text
Attention 계산 과정 확인

Q / K / V 역할 확인

Mask 적용 과정 확인

Tensor Shape 변화 추적

Encoder / Decoder 내부 구조 확인

Residual Connection과 LayerNorm 확인

Cross-Attention 연결 확인

Attention Weight 직접 확인
```

특히 오류가 발생했을 때 어느 단계의 Tensor Shape이나 Mask가 잘못되었는지 직접 추적하면서 Transformer의 구조를 더 구체적으로 이해할 수 있었습니다.

---

# 한계점

이 프로젝트는 Transformer 구조 학습을 목적으로 했기 때문에 실제 번역 시스템과 비교하면 여러 제한이 있습니다.

- 작은 Toy Dataset 사용
- 단순 Tokenizer 사용
- 한국어 형태소 분석기 미사용
- BPE / SentencePiece 등 Subword Tokenizer 미사용
- BLEU 등 정량적인 번역 평가 미사용
- Greedy Decoding만 사용
- Beam Search 미사용
- Learning Rate Warmup 미사용
- Noam Scheduler 미사용
- 대규모 번역 Dataset 미사용
- 실제 Production 번역 모델을 목표로 하지 않음

따라서 번역 품질보다는 Transformer 전체 구조가 학습과 추론까지 정상적으로 연결되는지를 확인하는 데 중점을 두었습니다.

---

# 향후 개선 가능성

프로젝트를 확장한다면 다음 기능을 추가할 수 있습니다.

```text
BPE 또는 SentencePiece 적용

더 큰 실제 번역 Dataset 사용

Learning Rate Warmup 적용

Noam Scheduler 적용

Beam Search 구현

BLEU 기반 번역 성능 평가

Model Checkpoint 저장

Attention Visualization

GPU 기반 대규모 학습
```

현재 프로젝트에는 위 기능들이 구현되어 있지 않습니다.

---

# 최종 정리

이 프로젝트에서는 `Attention Is All You Need`의 Encoder-Decoder Transformer를 PyTorch로 직접 구현했습니다.

단순히 `nn.Transformer`를 사용하는 대신 다음 구성요소를 순서대로 직접 구현하고 연결했습니다.

```text
Scaled Dot-Product Attention
Multi-Head Attention
Positional Encoding
Feed Forward Network
Encoder Layer
Encoder
Mask
Decoder Layer
Decoder
Transformer
Tokenizer
Vocabulary
Dataset
Training
Greedy Decoding
```

이후 작은 English → Korean 데이터셋을 이용해 실제 Training과 Inference를 진행했습니다.

마지막으로 PyTorch의 `nn.Transformer`와 직접 구현한 Transformer를 비교하면서 각 구성요소가 공식 API에서 어떻게 대응되는지도 확인했습니다.

프로젝트의 목표는 높은 번역 성능을 만드는 것이 아니라 Transformer 내부 구조와 데이터 흐름을 직접 구현하면서 이해하는 것이었습니다.
````
