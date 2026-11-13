# Character-level transformer on Shakespeare

A character-level, GPT-style transformer language model trained on the complete works of Shakespeare.
It reads text one character at a time and learns to predict the next one.

Built following Andrej Karpathy's ["Let's build GPT: from scratch, in code, spelled out"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
tutorial, then trained on Shakespeare's complete works from Project Gutenberg instead of the tutorial's dataset.

| File | Contents |
|---|---|
| `doc.py` | builds the model (multi-head self-attention, feed-forward layers, transformer blocks), trains it and saves the weights to `model.pth` |
| `generate.py` | loads `model.pth`, prints 1000 generated characters, then asks for a prompt to continue (type `exit` to quit) |
| `model.pth` | the trained weights |
| `train_log.txt` | the train/val loss printed during training |

## Data

`shakespeare.txt` is [Project Gutenberg ebook 100](https://www.gutenberg.org/ebooks/100), *The Complete Works of
William Shakespeare*, with the Gutenberg header and licence removed (only the text between the `*** START` and
`*** END` lines is kept). It is not committed. To recreate it from this folder:

```bash
curl -o pg100.txt https://www.gutenberg.org/cache/epub/100/pg100.txt
python3 -c "t=open('pg100.txt',encoding='utf-8').read(); s=t.index('\n',t.index('*** START')); open('shakespeare.txt','w',encoding='utf-8').write(t[s:t.index('*** END')].strip('\n')+'\n')"
```

Both scripts build the character vocabulary from this file, so `generate.py` needs the same file the weights were
trained on. The copy used here has 5,359,343 characters and a vocabulary of 100 characters
(SHA-256 `bfb5a5764506115cbae33bfdf66014697ef69d06339f001db209d3f2a56c83fc`). If Gutenberg changes the file and the
vocabulary size is no longer 100, `model.pth` will not load.

## Running

Install the requirements from the [machine learning README](../README.md#setup) (this project needs `torch`), then
from this folder:

```bash
python3 generate.py
```

Training is optional because `model.pth` is included. To retrain (this overwrites `model.pth`):

```bash
python3 doc.py | tee train_log.txt
```

Both scripts use Apple's `mps` GPU if available, otherwise the CPU.

## Training

| Setting | Value |
|---|---|
| parameters | 2.73 million |
| layers / heads / embedding size | 6 / 6 / 192 |
| context length | 128 characters |
| batch size | 32 |
| iterations | 10,000 (AdamW, learning rate 1e-3, dropout 0.2) |
| train / val split | first 70% / last 30% of the text |
| time | about 24 minutes on an Apple Silicon Mac (`mps`) |

Loss is averaged over 200 batches every 500 steps:

| Step | Train loss | Val loss |
|---|---|---|
| 0 | 4.6707 | 4.6723 |
| 1000 | 1.7074 | 1.7901 |
| 2000 | 1.5196 | 1.6570 |
| 3000 | 1.4321 | 1.5867 |
| 4000 | 1.3771 | 1.5459 |
| 5000 | 1.3482 | 1.5210 |
| 6000 | 1.3223 | 1.5057 |
| 7000 | 1.3072 | 1.4993 |
| 8000 | 1.2905 | 1.4831 |
| 9000 | 1.2795 | 1.4703 |
| 9500 | 1.2801 | 1.4612 |

Val loss was still decreasing at the end of the 10,000 steps (the last evaluation is at step 9500), so longer
training would likely improve it. The full log is in `train_log.txt`.

The model was first trained on *Hamlet* alone, but it overfit: val loss stopped improving at about 1.68 while train
loss kept falling, so it was retrained on the complete works.

## Sample output

From `echo exit | python3 generate.py` (an excerpt of the 1000 characters it prints). The model generates one
character at a time, so it imitates the style and layout of the plays rather than producing real English:

```
ANTONY.
Pray you, my sword! Well, sir, sit.

DROMIO OF EPHESUS.
I have dead. We are no man known should not.

DESDEMONA.
Nay, ha, I humbly, what would have predecious appelt
Of the voice of their woes, which may worray be.

LONGAVILLE.
We pray not.
What four majesty means? And for our Grace?
Let them tread not me this strength, some Rosencrantz of his lands,
And what dedicted me wear them in good others,
Yet this may lose taxe me words she speak. Love it that,
Drawing my heels. Your majesty is importain, or incursion
```

## Known issues

- Typing a character that is not in `shakespeare.txt` at the prompt prints an error and asks again.
- The model class is still called `BigramLanguageModel` (the name from the start of the tutorial), although it is a
  full transformer.
