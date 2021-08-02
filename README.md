# Text2SQL

Gathering information for **Semantic Parsing** problem, especially for `Text to SQL`.

# Semantic Parsing

> Source: [TRANX: A Transition-based Neural Abstract Syntax Parser for Semantic Parsing and Code Generation](https://arxiv.org/abs/1810.02720)
>
> Source2: [paperswithcode.com/task/semantic-parsing](https://paperswithcode.com/task/semantic-parsing)

**Semantic Parsing** is the task of transducing natural language utterances into formal meaning representations. The target meaning representations can be defined according to a wide variety of formalisms. This include linguistically-motivated semantic representations that are designed to capture the meaning of any sentence such as λ-calculus or the abstract meaning representations. Alternatively, for more task-driven approaches to Semantic Parsing, it is common for meaning representations to represent executable programs such as SQL queries, robotic commands, smart phone instructions, and even general-purpose programming languages like Python and Java.

# How to Start

## Prerequisite

```
sh install_requirements.sh
```

Download weight

```
sh get_weights.sh
```

## Help

```
python main.py --help
```

## Run

```
sh run_train.sh
```

```
tensorboard --logdir ./logs/ --bind_all --port 6006
```

# TODO

- [x] 06.10 - Train Model
- [x] Extend the dataset to all companies
- [ ] Improve Model
- [ ] Code Refactoring
- [ ] Get the original purpose: resolve the ambigious parts 
- [ ] Build Application with Streamlit