# Python Template for Research Projects

<p align="right">
  <img src="./docs/assets/logo.png" alt="Logo"/>
  <span style="color: gray;">Illustration by ChatGPT</span>
</p>

<div align="left">

[![Python 3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-3915/)
[![Pyenv](https://img.shields.io/badge/Pyenv-2.6.7-yellow.svg)](https://github.com/pyenv/pyenv#installation)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/%F0%9F%93%9A%20docs-Zenn-3ea8ff.svg)](https://zenn.dev/naoki0103/articles/my-python-template)
<img src="https://img.shields.io/static/v1?label=%F0%9F%8C%9F&message=If%20Useful&style=style=flat&color=BC4E99" alt="Star Badge"/>\
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
</div>

This repository is a comprehensive Python template designed to accelerate your research and development projects. It provides a well-structured foundation with modern Python tooling, including uv for dependency management and Pyenv for version management. The template comes pre-configured with essential data science libraries, automated environment setup, and practical scripts to help you focus on your core work rather than project setup.

## 📘 Usage
Just click [`THIS BUTTON`](https://github.com/C-Naoki/my-python-template/generate) or top-right green button to create a copy of this repository on your GitHub account.

### Main Commands
I introduce some useful commands below.

-  Initialize the environment
    ```bash
    make install
    ```
    - ⚠️ **You need to execute this command before running other commands.**
    - This command will create a virtual environment using `pyenv` and install the dependencies using `uv`. See the [`Makefile`](Makefile) for more details.
    - Some key packages (`numpy`, `pandas`, `scikit-learn`, etc.) are automatically installed by this command. See the [`pyproject.toml`](pyproject.toml) for more details and adjust them as needed.

-  Run the quick demo
    ```bash
    make run
    ```
    - You need to modify the [`bin/demo.sh`](bin/demo.sh) file to specify the model and input data you want to use. (By default it only displays the configurations.)

- Check whether cuda is available
    ```bash
    make cuda_check
    ```

- Create a requirements.txt file
    ```bash
    make freeze
    ```
    - This is useful when you want to use your project on an environment that does not support `uv`.

## 🌳 Directory Structure
The main directories and files are as follows:
- `bin/`: Contains useful scripts for running experiments and managing the project.
- `docs/`: Documentation files, including Sphinx configuration and report templates.
- `notebooks/`: Jupyter notebook templates for exploratory data analysis and prototyping.
- `out/`: Output directory for saving results, models, and logs.
- `src/`: Main source code directory.
- `tests/`: Unit tests and test cases for the project.

<details>
<summary>&thinsp;See details (Last updated on Feb 6, 2026)</summary>

```bash
.
├── bin
│   ├── add_pth.sh
│   ├── demo.sh
│   └── run_wrapper.sh
├── data
├── docs
│   ├── assets
│   │   └── logo.png
│   ├── reports
│   │   ├── memo.pdf
│   │   └── memo.tex
│   └── source
│       └── conf.py
├── LICENSE
├── Makefile
├── notebooks
│   ├── font
│   │   ├── Times New Roman Bold Italic.ttf
│   │   ├── Times New Roman Bold.ttf
│   │   ├── Times New Roman Italic.ttf
│   │   └── Times New Roman.ttf
│   └── template.ipynb
├── out
│   └── .gitkeep
├── pyproject.toml
├── README.md
├── src
│   ├── __init__.py
│   ├── config
│   │   ├── model
│   │   │   └── proposal.yaml
│   │   └── settings.yaml
│   ├── main.py
│   ├── models
│   │   ├── __init__.py
│   │   └── ours
│   │       ├── __init__.py
│   │       └── module
│   │           └── __init__.py
│   └── utils
│       ├── __init__.py
│       ├── io_helper.py
│       ├── metrics.py
│       └── preprocessor.py
├── tests
│   ├── __init__.py
│   └── test_cuda.py
└── uv.lock
```
</details>

## 🧩 Extensions
I have prepared practical github workflows to enhance the development process.

**clean_gitkeep:** This workflow can automatically delete unnecessary `.gitkeep` files from the repository. If you want to use it, you need to do the following:

1. **Open the GitHub Repository Settings**: Navigate to the GitHub repository page and click on the "⚙️ Settings" tab at the top of the page.
2. **Navigate to the Actions Menu**: Look for the Code and automation section in the left sidebar.
3. **Select General**: Within the expanded Actions menu, click on the "General" option.
4. **Enable Read and write permissions**: Check the box of `Read and write permissions` within Workflow permissions section, at the bottom of the page.

More workflows are in progress and will be available soon. Stay tuned!

## 🙋‍♂️ Support
💙 If you like this dotfiles, give it a ⭐ and share it with friends!

## ✉️ Contact
💥 If you have any questions or encounter issues, feel free to open an [issue](https://github.com/C-Naoki/my-python-template/issues). I appreciate your feedback and look forward to hearing from you!

## 📄 License
Licensed under the APLv2. See the [LICENSE](https://github.com/C-Naoki/my-python-template/blob/main/LICENSE) file for details.
