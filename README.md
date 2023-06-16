# Python Template for Project
[![Python 3.10](https://img.shields.io/badge/Python-3.10-green.svg)](https://www.python.org/downloads/release/python-390/)
[![Poetry](https://img.shields.io/badge/Poetry-1.5.1-blue.svg)](https://python-poetry.org/)

This repository is a my python template to help you get started quickly on a new project.

## How to Use This Template for a New Project

Follow these steps to start a new project based on this template:

1. **Create a New Repository**: First, create a new repository on GitHub. This will be the repository for your new project.

2. **Clone the template to a New Directory**: On your local machine, clone this template repository into a new directory that will become your new project. Then remove the `.git` directory to completely decouple it from the template. Use these commands:

    ```bash
    git clone https://github.com/C-Naoki/my-python-templete.git <new-project>
    cd <new-project>
    rm -rf .git
    ```

3. **Initialize a New Repository**: Then, initialize a new repository in the new directory:

    ```bash
    git init
    ```

4. **Set the Remote of the New Repository**: Set your new GitHub repository as the remote for your new project:

    ```bash
    git remote add origin <url-of-your-new-repository>
    ```

5. **Push to the New Repository**: Stage all files, commit them, and push them to your new repository:

    ```bash
    git add .
    git commit -m ":tada: initial commit"
    git push -u origin master
    ```

By following these steps, you can start a new project based on this starter set.
