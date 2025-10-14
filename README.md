❗❗ Replace this readme with a description of your project as soon as possible ❗❗

# python-template-maxi
Hi! This is an advanced template for Python projects. 
The purpose of this template is to be a complete template with simple checks in place to maintain a minimum level of code quality.
However, it is not intended to teach you how to write good code. 
That part is up to you :)


### Python templates

|        | link | Focus                             |
|:------:|:----:|-----------------------------------|
| 🐍     | [Mini](https://github.com/adraismawur/python-template-mini) | File and code structure           |
| 🐍🐍   | [Midi](https://github.com/adraismawur/python-template-midi) | Environments, Requirements and packaging        |
| 🐍🐍🐍 | [Maxi](https://github.com/adraismawur/python-template-maxi) | Testing, Automatic formatting and checks   |


### Table of contents:

- [Quick start](#quick-start)
- [Why should I use this?](#why-should-i-use-this)
- [What is included in this repository?](#what-is-included-in-this-repository)
- [Guide to elements introduced here](#guide-to-elements-introduced-here)

## Quick start:

### Before you begin

❗ Compared to the previous templates, this repository introduces more concepts.  
❗ Some of the elements introduced here can feel annoying to work with until you get used to them.  

### Prerequisites

❗ This template assumes you are familiar with the concepts explained in the [Mini template](https://github.com/adraismawur/python-template-mini) and the [Midi template](https://github.com/adraismawur/python-template-midi)  

For this repository you must install the following tools, which requires you to be familiar with pip:
- [Black (https://github.com/psf/black)](https://github.com/psf/black)
- [Flake8 (https://flake8.pycqa.org/en/latest/)](https://flake8.pycqa.org/en/latest/)

### Set up the repository (local only)

❕Choose this if you are just starting out, or if you do not want to create a page on Github for your code.

1. On this GitHub page, click the **Code** dropdown button in the top-right
2. Click Download ZIP
3. Extract the files somewhere in a new directory

### Set up a remote repository (on GitHub)

❕Choose this if you want to ensure your code is always saved online, or if you want to share your code.

1. Create a new repository of your own by pressing the green button in the top right named "use this template" -> Create a new repository.
   Or [click here](https://github.com/new?template_name=python-template-maxi&template_owner=adraismawur)
2. Give your repository a nice name and description
3. Choose whether you want the repository to be public (anyone can see your code), or private.
4. Press "Create repository"

You will be taken to your own github repository page after a few seconds.
From here, you can make edits directly to your files, but it is more practical to download your repository to your local device. (cloning)

1. On your github repository page, click the green button with the text "Code" in the top-right
2. Copy the URL that starts with ```git@github.com:```
3. On your device, open a terminal and navigate to a folder where you want your project to be stored
4. Use the command ```git clone [git url from step 2]```

Your repository will now appear in the folder you navigated to in step 3

### Setting up pre-commit

1. Open a shell in your repository directory
2. type pre-commit.

If everything worked correctly, this command should complete with no warnings or errors.

### Start coding!

Generally, it is recommended to have the starting point of your code in a file with the name `main.py` inside a folder with the name `your_project_name`.
In there, the `main()` function calls all other functions from separate folders (called modules).

Things to consider while you do so:
- Commit early, commit often
- Aim for ~300 lines of code max for a python file
- You do not have to use the template/ subfolder if you have only a few functions/files
- Have fun!

### Running your unit tests

All that is needed to execute your unit tests is to run ```pytest``` in your repository directory.

### Committing

1. Run `git add .` to stage all files
    - You can stage individual files by giving only the paths to those files, the filenames or a pattern, e.g. ```git add your_project_name/**/*.py```
2. Run `Git commit -m "initial commit"` to make your first commit

If you ran pre-commit before doing this and pre-commit works correctly, you should see two checks (black and flake8) passing with no errors.

If there are issues with your code caught by flake8 or by black, the commit will not complete.  
You need to make changes and stage them again using ```git add``` to make sure pre-commit sees them.  
If changes are made by black, these changes are not automatically staged, and you need to add them as well using ```git add```

## Why should I use this?

Regardless of our skill level or best intentions, we all make mistakes when coding.

This template contains a few automated checks against your code that you can use to prevent some mistakes before they are commited, or when they are pushed to the repository.

## What is included in this repository?

- Everything that is included in the [Mini repository](https://github.com/adraismawur/python-template-mini)
  - A [README](#a-readme-file) file
  - A (copyleft) [license](#a-license)
  - A basic [folder structure](#folder-structure)
  - A .gitignore file
- Everything that is included in the [Midi repository](https://github.com/adraismawur/python-template-midi)
  - A pyproject.toml file
  - A requirements.txt file
- A test folder with an example unit test
- A black configuration in the _pyproject.toml_ file
- A flake8 configuration in the .flake8 file
- A pre-commit configuration file (.pre-commit-config.yaml), with the following hooks:
  - Black
  - Flake8
- A GitHub workflow that runs your unit tests every time you push your changes

## Guide to elements introduced here

### Unit testing

A unit test is a small, simple piece of code that tests a unit (usually a specific method) of your code. Unit tests are at their best if they test exactly one condition in exactly one method, but this is usually difficult to achieve.

Unit tests are difficult to write, but when written well can catch instances where a change you have made to your code causes a new issue (regression).
It can also help you first define the ways in which you want your program to behave before writing code (test-driven development).


### Black

[Black](https://github.com/psf/black) is a code formatter. 
Any code base that uses it will have a style that is the same no matter who wrote the code. 
A consistent code style makes it easier to read, maintain and expand on code.


❗ __It is highly recommended that you find a way to automatically apply black whenever you save your code.__

Style and formatting in coding languages can be a topic of contention.
Black has made a couple of choices for you, some of which are explained below.

#### Why does Black use spaces instead of tabs?

Pretty much all of Black's choices come from adhering to the [PEP8 Style guide](https://peps.python.org/pep-0008/).

Wherever it does not, the authors have made their own decisions and are just applying them consistently.

#### Why does Black limit lines to 88 characters?

PEP8 [recommends 79](https://peps.python.org/pep-0008/#maximum-line-length), but Black extends it to reduce average lines of code.

Additional context: 79 is a historical choice (small monitors could commonly support about 80 characters) as well as a practical one; Using a small character limit ensures that you can put multiple files next to each other on common screen sizes, which really helps in comparing files.

### Flake8

Flake8 is a linter.
Linters check your code against a set of rules such that whenever a rule violation is found, a warning or error is raised.

If you want to run flake8 manually, simply use the command ```flake8```.

How Flake8 judges your code is customizable through a configuration file, also included in this repository. See https://flake8.pycqa.org/en/latest/user/configuration.html for more information.

### Pre-commit

[pre-commit](https://pre-commit.com/), a Git [hook](https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks) added to this repository ensures a basic level of code style and prevents some minor issues. 

As the name implies, plugins configured in the configuration will be run before attempting to commit changes to your (local) repository.
Since you (hopefully) commit early and often, pre-commit ensures that your code remains nicely stylized and readable.

❗ __When any hook in pre-commit detects an issue, your commit will fail.__

❗ __When a pre-commit hook modifies your files on commit, it not add (stage) the modification for your current commit!__  
❗ __You have to review and add the modification yourself.__

The following plugins are configured in this template:

#### Black

This applies black on any code in your staged changes (modified files you are about to commit) and modifies the code if any issues are found.

#### Flake8

By default, the flake8 linter pre-commit hook only prevents you from committing code that contains errors, and allows warnings.

#### More git hooks

You can find more plugins at [https://pre-commit.com/hooks.html](https://pre-commit.com/hooks.html).
For more information on how the .pre-commit-config.yaml should look, see [https://pre-commit.com/#plugins](https://pre-commit.com/#plugins).


### Github workflow

TODO

