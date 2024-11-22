# Contributing to the Repository: 

## Gain the habit of creating new branches:
Ideally in the main branch, we want to keep the stable, fully-functioning and bug-free codebase. If you plan to add a new method or try fixing something first checkout to a new branch, prototype there and make sure it is working then we can merge it to the main branch.

## Commit Key Words:
Please use the following keywords/prefixes when committing so that the type of the commit can be easily seen by the others.

Note, if the file ends with `_acc.py` it implies the script has DeepSpeed training through Accelerate.
- `feat:` adding a new feature
- `fix:` fixing a bug in the code.
- `config:` any type of change in the configs or bash files.
- `refactor:` re-arranging the code snippets, getting rid of unnecessary parts etc.

One example commit message can be written as follows: `feat: adding I-LoRA`