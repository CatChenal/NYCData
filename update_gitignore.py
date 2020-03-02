# update_gitignore.py SKELETON

"""
Need: 
I want a function to place in the first cell of my jupyter lab template, which contains frequently used libraries imports & other tidbits so that I can update `.gitignore` from a cell.

Command line solutions:

A. Nix and Mac OS:
A shell recipe to update .gitignore in order to exclude large files (>= 1G) is:
>find . -size +1G | sed 's|^./||g' | cat >> .gitignore

B. Windows
forfiles /S /M * /C “cmd /c if @fsize GEQ 1073741824 echo @path > over_1GB.txt

The Windows forfiles command doc:
```
forfiles: Selects one or more files and runs a command that refers to these files. Usually used for batch and script files.
```

Since my system is windows 10, I want to implement (B).
"""
import os
import subprocesses