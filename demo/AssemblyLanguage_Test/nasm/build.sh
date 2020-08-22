#! /bin/bash

nasm -f elf64 -o funset.o funset.asm
ld -e _main -melf_x86_64 -o funset funset.o
#./funset
