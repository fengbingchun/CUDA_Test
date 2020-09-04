; reference: https://github.com/brianrhall/Assembly/blob/master/Chapter_9/Program%209.3/x86_64/Program_9.3_NASM.asm

extern _printInt

%macro intAdd 3
   mov rax, [%2]
   add rax, [%3]
   mov [%1], rax
%endmacro

section .data
intA: dq 2
intB: dq 4
intC: dq 3
intD: dq 7
result: dq 0

section .text
global macro_usage
macro_usage:
	push rbp ; set up stack frame, must be alligned

	intAdd result, intA, intB
	mov rdi, qword [result]
	mov rax, 0 ; or can be xor rax, rax, no vector registers in use
	call _printInt
	
	intAdd result, intC, intD
	mov rdi, qword [result]
	mov rax, 0
	call _printInt

	pop rbp ; restore stack
	mov rax, 0 ; normal exit, no error, return value
	ret ; return to operating system
