; reference: https://github.com/brianrhall/Assembly/blob/master/Chapter_9/Program%209.3/x86_64/Program_9.3_MASM.asm
extrn ExitProcess : proc

_printInt  PROTO C

intAdd MACRO dest, source1, source2
	mov rax, source1
	add rax, source2
	mov dest, rax
ENDM

.DATA
intA QWORD 2
intB QWORD 4
intC QWORD 3
intD QWORD 7
result QWORD 0

.CODE
macro_usage PROC
	push rbp

	intAdd result, intA, intB
	mov rcx, result
	mov rax, 0
	call _printInt

	intAdd result, intC, intD
	mov rcx, result
	mov rax, 0
	call _printInt

	pop rbp
	ret
macro_usage ENDP
END
