FOCUS_Kernel.obj: FOCUS_Kernel.cu
	nvcc -dc -arch=sm_50 FOCUS_Kernel.cu -o FOCUS_Kernel.obj

FOCUS_Class.obj: FOCUS_Class.cu
	nvcc -dc -arch=sm_50 FOCUS_Class.cu -o FOCUS_Class.obj

FOCUS_Example01.obj: FOCUS_Example01.cu
	nvcc -dc -arch=sm_50 FOCUS_Example01.cu -o FOCUS_Example01.obj

FOCUS_Example02.obj: FOCUS_Example02.cu
	nvcc -dc -arch=sm_50 FOCUS_Example02.cu -o FOCUS_Example02.obj

FOCUS_Example03.obj: FOCUS_Example03.cu
	nvcc -dc -arch=sm_50 FOCUS_Example03.cu -o FOCUS_Example03.obj

FOCUS_Example04.obj: FOCUS_Example04.cu
	nvcc -dc -arch=sm_50 FOCUS_Example04.cu -o FOCUS_Example04.obj

compileExample01: FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example01.obj
	nvcc FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example01.obj -o FOCUS_Example01.exe

compileExample02: FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example02.obj
	nvcc FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example02.obj -o FOCUS_Example02.exe

compileExample03: FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example03.obj
	nvcc FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example03.obj -o FOCUS_Example03.exe

compileExample04: FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example04.obj
	nvcc FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example04.obj -o FOCUS_Example04.exe

all:
	make compileExample01
	make compileExample02
	make compileExample03
	make compileExample04

clean:
	del *.obj *.exe *.exp *.lib