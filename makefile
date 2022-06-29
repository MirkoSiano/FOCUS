FOCUS_Kernel.obj: FOCUS_Kernel.cu
	nvcc -dc -arch=sm_50 FOCUS_Kernel.cu -o FOCUS_Kernel.obj

FOCUS_Class.obj: FOCUS_Class.cu
	nvcc -dc -arch=sm_50 FOCUS_Class.cu -o FOCUS_Class.obj

FOCUS_Example01.obj: FOCUS_Example01.cu
	nvcc -dc -arch=sm_50 FOCUS_Example01.cu -o FOCUS_Example01.obj

FOCUS_Example02.obj: FOCUS_Example02.cu
	nvcc -dc -arch=sm_50 FOCUS_Example02.cu -o FOCUS_Example02.obj

compileExample01: FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example01.obj
	nvcc FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example01.obj -o FOCUS_Example01.exe

compileExample02: FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example02.obj
	nvcc FOCUS_Kernel.obj FOCUS_Class.obj FOCUS_Example02.obj -o FOCUS_Example02.exe

all:
	make compileExample01
	make compileExample02

clean:
	del *.obj *.exe *.exp *.lib