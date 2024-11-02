from jtop import jtop

# Open jtop interface
with jtop() as jetson:
    # Check if board is compatible
    if jetson.ok():
        #print(jetson.power['tot']['power']/1000)
        print(jetson.power['rail']['VDD_CPU_GPU_CV']['power']/1000)
