
with open("test.txt", "r") as f:
	lines = f.readlines()
    
for i in range(len(lines)):
    lines[i] = lines[i].strip().split()

loss =[]
pearson = []
MSE = []

for i in range(len(lines)):
    if 'Test' in lines[i] and 'MSE:' in lines[i]:
        MSE.append(lines[i][-1])
        pearson.append(lines[i][9])
        loss.append(lines[i][7]) 
        



print("mse: ",min(MSE), "\npearson: ", max(pearson), "\nloss: ", min(loss))