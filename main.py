from clustering import ProfileClustering
import json
import os

cwd = os.getcwd()
input_data_file_path = cwd + "/input.json"
f = open(input_data_file_path,'r')
input_data = f.read()
f.close()

# update the number of component and number of cluster
number_of_component = 3
number_of_cluster = 3

pc = ProfileClustering(number_of_component,number_of_cluster)
output_data = pc.process(input_data)
file = open('output.json','w')
json.dump(output_data, file)
file.close()