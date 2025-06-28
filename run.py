import os


os.system('python prepare_input_data.py')

os.system('python prepare_charts.py')
os.system('python predict_charts.py') 
os.system('python generate_recommendation.py')
